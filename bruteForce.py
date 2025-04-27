from parameters import SCREW_SPACE_DIM, EUC_SPACE_DIM, EPSILON, BF_ITER_MAX, BF_PRINT_ACTIONS, BF_MAG_MUL
from messages import MSG_UNSOLVED, MSG_UNIMPLEMENTED
from subspace import Subspace
from numSolvers import solveCVXPY
from utils import L0Filter, nullspace, invertMap, copyToClipboard
from graph import Graph, CMGraph, StageTwist
from model import CompliantJoint

import numpy as np
import screwVectors as sv

class Solution(object):
    def __init__(self, graph, velocity):

        self.graph = graph
        self.vel = velocity
        self.nodeTwists = None # m x n x 6
        self.edgeTwists = None # m x e x 6
        self.jft = None # 6 x 6e

        self.status = {}

        if(velocity is not None):
            self._initTwists()
            self.count = 0
    
    def _initTwists(self):
        
        self.jft = self.graph.fullJFT()

        nodeTwists, edgeTwists = [], []
        for modeId in range(len(self.graph.ioSpecs)):
            # get each edge's twist vector
            edgeMask = np.repeat(np.identity(self.graph.edgeCount),
                                    SCREW_SPACE_DIM, axis=1) # e x 6e
            
            edgeTwist = edgeMask @ (self.vel[:, modeId].reshape(1, -1) * self.jft).T
            edgeTwists += [edgeTwist]

            # sum and find each node's displacement
            Ps = self.graph.nodePaths(self.graph.ioSpecs[modeId].groundId) # n x e
            nodeTwist = Ps @ edgeTwist
            nodeTwists += [nodeTwist]

        self.nodeTwists = np.stack(nodeTwists, axis=0)
        self.edgeTwists = np.stack(edgeTwists, axis=0)

    def clone(self):

        return Solution(self.graph.clone(), np.copy(self.vel))

    def printChainVelocity(self, printEdgeDOF=False, printAll=False):

        for modeId in range(len(self.graph.ioSpecs)):
            print("mode", modeId)
            chains = self.graph.specChains(modeId, skipZero=not printAll, trainOnly=True)
            for chain in chains:
                print("\t** chain", chain[0], '-', chain[-1])
                for i in range(len(chain)):
                    sId = chain[i]
                    nT = self.nodeTwists[modeId, sId]
                    nT[nT == 0] = 0
                    print("\tnode", sId, ',', nT)
                    print()
                    if(i < (len(chain) - 1)):
                        eId = chain[i + 1]
                        edgeId = self.graph.edgeBetweenNodes(sId, eId)
                        dir = 1 if self.graph.C[edgeId, sId] == -1 else -1
                        eT = self.edgeTwists[modeId, edgeId] * dir
                        eT[eT == 0] = 0
                        print("\t\tedge", edgeId, ',',  eT)
                        
                        if(not printEdgeDOF): continue

                        print("\t\tDOF")
                        dofs = self.edgeTwists[:, edgeId] * dir
                        dofs[dofs == 0] = 0
                        for dof in dofs: print("\t\t\t", sv.unitize(dof))
                        print()
                print()

    def freedomAnalysis(self, simplifyTwists=True, printFullReport=False):

        print("IO decoupling check")

        configModes = self.graph.configGroups()
        for cId in configModes:
            mIds = list(configModes[cId])
                
            jfts, dims = [], []
            for eId in range(self.graph.edgeCount):
                twists = self.edgeTwists[mIds, eId]
                
                if(simplifyTwists):
                    edgeDOFs = Subspace.simplifySpace(twists, True)
                else:
                    edgeDOFs = twists
                
                jfts += [edgeDOFs]
                dims += [len(edgeDOFs)]

            dims = np.asarray(dims)
            JFT = np.concatenate(jfts, axis=0).T # 6 * n

            eqs = []
            Qs = self.graph.Qloops()
            for loop, _ in Qs:
                factor = np.repeat(loop, dims)
                factored = JFT * factor
                eqs += [factored]
            
            eqs = np.concatenate(eqs, axis=0)
            X = nullspace(eqs) # m * n
            
            if(printFullReport): print("X", X.shape)
            factor = np.repeat(self.graph.path(1, 0), dims)

            for mId in mIds:
                spec = self.graph.ioSpecs[mId]
                A, B = [], []
                for inp in spec.input:
                    if(printFullReport): print(inp.id, ':', inp.twist)
                    factor = np.repeat(self.graph.path(spec.groundId, inp.id), dims)

                    Ainp = (JFT * factor) @ X.T
                    Binp = inp.twist
                    A += [Ainp]
                    B += [Binp]
                
                A = np.concatenate(A, axis=0)
                B = np.concatenate(B, axis=0)
                sol = Subspace.solveLinearSys(A, B)
                if(printFullReport): print(sol)

                decoup = []
                for out in spec.output:
                    if(printFullReport): print(out.id, ':')
                    factor = np.repeat(self.graph.path(spec.groundId, out.id), dims)
                    ref = (JFT * factor) @ X.T @ sol.refPt
                    ref[abs(ref) < EPSILON] = 0
                    if(sol.spanRank == 0):
                        span = np.zeros((0, SCREW_SPACE_DIM))
                    else:
                        span = ((JFT * factor) @ X.T @ sol.spanSpace.T).T
                        span[abs(span) < EPSILON] = 0
                    if(np.linalg.norm(span) > EPSILON): decoup += [out.id]

                    if(printFullReport): print(ref)
                    if(printFullReport): print(span)
                
                if(not printFullReport):
                    if(len(decoup) == 0): print("\t Mode", mId, "Ok")
                    else:  print("\t Mode", mId, "decoupled stages:", decoup)
                if(printFullReport): print()

    def edgeFreedomDeg(self):

        degs = np.zeros(self.graph.edgeCount, dtype=np.int32)

        for i in range(self.graph.edgeCount):
            vels = self.edgeTwists[:, i]
            consDeg = len(nullspace(vels))
            degs[i] = SCREW_SPACE_DIM - consDeg

        return degs

    def isEdgeRigid(self):

        edgeTwistNorm = np.linalg.norm(self.edgeTwists, axis=-1)
        isRigid = np.all(edgeTwistNorm < EPSILON, axis=0)

        return isRigid

    def outputJSON(self, copy=False):

        json = self.graph.outputJSON()

        json["displacement"] = self.nodeTwists.tolist()
        json["jointScrew"] = self.edgeTwists.tolist()
        json["isRigid"] = (np.linalg.norm(self.edgeTwists, axis=-1) < EPSILON).tolist()
        json["fixedIds"] = [m.groundId for m in self.graph.ioSpecs]
        
        json["IOstageIds"] = []
        for mId in range(len(self.graph.ioSpecs)):
            ioIds = [mo.id for mo in self.graph.ioSpecs[mId].iterIOs()]
            json["IOstageIds"] += [ioIds]

        if(copy): copyToClipboard(str(json), "displacement info")

        return json

    def genModelingInstructions(self):
        
        instructions = []
        if(self.status["valid"] and self.status["complete"]):
            for i in range(self.graph.edgeCount):
                twists = self.edgeTwists[:,i,:]
                center = self.graph.edgeCenter(i)
                joint = CompliantJoint(twists, center, id=i)
                info = joint.genInfo()
                instructions += [info]
        
        return instructions

class TopologySimplification(object):
    name = "simplification"
    statusCode = 2

    def __init__(self):
        
        self.isRigid = None
    
    def reset(self):
        self.__init__()

    def examine(self, sol: Solution):

        isTodo = self.check(sol)
        
        if(not isTodo): return sol.graph, False
        if(BF_PRINT_ACTIONS): print("\t Simplify Topology")

        newGraph = self.autofix(sol)
        
        return newGraph, True
    
    def check(self, sol: Solution):
        
        isRigid = sol.isEdgeRigid()

        self.isRigid = isRigid
        
        return np.any(isRigid)
    
    def report(self, sol: Solution):

        # initial grouping
        groups = self._groupRaw(sol.graph, self.isRigid)
        gIds, nIds = np.nonzero(groups)

        info = {}
        groupMap = {}
        for i in range(len(gIds)):

            groupMap[int(gIds[i])] = groupMap.get(int(gIds[i]), []) + [int(nIds[i])]

        info["map"] = groupMap

        return info
    
    def fix(self, sol: Solution, fix: dict):

        mode = fix["mode"]
        
        if(mode == "auto"):
            newGraph = self.autofix(sol)
        elif(mode == "semi"):
            newGraph = self._semiFix(sol, fix)
        elif(mode == "manual"):
            newGraph = self._manualFix(sol, fix)

        return newGraph
    
    def autofix(self, sol: Solution):

        # simplify graph components
        groups, newEdges = self._groupNodes(sol.graph, self.isRigid)
        newC = self._groupC(sol.graph, groups, newEdges)
        newN = self._groupN(sol.graph, groups)
        newSpecs = self._groupSpecs(sol.graph, groups)
        newE = self._groupE(newN, newC)

        # rebuild graph
        newGraph = CMGraph(C=newC, N=newN, E=newE, ioSpecs=newSpecs)
        
        return newGraph
    
    def _semiFix(self, sol, fix):

        groups = self._semiautoGrouping(sol, fix)
        newEdges = self._supplementEdges(sol.graph, groups)

        newC = self._groupC(sol.graph, groups, newEdges)
        newN = self._groupN(sol.graph, groups)
        newSpecs = self._groupSpecs(sol.graph, groups)
        newE = self._groupE(newN, newC)

        # rebuild graph
        newGraph = CMGraph(C=newC, N=newN, E=newE, ioSpecs=newSpecs)
        
        return newGraph
    
    def _manualFix(self, sol, fix):
        
        groups = self._manualGrouping(sol, fix)
        newEdges = self._supplementEdges(sol.graph, groups)

        newC = self._groupC(sol.graph, groups, newEdges)
        newN = self._groupN(sol.graph, groups)
        newSpecs = self._groupSpecs(sol.graph, groups)
        newE = self._groupE(newN, newC)

        # rebuild graph
        newGraph = CMGraph(C=newC, N=newN, E=newE, ioSpecs=newSpecs)
        
        return newGraph
    
    def _manualGrouping(self, sol, fix):

        # convert to id map
        nomination = set()
        inv = {}
        for nId in fix["groups"]:
            nomination.add(int(nId))
            gId = fix["groups"][nId]
            inv[gId] = inv.get(gId, []) + [int(nId)]
        
        groups = []
        for nId in range(sol.graph.nodeCount):
            if(nId in nomination): continue

            vec = np.zeros(sol.graph.nodeCount, dtype=np.int32)
            vec[nId] = 1

            groups += [vec]

        for gId in inv:
            vec = np.zeros(sol.graph.nodeCount, dtype=np.int32)
            vec[inv[gId]] = 1

            groups += [vec]
        
        groups = np.stack(groups, axis=0)

        return groups
    
    def _semiautoGrouping(self, sol, fix):

        groups = self._groupRaw(sol.graph, self.isRigid)

        if(len(fix["preserved"]) == 0): return groups

        groups[:, fix["preserved"]] = 0
        groups = groups[np.any(groups != 0, axis=1)] # remove all zero rows

        
        indi = []
        for nId in fix["preserved"]:
            vec = np.zeros(sol.graph.nodeCount, dtype=np.int32)
            vec[nId] = 1

            indi += [vec]

        indi = np.stack(indi, axis=0)

        groups = np.concatenate([groups, indi], axis=0)

        return groups
    
    def _groupNodes(self, graph, isRigid):
        
        # initial grouping
        groups = self._groupRaw(graph, isRigid)
        newEdges = self._supplementEdges(graph, groups)
        
        return groups, newEdges

    def _groupRaw(self, graph, isRigid):

        # collapse edges
        zeroEdgeIds = np.argwhere(isRigid).flatten()
        rigidC =  graph.C[zeroEdgeIds]
        
        # group bodies
        groupGraph = Graph(C=rigidC).unsigned()
        groups = groupGraph.subGraphGrouping()
        
        return groups

    def _supplementEdges(self, graph, groups):
        
        oldTopoDist = self._oldTopoDist(graph, groups)
        newGraph = Graph(C=self._groupC(graph, groups))
        newA = np.abs(newGraph.A)
        newSpecs = self._groupSpecs(graph, groups)
        
        for spec in newSpecs:
            inps = spec.enumIds(inp=True, out=False)
            outs = spec.enumIds(inp=False, out=True)

            trainNodes = np.zeros(len(newA), dtype=np.int32)

            # gear train #TODO: add heuristics to determine order of gear train analysis
            for i in range(len(inps)):
                inpCur = inps.pop(i)
                for j in range(len(outs)):
                    outCur = outs.pop(j)

                    zeroOut = [o.id for o in spec.output if not o.isZero] # zero output will sink

                    exclusion = inps + [spec.groundId] + zeroOut # gear trains passing through input will get attenuated
                    newA, pathNodes = self._addEdge(newA, inpCur, outCur, oldTopoDist, exclusion)
                    trainNodes += pathNodes

                    outs.insert(j, outCur)
                inps.insert(i, inpCur)

            # input output node train
            ios = inps + outs
            trainNodes[ios] = 0
            trainNodes = np.where(trainNodes != 0)[0].tolist()

            for i in range(len(ios)):
                ioCur = ios.pop(i)

                exclusion = ios + trainNodes
                newA, _ = self._addEdge(newA, spec.groundId, ioCur, oldTopoDist, exclusion)
            
                ios.insert(i, ioCur)

        newA = np.triu(newA) + np.tril(newA) * -1

        deltaA = newA - newGraph.A
        newEdges = np.triu(deltaA)
        start, end = np.where(newEdges != 0)
        newC = np.zeros((len(start), len(newA)))
        indices = np.arange(len(start), dtype=np.int32)
        newC[indices, start] = -1
        newC[indices, end] = 1

        return newC
    
    def _oldTopoDist(self, graph, groups):

        A = np.copy(graph.A)
        A[A == -1] = 1
        dist = Graph.crossDist(A)
        newA = np.zeros((len(groups), len(groups)), dtype=np.int32)

        for i in range(len(groups)):
            for j in range(len(groups)):

                mask = groups[i].reshape((-1, 1)) * groups[j]
                dists = mask * dist
                if(np.all(dists == 0)): d = 0
                else:
                    dists = np.unique(dists)
                    if(dists[0] == 0): d = dists[1]
                    else: d = dists[0]

                newA[i][j] = d

        np.fill_diagonal(newA, 0)

        return newA

    def _groupC(self, graph, groups, addEdges=[]):

        edges = graph.C
        if(len(addEdges) > 0 and groups.shape[1] == addEdges.shape[1]):
            edges = np.concatenate([edges, addEdges], axis=0)

        newEdges = np.matmul(graph.C, groups.T)

        if(len(addEdges) > 0 and groups.shape[0] == addEdges.shape[1]):
            newEdges = np.concatenate([newEdges, addEdges], axis=0)

        sId = np.where(newEdges == -1)
        eId = np.where(newEdges == 1)
        revFlag = sId[0][np.where(sId[-1] > eId[-1])]
        newEdges[revFlag] *= -1
        newEdges  = np.unique(newEdges, axis=0)
        newEdges = newEdges[np.linalg.norm(newEdges, axis=-1) > 0]
        
        return newEdges

    def _groupN(self, graph, groups):

        # get all current node centers
        ioMask = graph.ioStageMask(combine=True)
        centers = graph.N[:, :EUC_SPACE_DIM]

        # if one or more IO node in group, force IO stage center (single:force, multiple:average)
        centerMasked = ioMask.reshape(-1, 1) * centers
        stageCount = np.sum(ioMask * groups, axis=-1).reshape(-1, 1)
        stageCount[stageCount == 0] = 1 # avoid div-0
        centerForced = np.matmul(groups, centerMasked) / stageCount

        # for the rest, just average
        stageCount = np.sum(groups, axis=-1).reshape(-1, 1)
        stageCount[stageCount == 0] = 1 # avoid div-0
        centerOther = np.matmul(groups, centers) / stageCount

        # remix center points
        pickerMask = np.sum(groups * ioMask, axis=-1).reshape(-1, 1)
        pickerMask[pickerMask > 0] = 1
        newCenters = pickerMask * centerForced + (1 - pickerMask) * centerOther
        
        # find new spec ids
        BCs = graph.N[:, EUC_SPACE_DIM:]
        newBCs = np.matmul(groups, BCs)
        newN = np.concatenate([newCenters, newBCs], axis=-1)

        return newN
    
    def _groupSpecs(self, graph, groups):

        newSpecs = []
        for spec in graph.ioSpecs:
            dummy = spec.clone()
            dummy.groundId = int(np.where(groups[:, dummy.groundId] == 1)[0][0])

            for mo in dummy.input + dummy.output:
                mo.id = int(np.where(groups[:,mo.id] == 1)[0][0])
            
            newSpecs += [dummy]
        
        return newSpecs

    def _groupE(self, newN, newC):

        sId = np.where(newC == -1)[1]
        eId = np.where(newC == 1)[1]

        sPt = newN[sId, :EUC_SPACE_DIM]
        ePt = newN[eId, :EUC_SPACE_DIM]

        Ec = (sPt + ePt) * .5
        Ew = np.zeros((newC.shape[0], SCREW_SPACE_DIM))

        newE = np.concatenate([Ec, Ew], axis=-1)

        return newE

    def _addEdge(self, A, start, end, cost, exclusion=[]):

        A = np.copy(A)

        # group into clouds
        groups = Graph._groupSubgraphs(A, exclusion=exclusion)
        
        # get dense graph and edge costs
        denseA = 1 - np.identity(len(A), dtype=np.int32)
        cost = np.copy(cost).astype(np.float64)
        interGroupMask = groups.reshape(len(groups), -1, 1) *\
                        groups.reshape(len(groups), 1, len(A))
        mask = np.sum(interGroupMask, axis=0)
        invMask = 1 - mask

        denseA = denseA * invMask + A * mask
        cost *= invMask
        cost = cost * (1 - A) + EPSILON * A

        # find  minimum link plan
        prev, _ = Graph. dijkstra(denseA, start, cost, exclusion)
        pathNodes = Graph._trackPath(prev, start, end)
        pathMask = np.zeros(len(A), dtype=np.int32)
        pathMask[pathNodes] = 1
        path = np.asarray(pathNodes)
        
        # find new edges
        sId = path[:-1]
        eId = path[1:]
        adj = A[sId, eId]
        missingEdge = np.where(adj == 0)
        sMissing, eMissing = sId[missingEdge], eId[missingEdge]
        
        # add edge between clouds
        A[sMissing, eMissing] = 1
        A[eMissing, sMissing] = 1
        
        return A, pathMask

class FixLinearArc(object):
    name = "linear"
    statusCode = 2

    def __init__(self):

        self.deg = None
        self.keep = None
        self.pivs = None
        
    def reset(self):
        self.__init__()

    def examine(self, sol: Solution):

        isTodo = self.check(sol)
        
        if(not isTodo): return sol.graph, False
        if(BF_PRINT_ACTIONS): print("\t Remove linear edges and nodes")

        newGraph = self.autofix(sol)
        
        return newGraph, True

    def check(self, sol: Solution):
        
        deg = sol.graph.degree()
        ioMask = sol.graph.ioStageMask(combine=True)
        deg[ioMask != 0] = 0

        keep = np.where(deg != 2)[0]
        pivs = np.where(deg == 2)[0]

        self.keep = keep
        self.pivs = pivs

        return not len(pivs) == 0
    
    def report(self, sol: Solution):

        info = {}
        info["keep"] = [int(nId) for nId in self.keep]
        info["pivs"] = [int(nId) for nId in self.pivs]

        return info
    
    def fix(self, sol: Solution, fix: dict):

        mode = fix["mode"]
        
        if(mode == "auto"):
            newGraph = self.autofix(sol)
        elif(mode == "manual"):
            newGraph = self._manualFix(sol, fix)
            
        return newGraph

    def autofix(self, sol:Solution):

        if(len(self.pivs) == 0):
            return sol.graph.clone()

        newC, newE = self._updateGraph(sol)
        newN = sol.graph.N[self.keep]
        newSpecs = self._updateSpecs(sol)

        newGraph = CMGraph(C=newC, N=newN, E=newE, ioSpecs=newSpecs)

        return newGraph
    
    def _manualFix(self, sol, fix):

        pivs = sorted(fix["pluck"])
        mask = np.ones(sol.graph.nodeCount, dtype=np.int32)
        mask[pivs] = 0
        mask[self.keep] = 1 # overrule
        
        keep = np.where(mask == 1)[0]
        pivs = np.where(mask != 1)[0]
        
        self.keep = keep
        self.pivs = pivs
        
        return self.autofix(sol)

    def _updateGraph(self, sol: Solution):

        mask = np.zeros(sol.graph.nodeCount, dtype=np.int32)
        mask[self.pivs] = 1

        factored = np.abs(sol.graph.C) @ mask
        factored = factored.reshape(-1)
        keepEdge = factored == 0
        keepC = sol.graph.C[keepEdge]
        keepE = sol.graph.E[keepEdge]
        suppC, suppE = [], []
        for start, end in self._newEdges(sol):
            c = np.zeros(sol.graph.nodeCount, dtype=np.int32)
            c[start] = -1
            c[end] = 1
            if(any((keepC @ c) == 2)): continue
            suppC += [c]
            suppE += [sol.graph.newEdgeFromNodeIds(start, end)]
        
        if(len(suppC) == 0): return keepC[:, self.keep], keepE
        
        suppC = np.stack(suppC, axis=0)
        suppE = np.stack(suppE, axis=0)

        newC = np.concatenate([keepC, suppC], axis=0)
        newC = newC[:, self.keep]
        newE = np.concatenate([keepE, suppE], axis=0)
        
        return newC, newE

    def _newEdges(self, sol:Solution):

        pivs = np.copy(self.pivs).tolist()
        piv = pivs.pop(0)
        visited = set([piv])
        curStart, curEnd = sol.graph.adjNodes(piv)
        chains = []

        while(True):

            if(curStart in pivs): # expanding head
                nextStart, nextEnd = sol.graph.adjNodes(curStart)
                visited.add(curStart)
                pivs.remove(curStart)
                if(nextStart not in visited): curStart = nextStart
                elif(nextEnd not in visited): curStart = nextEnd
            elif(curEnd in pivs): # expanding tail
                nextStart, nextEnd = sol.graph.adjNodes(curEnd)
                visited.add(curEnd)
                pivs.remove(curEnd)
                if(nextStart not in visited): curEnd = nextStart
                elif(nextEnd not in visited): curEnd = nextEnd
            else: # cannot expand further, group chain
                if(curStart > curEnd): curStart, curEnd = curEnd, curStart
                chains += [(curStart, curEnd)]
                if(len(pivs) > 0):
                    piv = pivs.pop(0)
                    visited = set([piv])
                    curStart, curEnd = sol.graph.adjNodes(piv)
                else:
                    break

        return chains

    def _updateSpecs(self, sol:Solution):

        idMap = np.arange(sol.graph.nodeCount, dtype=np.int32)
        idSource = idMap[self.keep]
        idTarg = np.arange(len(idSource))
        idMap[idSource] = idTarg
        idMap[self.pivs] = -1
        newSpecs = self._reSpec(sol.graph, idMap)

        return newSpecs
    
    def _reSpec(self, graph, idMap):

        specs = []
        for spec in graph.ioSpecs:
            dummy = spec.clone()

            dummy.groundId = idMap[spec.groundId]

            for mo in dummy.input + dummy.output:
                mo.id = idMap[mo.id]
            
            specs += [dummy]

        return specs

class EdgeElimination(object):
    name = "edgeCull"
    statusCode = 2

    def __init__(self):

        self.mask = None
        self.edgesToKeep = None
        self.edgesToDel = None

    def reset(self):
        self.__init__()
    
    def examine(self, sol: Solution):

        isTodo = self.check(sol)
        
        if(not isTodo): return sol.graph, False
        if(BF_PRINT_ACTIONS): 
            print("\t Eliminate redundant edges")
            print("\t\t Edges to remove:", np.where(self.mask == 0)[0])

        newGraph = self.autofix(sol)
        
        return newGraph, True
    
    def check(self, sol: Solution):
        
        # 1. find edges using all 6 DOF
        if(sol.vel.shape[0] >= SCREW_SPACE_DIM):
            edgeDegs = sol.edgeFreedomDeg()
            fullFreedomEdgeMask = edgeDegs >= SCREW_SPACE_DIM
        else:
            fullFreedomEdgeMask = np.full(sol.graph.edgeCount, False)
        
        # 2. find io direct connect decoupled edges
        # 2.1 find edges not used by transmission chains
        mask = self._edgesTraffic(sol.graph)
        unused = mask == 0

        # 2.2 find edges whose both ends land on io stages
        ioMask = sol.graph.ioStageMask(combine=True)
        isStartIO = ioMask[sol.graph.edgeStartId()] > 0
        isEndIO = ioMask[sol.graph.edgeEndId()] > 0
        ioEdgeMask = np.logical_and(isStartIO, isEndIO)
        decoupIoEdgeMask = np.logical_and(ioEdgeMask, unused)

        # 3. check if modifications are needed
        edgesToDel = np.logical_or(fullFreedomEdgeMask, decoupIoEdgeMask)
        edgesToKeep = np.logical_not(edgesToDel)

        self.mask = mask
        self.edgesToKeep = np.where(edgesToKeep)[0]
        self.edgesToDel = np.where(edgesToDel)[0]

        return not np.all(edgesToKeep)
    
    def report(self, sol: Solution):

        info = {}
        info["keep"] = [int(nId) for nId in self.edgesToKeep]
        info["pivs"] = [int(nId) for nId in self.edgesToDel]

        return info
    
    def fix(self, sol: Solution, fix: dict):

        mode = fix["mode"]
        
        if(mode == "auto"):
            newGraph = self.autofix(sol)
        elif(mode == "manual"):
            newGraph = self._manualFix(sol, fix)
        
        return newGraph

    def _manualFix(self, sol, fix):

        pivs = fix["pluck"]
        mask = np.ones(sol.graph.edgeCount, dtype=np.int32)
        mask[pivs] = 0
        mask[self.edgesToKeep] = 1

        self.edgesToKeep = np.where(mask == 1)[0]
        self.edgesToDel = np.wehre(mask == 0)[0]

        return self.autofix(sol)

    def autofix(self, sol: Solution):

        newGraph = sol.graph.clone()
        newC = np.copy(sol.graph.C)[self.edgesToKeep]
        newN = np.copy(sol.graph.N)
        newE = np.copy(sol.graph.E)[self.edgesToKeep]
        newSpecs = [spec.clone() for spec in sol.graph.ioSpecs]

        newGraph = CMGraph(C=newC, N=newN, E=newE, ioSpecs=newSpecs)

        return newGraph

    def _edgesTraffic(self, graph):

        # find non-sinking io pairs
        effective, ineffective = {}, {}

        for mId in range(len(graph.ioSpecs)):
            spec = graph.ioSpecs[mId]     
            for inp in spec.input:            
                for out in spec.output:

                    isSink = inp.isZero or out.isZero
                    tar = ineffective if isSink else effective

                    tar[mId] = tar.get(mId, []) + [(inp.id, out.id)]
        
        # find effective edges
        mask = np.zeros(graph.edgeCount, dtype=np.int32)
        for mId in effective:
            for start, end in effective[mId]:
                track = graph._specIoPairchain(start, end, graph.ioSpecs[mId].groundId)
                P = graph._makePathVec(track)
                P = np.abs(P)
                mask += P
            
        return mask

class FixDecouple(object):
    name = "decouple"
    statusCode = 1

    def __init__(self):

        self.decoup = None

    def reset(self):
        self.__init__()
    
    def examine(self, sol: Solution):
        
        isTodo  = self.check(sol)
        
        if(not isTodo): return sol.graph, False
        if(BF_PRINT_ACTIONS): print("\t Fix Decouple")

        newGraph = self.autofix(sol)

        return newGraph, True
        
    def check(self, sol: Solution):

        decoup = {}
        for modeId in range(len(sol.graph.ioSpecs)):
            cond = FixDecouple._checkModalDecouple(sol, modeId)

            chains = sol.graph.specChains(modeId, skipZero=True)

            for chain in chains:
                mask = sol.graph._makePathVec(chain)
                pivots = np.where(np.logical_and(cond, mask != 0))[0]
                
                for piv in pivots:
                    decoup[modeId] = decoup.get(modeId, []) + [piv]

        # simplify
        for key in decoup: decoup[key] = list(set(decoup[key]))

        self.decoup = decoup

        return len(decoup) != 0
    
    def report(self, sol: Solution):

        info = {}
        info["edges"] = self._reportDecoupEdges()
        
        return info
    
    def _reportDecoupEdges(self):

        info = {}
        for modeId in self.decoup:
            info[str(modeId)] = [int(eId) for eId in self.decoup[modeId]]
        
        return info
    
    def fix(self, sol: Solution, fix: dict):

        mode = fix["mode"]
        
        if(mode == "auto"):
            newGraph = self.autofix(sol)
        elif(mode == "semi"):
            newGraph = self._semiFix(sol, fix)
        elif(mode == "manual"):
            newGraph = self._manualFix(sol, fix)

        return newGraph
    
    def _manualFix(self, sol, fix):

        return sol.graph.clone()

    def _semiFix(self, sol, fix):

        return sol.graph.clone()
    
    def autofix(self, sol: Solution):

        newGraph = sol.graph.clone()
        toBias, toSplit = FixDecouple._planDecoupFix(sol, self.decoup)
        if(BF_PRINT_ACTIONS and len(toBias) > 0): print("\t\t Nodes to bias (mode: id):", toBias)
        if(BF_PRINT_ACTIONS and len(toSplit) > 0): print("\t\t Edges to split (mode:id):", toSplit)

        FixDecouple._biasNodes(sol, newGraph, toBias, additive=False)
        FixDecouple._splitEdges(sol, newGraph, toSplit)

        return newGraph

    # common methods

    @staticmethod
    def _planDecoupFix(sol, decoup):

        toBias, toSplit = {}, {} # mode ID -> node/edge ID

        for modeId in decoup:
            pivIds = set(decoup[modeId])
            modeIOs = set(sol.graph.ioSpecs[modeId].enumIds())

            for pivId in pivIds:
                sId = sol.graph.edgeStartId(pivId)
                eId = sol.graph.edgeEndId(pivId)

                curNodePool = toBias.get(modeId, set())
                curEdgePool = toSplit.get(modeId, set())
                
                if(sId in curNodePool or\
                   eId in curNodePool or\
                   pivId in curEdgePool): continue
                
                if(sId not in modeIOs):
                    curNodePool.add(sId)
                    toBias[modeId] = curNodePool
                elif(eId not in modeIOs):
                    curNodePool.add(eId)
                    toBias[modeId] = curNodePool
                else:
                    curEdgePool.add(pivId)
                    toSplit[modeId] = curEdgePool
        
        return toBias, toSplit

    @staticmethod
    def _checkModalDecouple(sol, modeId):

        edgeTwist = sol.edgeTwists[modeId]
        nodeTwist = sol.nodeTwists[modeId]
        
        edgeTwistNorm = np.linalg.norm(edgeTwist, axis=-1)
        nodeTwistNorm = np.linalg.norm(nodeTwist, axis=-1)
        
        # check node-edge-node displacement alignment
        start = sol.graph.edgeStartId()
        end = sol.graph.edgeEndId()

        inpDot = np.abs(np.sum(nodeTwist[start] * edgeTwist, axis=-1))
        outDot = np.abs(np.sum(nodeTwist[end] * edgeTwist, axis=-1))
        
        isSinker = np.logical_or(nodeTwistNorm[start] <= EPSILON, nodeTwistNorm[end] <= EPSILON)
        isRigid = edgeTwistNorm <= EPSILON
        bypass = np.logical_or(isSinker, isRigid)

        inpCheck = inpDot == edgeTwistNorm * nodeTwistNorm[start] # input and edge aligned
        outCheck = outDot == edgeTwistNorm * nodeTwistNorm[end] # output and edge aligned
        
        # find pivotal edges that require splitting
        condition = np.logical_and(np.logical_or(inpCheck, outCheck), np.logical_not(bypass))
        
        return condition
    
    @staticmethod
    def _applyBias(sol, newGraph, target, unitTwist, additive=True, tarMag=None, magMul=None):

        configId, modeId, pivId = target

        direction = FixDecouple._findDirecton(sol, newGraph, modeId, pivId, unitTwist)
        magnitude = FixDecouple._findMagnitude(sol, newGraph, pivId, unitTwist, tarMag)
        magnitude *= BF_MAG_MUL

        twist = unitTwist * magnitude * direction
        if(tarMag is None): twist *= magMul

        if(BF_PRINT_ACTIONS): print("\t\t Mode %d (config %d) node %d: "%(modeId, configId, pivId), twist)
        if(additive and pivId < len(sol.nodeTwists)): twist += sol.nodeTwists[modeId, pivId]
        cons = StageTwist(newGraph, modeId, pivId, twist)
        newGraph.cons += [cons]

    @staticmethod
    def _pickBias(sol, available, pivId, pivModes, order=None, mode='n'):
        
        if(order is None): order=["orthoFirst", "buildCost"]
        
        availSorted = FixDecouple._costSortDOFs(sol, available, pivId, pivModes, order=order, mode=mode)
        
        picked = availSorted[:len(pivModes)]
        if(len(picked) < len(pivModes)):
            supp = np.zeros((len(pivModes) - len(picked), SCREW_SPACE_DIM))
            picked = np.concatenate([picked, supp], axis=0)

        return picked
    
    @staticmethod
    def _costSortDOFs(sol, vecs, pivId, pivModes, order=["orthoFirst"], mode='n'):
        # order: "rotationFirst", "orthoFirst", "buildCost", "zFirst"
        
        if(len(order) == 0 or len(vecs) <= 1):
            return vecs
        
        if(order[0] == "buildCost"):
            if(mode == 'n'): center = sol.graph.nodeCenter(pivId)
            else: center = sol.graph.edgeCenter(pivId)
            costs = FixDecouple._calBuildCost(sol, vecs, center)
            sortedIds = np.argsort(costs)
            vecs = vecs[sortedIds]
            return vecs

        if(order[0] == "rotationFirst"):
            mask = sv.isRot(vecs)
        elif(order[0] == "translationFirst"):
            mask = sv.isTrans(vecs)
        elif(order[0] == "orthoFirst"):
            if(mode == 'n'): scopedVecs = sol.nodeTwists[pivModes, pivId]
            else: scopedVecs = sol.edgeTwists[pivModes, pivId]
            axes, _, _, _ = sv.decompose(scopedVecs)
            rotIndex = axes @ vecs[:, :EUC_SPACE_DIM].T
            rotOrth = np.all(rotIndex == 0, axis=0)
            isRot = np.linalg.norm(vecs[:, :EUC_SPACE_DIM], axis=-1) > EPSILON
            transIndex =  axes @ vecs[:, EUC_SPACE_DIM:].T
            transOrtho = np.all(transIndex == 0, axis=0)
            mask = np.logical_or(np.logical_and(rotOrth, isRot), 
                                 np.logical_and(transOrtho, np.logical_not(isRot)))
        elif(order[0] == "zFirst"):
            rotZindex = np.abs(vecs[:, :EUC_SPACE_DIM] @ np.asarray([0, 0, 1]))
            isRot = np.linalg.norm(vecs[:, :EUC_SPACE_DIM], axis=-1) > EPSILON
            rotZflag = np.abs(rotZindex - 1) < EPSILON
            transZindex = np.abs(vecs[:, EUC_SPACE_DIM:] @ np.asarray([0, 0, 1]))
            transZflag = np.abs(transZindex - 1) < EPSILON
            mask = np.logical_or(np.logical_and(rotZflag, isRot), 
                                 np.logical_and(transZflag, np.logical_not(isRot)))
        else:
            raise Exception(MSG_UNIMPLEMENTED)

        firstVecs = vecs[mask]
        secondVecs = vecs[np.logical_not(mask)]
        first = FixDecouple._costSortDOFs(sol, firstVecs, pivId, pivModes, order=order[1:], mode=mode)
        second = FixDecouple._costSortDOFs(sol, secondVecs, pivId, pivModes, order=order[1:], mode=mode)
        sorted = np.concatenate([first, second], axis=0)

        return sorted
    
    @staticmethod
    def _calBuildCost(sol, vecs, center):
        
        # find available vectors that are not used by any mode
        axis, rotMag, transMag, _ = sv.decompose(vecs)
        rotMag = rotMag.reshape(-1)
        transMag = transMag.reshape(-1)

        # find grounding
        closestGroundId = FixDecouple._closestGround(sol.graph, center)
        gVec = center - sol.graph.nodeCenter(closestGroundId)
        gDist = np.linalg.norm(gVec)
        gVecUnit = gVec / gDist

        # calculate cost based on grounding difficulty
        dirDot = np.abs(np.dot(axis, gVecUnit))
        rotMag = rotMag.reshape(-1)
        transMag = transMag.reshape(-1)
        costRot = (1 - dirDot) / gDist # minimal cost when rotation axis is aligned
        costTrans = dirDot / (gDist * .5) # minimal cost when perpendicular
        costRot[rotMag == 0] = 0
        costTrans[transMag == 0] = 0
        cost = costRot + costTrans

        return cost

    @staticmethod
    def _findDirecton(sol, graph, modeId, pivId, unitTwist):

        chains = graph.specChains(modeId)
        maxNorm = 0
        maxIndicator = None
        for c in chains:
            if pivId not in c: continue
            pIndex = c.index(pivId)
            prevIndex = pIndex  - 1
            prevId = c[prevIndex]
            if(np.linalg.norm(sol.nodeTwists[modeId, prevId]) < EPSILON): continue
            prevCenter = graph.nodeCenter(prevId)
            tangent = sv.tangent(unitTwist, prevCenter)
            tangentNorm = np.linalg.norm(tangent)
            pointer = graph.nodeCenter(pivId) - prevCenter
            indicator = np.dot(pointer, tangent)
            if(tangentNorm >= maxNorm): 
                maxNorm, maxIndicator = tangentNorm, indicator
        
        if(maxIndicator is None): return 1
        if(maxIndicator > EPSILON): return 1
        elif(maxIndicator < -EPSILON): return -1
        else:
            #raise Exception(MSG_UNIMPLEMENTED)
            return 1
    
    @staticmethod
    def _findMagnitude(sol, graph, pivId, unitTwist, tarMag=None):
        
        if(tarMag is None):
            pivCen = graph.nodeCenter(pivId)
            refScrews = sol.nodeTwists[:, pivId]
            refScrews = refScrews[np.linalg.norm(refScrews, axis=-1) > EPSILON]
            tarMag = np.average(sv.travelDist(pivCen, refScrews))
            if(tarMag < EPSILON):
                _, rotMag, transMag, _ = sv.decompose(refScrews)
                tarMag = np.average(np.maximum(rotMag, transMag))
                
        # get average edge length from new node
        adjEdges = graph.adjEdges(pivId)
        edgeLengths = graph.edgeLength(adjEdges)
        avgEdgeLen = np.average(edgeLengths)

        mag = 1
        if(sv.isTrans(unitTwist)):
            mag = tarMag
        elif(sv.isRot(unitTwist)):
            mag = tarMag / avgEdgeLen
        else:
            raise Exception(MSG_UNIMPLEMENTED)
        
        return mag

    @staticmethod
    def _closestGround(graph, base):

        grounds = [spec.groundId for spec in graph.ioSpecs]
        pts = graph.nodeCenter(grounds)
        closestGround = int(grounds[np.argmin(np.linalg.norm(base - pts, axis=-1))])

        return closestGround

    # node biasing

    @staticmethod
    def _biasNodes(sol, newGraph, toBias, additive=True, order=None, magMul=1):

        configMap = sol.graph.configGroups()
        invMap = invertMap(toBias, outType="set")
        transMask = sol.graph.chainEdgesMask(True, True)

        # for each pivoting node to bias
        for pivId in invMap:

            # for each configurable mode involved for biasing
            for configId in configMap:
                pivModes = list(invMap[pivId].intersection(configMap[configId]))
                if(len(pivModes) == 0): continue
            
                available = FixDecouple._availTwistsNode(sol, transMask, pivId, pivModes)
                picked = FixDecouple._pickBias(sol, available, pivId, pivModes, order=order)
                
                # apply bias
                for i in range(len(pivModes)):
                    modeId, unitTwist = pivModes[i], picked[i]
                    target = configId, modeId, pivId
                    FixDecouple._applyBias(sol, newGraph, target, unitTwist, additive=additive, magMul=magMul)
    
    @staticmethod
    def _availTwistsNode(sol, transMask, pivId, pivModes):
            
        pivCen = sol.graph.nodeCenter(pivId)
        
        # find velocities used by neighboring edges for each mode under configuration
        nEdgeIds = sol.graph.adjEdges(pivId)
        nEdgeidsMasked = nEdgeIds[np.where(transMask[nEdgeIds])]
        scopedModes = sol.edgeTwists[pivModes]
        dofs = scopedModes[:, nEdgeidsMasked]
        dofs = np.concatenate(dofs, axis=0)

        # select biasing unit twists
        available = sv.cleanDOFs(dofs, pivCen)

        return available
    
    # edge splitting

    @staticmethod
    def _splitEdges(sol, newGraph, toSplit, order=None, magMul=1):

        configMap = sol.graph.configGroups()
        invMap = invertMap(toSplit, outType="set")

        # for each pivoting node to bias
        for pivId in invMap:
            eCen = sol.graph.edgeCenter(pivId)

            # split edge and add new node
            closestGroundId = FixDecouple._closestGround(sol.graph, eCen)
            _, _, nnId = newGraph.splitEdge(pivId, closestGroundId)

            # for each configurable mode involved for biasing
            for configId in configMap:
                pivModes = list(invMap[pivId].intersection(configMap[configId]))
                if(len(pivModes) == 0): continue

                available = FixDecouple._availTwistsEdge(sol, pivId, pivModes)
                picked = FixDecouple._pickBias(sol, available, pivId, pivModes,
                                               order=order, mode='e')

                # apply bias
                for i in range(len(pivModes)):
                    modeId, unitTwist = pivModes[i], picked[i]
                    target = configId, modeId, nnId

                    tarMag = np.average(sv.travelDist(sol.graph.edgeCenter(pivId), 
                                                      sol.edgeTwists[modeId, pivId]))
                    tarMag *= magMul
                    FixDecouple._applyBias(sol, newGraph, target, unitTwist,
                                           additive=False, tarMag=tarMag)

    @staticmethod
    def _availTwistsEdge(sol, pivId, pivModes):

        pivCen = sol.graph.edgeCenter(pivId)
        dofs = sol.edgeTwists[pivModes, pivId]
        available = sv.cleanDOFs(dofs, pivCen)

        return available
    
class FixCrossModalDecouple(FixDecouple):
    name = "decoupleX"
    statusCode = 1

    def __init__(self):
        super().__init__()
    
    def examine(self, sol: Solution):

        isTodo = self.check(sol)
        
        if(not isTodo): return sol.graph, False
        if(BF_PRINT_ACTIONS): print("\t Fix Cross Modal Decoupling")
        newGraph = self.autofix(sol)

        return newGraph, True

    def check(self, sol: Solution):
        
        configModes = sol.graph.configGroups()
        
        decoup = {}
        for cId in configModes:
            mIds = configModes[cId]
            
            # get gear trains for each mode under config
            paths, modeIds = [], []
            for mId in mIds:
                chains = sol.graph.specChains(mId, skipZero=True, trainOnly=True) # multiple chains per mode
                paths += [sol.graph._makePathVec(c) for c in chains]
                modeIds += [mId] * len(chains)
            paths = np.stack(paths, axis=0) # stack of all config-mode path vectors
            modeIds = np.asarray(modeIds) # membership of eeach chain path vector
            pivs =  np.nonzero(paths) # find non zero edges (a chain edge)

            # find pivots to focus on
            pivMode = modeIds[pivs[0]]
            pivEdge = pivs[1]
            pivDir = paths[pivs].reshape(-1, 1)
            inpSigId = np.where(sol.graph.C[pivEdge] == pivDir * -1)[1]
            inpSig = sol.nodeTwists[pivMode, inpSigId]
            
            for eId in np.unique(pivEdge):
                modeIds = pivMode[np.where(pivEdge == eId)]
                
                sigs = inpSig[np.where(pivEdge == eId)]
                edgeSpace = sol.edgeTwists[list(mIds), eId]
                edgeSpace = sv.norm(edgeSpace)
                sigs = sv.norm(sigs)
                edgeCons = sv.norm(nullspace(edgeSpace))
                sigOut = edgeCons @ sigs.T # potential issue here
                sigOutMag = np.linalg.norm(sigOut, axis=0)
                
                if(np.all(sigOutMag >= EPSILON)): continue
                
                modes = modeIds[np.where(sigOutMag < EPSILON)]
                for mode in modes: decoup[mode] = decoup.get(mode, []) + [eId]
        
        # simplify
        for key in decoup: decoup[key] = list(set(decoup[key]))

        self.decoup = decoup

        return len(decoup) != 0
   
    def report(self, sol: Solution):

        toBias, toSplit = FixDecouple._planDecoupFix(sol, self.decoup)

        info = {}
        info["edges"] = self._reportDecoupEdges()
        info["nodes"] = set()
        for cId in toBias: info["nodes"].update(toBias[cId])
        info["nodes"] = [int(nId) for nId in info["nodes"]]

        suggestions = self._suggestBiasFix(sol, toBias)
        suggestions += self._suggestSplitFix(sol, toSplit)
        info["suggestions"] = [entry.outputJSON() for entry in suggestions]
        
        return info
    
    def _suggestBiasFix(self, sol: Solution, toBias, order=None):

        configMap = sol.graph.configGroups()
        invMap = invertMap(toBias, outType="set")
        transMask = sol.graph.chainEdgesMask(True, True)

        suggestions = []

        # for each pivoting node to bias
        for pivId in invMap:

            # for each configurable mode involved for biasing
            for configId in configMap:
                pivModes = list(invMap[pivId].intersection(configMap[configId]))
                if(len(pivModes) == 0): continue
            
                available = FixDecouple._availTwistsNode(sol, transMask, pivId, pivModes)
                picked = FixDecouple._pickBias(sol, available, pivId, pivModes, order=order)
                
                # apply bias
                for i in range(len(pivModes)):
                    modeId, unitTwist = pivModes[i], picked[i]
                    direction = FixDecouple._findDirecton(sol, sol.graph, modeId, pivId, unitTwist)
                    magnitude = FixDecouple._findMagnitude(sol, sol.graph, pivId, unitTwist)
                    magnitude *= BF_MAG_MUL * direction
                    entry = DecoupFixSuggestion(configId, modeId, pivId, available, unitTwist, magnitude, target="node")
                    suggestions += [entry]
        
        return suggestions

    def _suggestSplitFix(self, sol: Solution, toSplit, order=None):

        newGraph = sol.graph.clone()

        configMap = sol.graph.configGroups()
        invMap = invertMap(toSplit, outType="set")

        suggestions = []

        # for each pivoting node to bias
        for pivId in invMap:

            eCen = sol.graph.edgeCenter(pivId)

            # split edge and add new node
            closestGroundId = FixDecouple._closestGround(sol.graph, eCen)
            _, _, nnId = newGraph.splitEdge(pivId, closestGroundId)
            
            # for each configurable mode involved for biasing
            for configId in configMap:
                pivModes = list(invMap[pivId].intersection(configMap[configId]))
                if(len(pivModes) == 0): continue
                available = FixDecouple._availTwistsEdge(sol, pivId, pivModes)
                picked = FixDecouple._pickBias(sol, available, pivId, pivModes,
                                               order=order, mode='e')

                # apply bias
                for i in range(len(pivModes)):
                    modeId, unitTwist = pivModes[i], picked[i]
                    tarMag = np.average(sv.travelDist(sol.graph.edgeCenter(pivId), 
                                                      sol.edgeTwists[modeId, pivId]))
                    direction = FixDecouple._findDirecton(sol, newGraph, modeId, nnId, unitTwist)
                    magnitude = FixDecouple._findMagnitude(sol, newGraph, nnId, unitTwist, tarMag)
                    magnitude *= BF_MAG_MUL * direction

                    entry = DecoupFixSuggestion(configId, modeId, pivId, available, unitTwist, magnitude, target="edge")
                    suggestions += [entry]

        del newGraph

        return suggestions
    
    def _reportDecoupEdges(self):

        info = {}
        for modeId in self.decoup:
            info[str(modeId)] = [int(eId) for eId in self.decoup[modeId]]
        
        return info
     
    def fix(self, sol: Solution, fix: dict):

        mode = fix["mode"]
        
        if(mode == "auto"):
            newGraph = self.autofix(sol)
        elif(mode == "semi"):
            newGraph = self._semiFix(sol, fix)
        elif(mode == "manual"):
            newGraph = self._manualFix(sol, fix)

        return newGraph
    
    def autofix(self, sol: Solution):
        
        newGraph = sol.graph.clone()
        toBias, toSplit = FixDecouple._planDecoupFix(sol, self.decoup)
        if(BF_PRINT_ACTIONS and len(toBias) > 0): print("\t\t Nodes to bias (mode: id):", toBias)
        if(BF_PRINT_ACTIONS and len(toSplit) > 0): print("\t\t Edges to split (mode:id):", toSplit)
        
        FixDecouple._biasNodes(sol, newGraph, toBias, additive=False)
        FixDecouple._splitEdges(sol, newGraph, toSplit)

        return newGraph
    
    def _manualFix(self, sol, fix):
        
        newGraph = sol.graph.clone()
        
        for assignment in fix["assignments"]:

            if(assignment["pick"] is None): continue # no assignment, skip

            if("nodeId" in assignment): self._applyNodeFix(newGraph, assignment)
            elif("edgeId" in assignment): self._applyEdgeFix(newGraph, assignment)

        return newGraph
    
    def _applyNodeFix(self, newGraph, assignment):
            
        modeId = assignment["modeId"]
        nodeId = assignment["nodeId"]
        twist = np.asarray(assignment["pick"])
        
        cons = StageTwist(newGraph, modeId, nodeId, twist)
        newGraph.cons += [cons]

    def _applyEdgeFix(self, newGraph, assignment):
        
        modeId = assignment["modeId"]
        edgeId = assignment["edgeId"]
        twist = np.asarray(assignment["pick"])

        eCen = newGraph.edgeCenter(edgeId)
        closestGroundId = FixDecouple._closestGround(newGraph, eCen)
        _, _, nnId = newGraph.splitEdge(edgeId, closestGroundId)

        cons = StageTwist(newGraph, modeId, nnId, twist)
        newGraph.cons += [cons]

    def _semiFix(self, sol, fix):

        order = fix["order"]
        magMul = fix["magMul"]

        newGraph = sol.graph.clone()
        toBias, toSplit = FixDecouple._planDecoupFix(sol, self.decoup)
        if(BF_PRINT_ACTIONS and len(toBias) > 0): print("\t\t Nodes to bias (mode: id):", toBias)
        if(BF_PRINT_ACTIONS and len(toSplit) > 0): print("\t\t Edges to split (mode:id):", toSplit)

        FixDecouple._biasNodes(sol, newGraph, toBias, additive=False, order=order, magMul=magMul)
        FixDecouple._splitEdges(sol, newGraph, toSplit, order=order, magMul=magMul)

        return newGraph

class DecoupFixSuggestion(object):
    def __init__(self, configId, modeId, targId, avaTwists, picked, magnitude, target="node"):

        self.cId = configId
        self.mId = modeId

        if(target == "node"): self.nId, self.eId = targId, None
        elif(target == "edge"): self.nId, self.eId = None, targId
        else: raise Exception(MSG_UNIMPLEMENTED)

        self.avaTwists = avaTwists
        self.picked = picked
        self.mag = magnitude
    
    def outputJSON(self):

        info = {}

        info["configId"] = int(self.cId)
        info["modeId"] = int(self.mId)

        if(self.eId is None): info["nodeId"] = int(self.nId)
        elif(self.nId is None): info["edgeId"] = int(self.eId)
        else: raise Exception(MSG_UNIMPLEMENTED)

        info["avaTwists"] = self.avaTwists.tolist()
        info["cand"] = self.picked.tolist()
        info["mag"] = float(self.mag)

        return info

class NumAlgo(object):

    def __init__(self):

        self.input = None
        self.iter = 0
        self.isComplete = False
        self.isModified = False
        self.log = []
        self.maxIter = None
        self.heuristics = []
        self.isHeuristicsInit = False
        self.useCostVec = True

    def reset(self):
        self.__init__()

    def solve(self, graph, clone=True, useCostVec=True, returnLog=False, maxIter=None, summarizeComp=False):

        self.reset()

        self.input = graph
        
        if(clone): graph = graph.clone()
        self.useCostVec = useCostVec
        self._determineMaxIter(maxIter)
        self._getHeuristics()
        
        while(not self.isComplete and self.iter < self.maxIter):
            graph = self._step(graph)
            if(graph is None): break
        
        if(self.isComplete and summarizeComp): self._summarize(self.log[-1])

        # design is modified but incomplete, one more velocity calculation
        if(self.isComplete):
            status = {"valid": True, "complete": True}
            for h in self.heuristics: status[h.name] = 0
        else:
            if(graph is None):
                status = {"valid": False, "complete": True}
            else:
                sol, status = self.analyze(self.log[-1] if self.isComplete else graph)
                self.log += [sol]


        if(returnLog):
            self.log[-1].status = status
            return self.log, status
        else:
            self.log[-1].status = status
            return self.log[-1], status

    def analyze(self, graph, clone=True, useCostVec=True):

        self.reset()

        self.input = graph
        
        if(clone): graph = graph.clone()
        self.useCostVec = useCostVec
        self._getHeuristics()

        if(not isinstance(graph, Solution)):
            vel = self._findVelocity(graph)
            if(vel is None): 
                print("Unable to find solution")
                return {"valid": False}
            sol = Solution(graph, vel)
        else:
            sol = graph
        
        status = {"valid": True}
        isComplete = True
        for h in self.heuristics:
            flag = h.check(sol)
            code = h.statusCode if flag else 0
            status[h.name] = code
            if(flag): status[h.name + "Info"] = h.report(sol)
            if(code == 1): isComplete = False
        
        status["complete"] = isComplete

        return sol, status

    def fix(self, graph, fix):

        # find velocity space
        if(not isinstance(graph, Solution)):
            vel = self._findVelocity(graph)
            if(vel is None): return {"valid": False}
            sol = Solution(graph, vel)
        else:
            sol = graph
        
        # apply heuristical fix
        for h in self.heuristics:
            if(fix["heuristics"] == h.name): 
                newGraph = h.fix(sol, fix)
                break
        
        # reassess
        sol, status = self.analyze(newGraph)
        # vel = self._findVelocity(newGraph)
        # if(vel is None): return {"valid": False}
        # sol = Solution(newGraph, vel)
        # status = {}
        
        return sol, status

    def _step(self, graph):

        print("iteration: ", self.iter)

        isModified = False

        # step 1: find velocity to achieve transmission with given topology
        vel = self._findVelocity(graph)
        if(vel is None): print(MSG_UNSOLVED); return None
        
        sol = Solution(graph, vel)
        self.log += [sol]

        # step 2: simplify topology and fix transmission
        for h in self.heuristics:
            graph, isModified = h.examine(sol)
            if(isModified): break

        # step 3: check for completeion (i.e., no change to topo)
        if(not isModified): 
            print("\t Complete")
            self.isComplete = True

        # step 4: housekeeping
        self.iter += 1
        for h in self.heuristics: h.reset()

        return graph

    def _determineMaxIter(self, maxIter=None):
        
        maxIter = maxIter if maxIter is not None else BF_ITER_MAX
        if(maxIter == -1): maxIter = float("inf") # iterate until solved
        else: maxIter = int(maxIter)

        self.maxIter = maxIter
    
    def _getHeuristics(self):

        if(self.isHeuristicsInit):
            for h in self.heuristics: h.reset()
            return
            
        heuristics = []

        # simplification heuristics
        topoSimp = TopologySimplification()
        linearFix = FixLinearArc()
        edgeElim = EdgeElimination()
        heuristics += [topoSimp, linearFix, edgeElim]

        # transmission fix
        fixDecoup = FixDecouple()
        fixCrossMod = FixCrossModalDecouple()
        heuristics += [fixDecoup, fixCrossMod]

        self.heuristics = heuristics

    def _findVelocity(self, graph):

        A, B = graph.modelLinearSys()
        cost = graph.costVec() if self.useCostVec else None
        vel = self._solveLinearSys(A, B, cost, cons=graph.cons)
        
        return vel

    def _solveLinearSys(self, A, B, cost=None, applyFilter=True, cons=None):

        # solve for valid solution
        sol = solveCVXPY(A, B, cost, cons=cons)
        
        if(sol is None): return None

        if(applyFilter): sol = L0Filter(sol)

        return sol

    def _summarize(self, sol):

        msg = "Solution Found"
        padding = 30

        print('=' * padding, msg, '=' * padding)
        sol.printChainVelocity(True, True)

        print('=' * ((padding * 2) + 2 + len(msg)))