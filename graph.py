import numpy as np
from messages import *
from screwVectors import *
from parameters import *
from utils import nullspace, nullspaceInt, fixFloat, printBlock, isNdNumerical, copyToClipboard
from subspace import Subspace
import itertools
import time

#np.set_printoptions(precision=NUM_PREC, suppress=True, linewidth=100000, edgeitems=30)

class PrescribedMotion():
    def __init__(self, id, twist, piv=None, vel=None):

        self.id = int(id)

        self.twist = None
        self.isZero = None

        self.pivot = piv
        self.vel = vel

        self.setTwist(twist)
    
    def outputJSON(self, kind):

        json  = {}

        json["stageId"] = self.id
        json["twist"] = self.twist.tolist()
        json["kind"] = kind
        
        if(self.pivot is None and self.vel is None):
            json["pivot"] = np.zeros(EUC_SPACE_DIM).tolist()
            json["vel"] = self.twist.tolist()

        if(self.pivot is not None): json["pivot"] = self.pivot.tolist()
        if(self.vel is not None): json["vel"] = self.vel.tolist()

        return json

    def setTwist(self, twist):

        self.twist = twist if isinstance(twist, np.ndarray) else np.asarray(twist)
        self.isZero = np.linalg.norm(self.twist) < EPSILON

    def setTarId(self, id):
        self.id = id

    def setVel(self, piv, vel):

        self.pivot = piv
        self.vel = vel

    def clone(self):

        twist = np.copy(self.twist)

        if(self.pivot is not None and\
           self.vel is not None): 
            piv = np.copy(self.pivot)
            vel = np.copy(self.vel)
        else:
            piv = None
            vel = None
        
        cloned = PrescribedMotion(self.id, twist, piv, vel)

        return cloned

    def __repr__(self):

        msg = "ID:%d motion: %s" % (self.id, str(self.twist))

        return msg

class KinematicIO():
    def __init__(self, groundId, inputMotions, outputMotions, configId=0):
        
        self.groundId = int(groundId) # int
        self.input = inputMotions # list
        self.output = outputMotions # list
        self.configId = int(configId) # int

    def outputJSON(self):

        json = {}
        json["groundId"] = self.groundId
        json["configId"] = self.configId
        json["pres"] = [inp.outputJSON("inp") for inp in self.input] +\
                       [out.outputJSON("out") for out in self.output]

        return json

    def clone(self):

        input = [inp.clone() for inp in self.input]
        output = [out.clone() for out in self.output]

        cloned = KinematicIO(self.groundId, input, output, self.configId)

        return cloned

    def iterIOs(self, inp=True, out=True):

        rtn = []

        if(inp): rtn += self.input
        if(out): rtn += self.output

        return rtn

    def enumIds(self, inp=True, out=True):

        return [s.id for s in self.iterIOs(inp, out)]

    def printInfo(self):

        for i in range(len(self.input)):
            print(" Input: %s" % self.input[i].__repr__())
        for i in range(len(self.output)):
            print("Output: %s" % self.output[i].__repr__())

class JointConstraint():
    def __init__(self, graph, id, twists):

        self.graph = graph
        self.id = id
        self.vecs = self._initTwists(twists)
    
    def _initTwists(self, twists):

        if(twists.ndim == 1):
            twists = twists.reshape(1, -1)
        
        return twists
    
    def model(self, A, B, x):

        jft = self.graph._jftFullRank(self.id).T
        vel = x[self.id * SCREW_SPACE_DIM: (self.id + 1) * SCREW_SPACE_DIM]

        cons = jft @ vel == np.zeros((SCREW_SPACE_DIM, 1))
        return cons

class StageConstraint():
    def __init__(self, graph, stageId, twists):

        self.graph = graph
        self.sId = stageId
        self.vecs = self._initTwists(twists)

    def outputJSON(self):

        output = {}
        
        output["twist"] = self.vecs.tolist()
        output["stageId"] = self.sId

        return output

    def _initTwists(self, twists):

        if(twists.ndim == 1):
            twists = twists.reshape(1, -1)
        
        return twists
    
    def modelCons(self):

        raise Exception(MSG_UNIMPLEMENTED)

class StageTwist(StageConstraint):
    def __init__(self, graph, modeId, stageId, twist):
        super().__init__(graph, stageId, twist)

        self.modeId = modeId

    def outputJSON(self):

        output = super().outputJSON()

        output["modeId"] = self.modeId
        
        return output

    def _initTwists(self, twists):

        if(twists.ndim == 1):
            twists = twists.reshape(1, -1)
        
        return twists
    
    def modelCons(self, A, B, x):

        P = self.graph.path(self.graph.ioSpecs[self.modeId].groundId, self.sId)
        A = self.graph.embedJFTtoLoop(P) # 6 * 6e
        cons = A @ x[:, self.modeId] == self.vecs.reshape(SCREW_SPACE_DIM)
        
        return cons
    
    def clone(self, graph):

        return StageTwist(graph, self.modeId, self.sId, np.copy(self.vecs))
    
class Graph(object):

    def __init__(self, A=None, C=None, E=None, N=None, meta=None, spec=None, bdEdge=True):
        
        self.nodeCount = 0 # number of nodes
        self.edgeCount = 0 # number of edges

        self.edgeFeatureLen = 0 # length of edge features
        self.nodeFeatureLen = 0 # length of node features

        self.A = None # np.ndarray(node, node), nodal adjacency matrix
        self.C = None # np.ndarray(edge, node), incidence matrix

        self.E = None # np.ndarray(edge, featurelen), edge feature matrix
        self.N = None # np.ndarray(node, featurelen), node feature matrix

        self.meta = meta # dict, metadata
        self.bdEdge = bdEdge

        if(spec!=None):
            self._initFromSpec(spec)
        else:
            self._initFromData(A, C, E, N)

    def _initFromSpec(self, spec):

        self.nodeCount = spec["nodeCount"]
        self.edgeCount = spec["edgeCount"]
        if("nodeFeatureLen" in spec): self.nodeFeatureLen = spec["nodeFeatureLen"]
        if("edgeFeaturelen" in spec): self.edgeFeatureLen = spec["edgeFeatureLen"] 
        
        self.A = np.zeros((self.nodeCount, self.nodeCount))
        self.C = np.zeros((self.edgeCount, self.nodeCount))

    def _initFromData(self, A, C, E, N):

        self._initTopology(A, C)
        self._initFeature(E, N)
    
    def _initTopology(self, A, C):

        # type conversion
        if(A is not None):
            if(isinstance(A, np.ndarray)): A = A.astype(np.int32)
            else: A = np.asarray(A).astype(np.int32)
        if(C is not None):
            if(isinstance(C, np.ndarray)): C = C.astype(np.int32)
            else: C = np.asarray(C).astype(np.int32)

        # check mode
        if(C is not None): # C-first
            
            self.C = C
            if(A is not None):
                self.A = A
            else:
                self._constructA()
                
        elif(A is not None):

            # C is invalid but A is valid
            self.A = A
            self._constructC()
        else:

            raise Exception(MSG_INVALID_GRAPH_SPEC)

        self.nodeCount = self.C.shape[1] if self.C.size > 0 else 1
        self.edgeCount = self.C.shape[0]
         
    def _initFeature(self, E, N):
        
        # type conversion
        if(not isinstance(E, np.ndarray) and E is not None):
            E = np.asarray(E)
        if(not isinstance(N, np.ndarray) and N is not None):
            N = np.asarray(N)
        
        self.N = N
        self.E = E

        if(isinstance(N, np.ndarray)): self.nodeFeatureLen = self.N.shape[-1]
        if(isinstance(E, np.ndarray)): self.edgeFeatureLen = self.E.shape[-1]
    
    def _constructA(self):

        self.nodeCount = self.C.shape[1] if self.C.size > 0 else 1

        self.A = np.zeros((self.nodeCount, self.nodeCount), dtype=np.int32)

        for row in self.C:
            nodeIds = np.nonzero(row)[0]
            assert len(nodeIds) == 2, MSG_INVALID_GRAPH_TOPO
            
            nodeId1, nodeId2 = nodeIds
            if(row[nodeId2] != 1): nodeId1, nodeId2 = nodeId2, nodeId1

            self.A[nodeId1, nodeId2] = 1
            self.A[nodeId2, nodeId1] = -1
            
    def _constructC(self):
        
        upper = np.triu(self.A)
        edges = np.transpose(np.nonzero(upper))
        edgeCount = len(edges)

        self.C = np.zeros((edgeCount, len(self.A)), dtpye=np.int32)
        for i in range(edgeCount):
            e = edges[i]
            start, end = e
            self.C[i, start] = -1
            self.C[i, end] = 1
    
    def _dijkstra(self, ground, cost=None, exclusion=[]):

        return Graph._dijkstra(self.A, ground, cost, exclusion)

    def _dijkstra_vanilla(self, ground, cost=None, exclude=[]):

        nodes = set(range(self.nodeCount))

        dist = {}
        for i in range(self.nodeCount): dist[i] = float("inf")
        dist[ground] = 0
        if(len(exclude) != 0):
            for excId in exclude: 
                dist.pop(excId)
                nodes.remove(excId)
        prev = {}

        exc = set(exclude)

        while(len(nodes) != 0):

            nodeDists = [(node, dist[node]) for node in nodes]
            cur = sorted(nodeDists, key=lambda x: x[1])[0][0]
            nodes.remove(cur)
            neighbors = np.nonzero(self.A[cur])[0]
            
            alt = dist[cur] + 1
            for n in neighbors:
                if(n in exc): continue
                if(alt < dist[n]):
                    dist[n] = alt
                    prev[n] = cur

        return dist, prev

    def _makePathVec(self, path):

        pathVec = np.zeros(self.edgeCount, dtype=np.int32)
        
        for i in range(len(path) - 1):
            nodeStart, nodeEnd = path[i], path[i + 1]
            en0, en1 = self.C[:, nodeStart], self.C[:, nodeEnd]
            flagger = en0 * en1
            e = np.nonzero(flagger)[0][0]
            pathVec[e] = self.A[nodeStart, nodeEnd]
        
        return pathVec
    
    def printInfo(self):

        topology = "Nc: %d Ec: %d" % (self.nodeCount, self.edgeCount)
        features = "Nf: %d Ef: %d" % (self.nodeFeatureLen, self.edgeFeatureLen)

        msg = "Graph(Topology: %s; Features: %s)" % (topology, features)

        print(msg)

    def _Q(self):

        Q = nullspaceInt(self.C.T)

        return Q

    def path(self, ground, target, cost=None, exclusion=[]):
        
        path = Graph.traverse(self.A, ground, target, cost=cost, exclusion=exclusion)
        P = self._makePathVec(path)

        return P

    def nodePaths(self, ground, cost=None, exclusion=[]):

        Ps = []
        for i in range(self.nodeCount):
            P = self.path(ground, i, cost, exclusion)
            Ps += [P]
        
        Ps = np.stack(Ps, axis=0)

        return Ps

    def clone(self):

        # topoology
        A = np.copy(self.A) if isinstance(self.A, np.ndarray) else torch.clone(self.A)
        C = np.copy(self.C) if isinstance(self.C, np.ndarray) else torch.clone(self.C)

        # features
        if(self.N is None):
            N = None
        else:
            N = np.copy(self.N) if isinstance(self.N, np.ndarray) else torch.clone(self.N)
        
        if(self.E is None):
            E = None
        else:
            E = np.copy(self.E) if isinstance(self.E, np.ndarray) else torch.clone(self.E)
        
        newGraph = Graph(A=A, C=C, N=N, E=E)

        return newGraph

    def unsigned(self, copy=False):

        graph = self.clone() if copy else self
        
        graph.A = np.abs(self.A)
        graph.C = np.abs(self.C)

        return graph

    def subGraphGrouping(self, exclusion=[]):

        return Graph._groupSubgraphs(self.A, exclusion=exclusion)
    
    def pathToVisit(self, path):

        factor = path.reshape(-1, 1)

        directedPath = factor * self.C
        directedPath[directedPath <= 0] = 0

        visit = np.sum(directedPath, axis=-1)

        return visit

    def splitEdge(self, edgeId, ground=None):
        
        newC = np.zeros((self.edgeCount + 1, self.nodeCount + 1),
                        dtype=np.int32) # expand C
        
        newC[:self.edgeCount, :self.nodeCount] = self.C # copy over

        sId = np.where(self.C[edgeId] == -1)[0]
        eId = np.where(self.C[edgeId] == 1)[0]

        newC[edgeId, eId] = 0
        newC[edgeId, -1] = 1
        newC[-1, eId] = -1
        newC[-1, -1] = 1
        
        addedGround = False
        # optionally add ground(common) edge to further constrain
        if(ground is not None and isinstance(ground, int)):
            gEdge = np.zeros(newC.shape[-1])
            addedGround = True

            if(ground == -1): # find closest ground
                ground = self.closestCommon(sId, eId)
            gEdge[ground] = -1
            gEdge[-1] = 1

            newC = np.append(newC, gEdge.reshape(1, -1), axis=0)
            
        self._initTopology(None, newC)

        return addedGround

    def edgeStartId(self, piv=None):

        ids = np.where(self.C == -1)[1]

        if(piv is None): return ids
        else: return ids[piv]
    
    def edgeEndId(self, piv=None):

        ids = np.where(self.C == 1)[1]

        if(piv is None): return ids
        else: return ids[piv]
    
    def closestCommon(self, id1, id2, exclusion=[]):
        
        distMatrix = Graph.crossDist(self.A, exclusion)

        dist1 = distMatrix[id1]
        dist2 = distMatrix[id2]
        summed = dist1 + dist2

        order = np.argsort(summed)
        order = np.delete(order, [id1, id2]) # remove id1, id2
        cc = order[0]

        return cc
    
    def adjEdges(self, nId):

        index = self.C[:, nId]
        connected = np.where(index != 0)[0]

        return connected
    
    def degree(self):

        connectivity = np.abs(self.A)
        degree = np.sum(connectivity, axis=-1)

        return degree

    def adjNodes(self, id):

        neighbors = np.where(self.A[id] != 0)[0]

        return neighbors

    def edgeBetweenNodes(self, start, end):
        s = self.C[:, start]
        e = self.C[:, end]
        index = s * e
        id = np.where(index == -1)[0][0]
        return id
    
    def tracePath(self, nodes):

        path = np.zeros(self.edgeCount, np.int32)

        for i in range(len(nodes) - 1):
            sId = nodes[i]
            eId = nodes[i + 1]
            eId = self.edgeBetweenNodes(sId, eId)
            path[eId] = 1 if self.C[eId, sId] == -1 else -1
        
        return path

    @staticmethod
    def _groupSubgraphs(A, exclusion=[]):

        if(isinstance(exclusion, set)): exclusion = list(exclusion)

        adjBase = A.astype(np.int32) + np.identity(len(A), dtype=np.int32)
        adj = adjBase

        adj[exclusion,:] *= 0
        adj[:,exclusion] *= 0


        # flood-matmul algorithm
        isComplete = False
        i = 0
        while(not isComplete):
            newAdj = np.matmul(adj, adj.T) # logarithmic 2^n
            newAdj[newAdj > 0] = 1
            difference = newAdj - adj
            if(np.all(difference == 0)): isComplete = True
            adj = newAdj
            i += 1

        groups = np.unique(adj, axis=0)

        return groups

    @staticmethod
    def denseNet(nodes):

        A = np.matmul(nodes.reshape(-1, 1), nodes.reshape(1, -1))
        A = np.tril(-1 * np.ones(A.shape, dtype=np.int32)) + np.triu(A)
        
        A *= (1 - np.identity(len(nodes), dtype=np.int32))

        return A

    @staticmethod
    def _trackPath(prev, start, end):

        path = [end]
        while(path[0] != start):
            cur  =  path[0]
            next =  prev[cur]
            path = [next] + path
        
        return path

    @staticmethod
    def traverse(A, start, end, cost=None, exclusion=[]):

        if(not USE_VECTORIZED_PATHFINDING):
            prev, _ = Graph.dijkstra(A, start, cost=cost, exclusion=exclusion)
            path = Graph._trackPath(prev, start, end)
        
        else:
            path = Graph._pathing(A, start, end, cost=cost, exclusion=exclusion)
            if(path[0] != start or path[-1] != end): path = None

        return path
    
    @staticmethod
    def dijkstra(A, ground, cost=None, exclusion=[]):
        
        if(not isinstance(cost, np.ndarray)):
            cost = np.ones(A.shape)
        else:
            if(cost.shape != A.shape):
                raise Exception("invalid cost matrix shape")

        nodeCount = A.shape[0]

        nValid = np.ones(nodeCount, dtype=np.int8)
        nValid[exclusion] = 0
        dist = np.full(nodeCount, np.inf)
        dist[ground] = 0

        prev = np.full((nodeCount), ground)
        
        while(not np.all(nValid == 0)):
            nodes = np.where(nValid == 1)[0]
            d = dist[nodes]
            cur = nodes[np.argsort(d)[0]]
            nValid[cur] = 0
            neighbors = np.nonzero(A[cur])[0]
            transCost = cost[cur][neighbors]
            alt = dist[cur] + transCost
            # compare and update node shorted directed neighbor
            update = np.minimum(alt, dist[neighbors])
            prev[neighbors[alt < dist[neighbors]]] = cur
            dist[neighbors] = update

        return prev, dist

    @staticmethod
    def _exhaustPathing(A, start, end, cost=None, exclusion=[]):
        
        if(not isinstance(cost, np.ndarray)):
            cost = np.ones(A.shape)
        else:
            if(cost.shape != A.shape):
                raise Exception("invalid cost matrix shape")
        
        dist = Graph.crossDist(A, exclusion)
        d = dist[start, end]

        # get traversal distance list
        orders = []
        for i in range(d + 1):
            sCond = dist[:,start] == i
            eCond = dist[:,end] == d - i
            mask = np.logical_and(sCond, eCond)
            nextNodes = np.where(mask)[0]
            orders += [nextNodes]

        # expand-build paths
        paths = np.asarray([[start]], dtype=np.int32)
        costs = np.zeros(paths.shape)
        for i in range(len(orders) - 1):

            # find next node adjacency
            curNodes = paths[:,-1]
            nextNodes = orders[i + 1]
            adjMask = dist[curNodes][:,nextNodes]
            adjMask = np.copy(adjMask)
            adjMask[adjMask != 1] = 0

            # expand paths based on how many branches are available
            branches = np.sum(adjMask, axis=-1)
            paths = np.repeat(paths, branches, axis=0)
            costs = np.repeat(costs, branches)

            # find next nodes per branch
            nextIds = np.where(adjMask == 1)[1]
            next = nextNodes[nextIds].reshape(-1, 1)

            # take step
            paths = np.concatenate([paths, next], axis=-1)
        
        costs = np.zeros(len(paths)) # TODO

        return paths, costs
    
    @staticmethod
    def _pathing(A, start, end, cost=None, exclusion=[]):

        paths, costs = Graph._exhaustPathing(A, start, end, cost, exclusion)
        
        ordered = np.argsort(costs)
        path = paths[ordered[0]].tolist()

        return path

    @staticmethod
    def crossDist(A, exclusion=[]):

        if(isinstance(exclusion, set)):
            exclusion = list(exclusion)

        A = np.copy(A)
        A[A == -1] = 1
        A[exclusion,:] *= 0
        A[:,exclusion] *= 0
        
        dist = np.zeros(A.shape, dtype=np.int32)
        filled = np.identity(A.shape[0], dtype=np.int32)

        i = 1
        isDone = False
        while (not isDone):
            frontier = filled @ A # neighbors
            frontier[filled != 0] = 0 # newly touched nodes
            dist[frontier != 0] = i
            filled[frontier != 0] = 1
            isDone = np.all(frontier == 0)
            i += 1
        
        return dist

class CMGraph(Graph):
    def __init__(self, A=None, C=None, E=None, N=None, meta=None, spec=None, ioSpecs=None, bypass=False):
        super().__init__(A, C, E, N, meta, spec)

        self.jfts = None
        self.jftsDeg = None
        self.FT = None
        self.X = None
        self.Q = None

        self.isFrozen = False

        self.cons = []
        self.ioSpecs = []

        if(ioSpecs is not None):
            self.ioSpecs = ioSpecs
            if(not bypass): self._initIOSpecs(ioSpecs)
    
    def _initIOSpecs(self, ioSpecs):

        #if(len(ioSpecs) > 1): raise Exception(MSG_UNIMPLEMENTED)

        spec = ioSpecs[0]
        self.N[spec.groundId, EUC_SPACE_DIM + 2] = 1

        for i in spec.input:
            self.N[i.id, EUC_SPACE_DIM + 0] = 1
            self.N[i.id, EUC_SPACE_DIM + 3:] = i.twist
        
        for o in spec.output:
            self.N[o.id, EUC_SPACE_DIM + 1] = 1
            self.N[o.id, EUC_SPACE_DIM + 3:] = o.twist

    def _Q(self, cacheResult=False, froceRecomp=False):

        if(self.isFrozen and not froceRecomp): 
            return self.Q # shortcut from cache

        Q = super()._Q()

        if(cacheResult):
            self.Q = Q

        return Q
    
    def _jfts(self, stack=True, cacheResult=False, froceRecomp=False):
        # joint freedom topology

        if(self.isFrozen and not froceRecomp):
            return self.jfts, self.jftsDeg # shortcut from cache

        jfts = []
        for line in self.E:
            jft = CMGraph._toJFT(line)
            jfts += [jft]
        jftsDeg = np.asarray([jft.shape[0] for jft in jfts])

        if(stack):
            jfts = np.concatenate(jfts, axis=0)

        if(cacheResult):
            self.jfts = jfts
            self.jftsDeg = jftsDeg
        
        return jfts, jftsDeg

    def _FT(self, cacheResult=False, forceRecomp=False):
        
        if(self.isFrozen and not forceRecomp):
            return self.FT # shortcut from cache
        
        jfts, jftsDeg = self._jfts()
        Q = self._Q()

        Qdeg = Q.shape[0]
        mask = np.repeat(Q, jftsDeg, axis=1)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, SCREW_SPACE_DIM, axis=-1)

        # freedom topology matrix
        FT = np.expand_dims(jfts, axis=0)
        FT = np.repeat(FT, Qdeg, axis=0)

        FT = FT * mask
        FT[FT == 0] = 0
        FT = np.concatenate(FT, axis=-1).T

        if(cacheResult):
            self.FT = FT

        return FT

    def _X(self, cacheResult=False, forceRecomp=False):

        if(self.isFrozen and not forceRecomp):
            return self.X # shortcut from cache
        
        FT = self._FT()
        
        X = nullspace(FT) # nullspaceInt
        X = X / np.linalg.norm(X, axis=-1).reshape((-1, 1)) # added
        X = fixFloat(X) # added
        if(cacheResult):
            self.X = X
        
        return X
    
    def _P(self, ground, target):

        if(self.isFrozen): jfts, jftsDeg = self.jfts, self.jftsDeg
        else: jfts, jftsDeg = self._jfts()

        P = self.path(ground, target)
        P = np.repeat(P, jftsDeg, axis=0)
        P = np.expand_dims(P, axis=-1)
        P = np.repeat(P, SCREW_SPACE_DIM, axis=-1)
        P = jfts * P

        return P

    def outputJSON(self, copy=False):

        if(self.C.size > 0):
            s = np.argwhere(self.C == -1)[:, 1]
            e = np.argwhere(self.C == 1)[:, 1]
            edgePair = np.stack([s, e], axis=-1)
        else:
            edgePair = np.zeros((0, 0))
        nodeCoord = self.N[:, :EUC_SPACE_DIM]
        dom = self.getDomain()
        edgeCoord = self.E[:, :EUC_SPACE_DIM]

        output = {}
        output["edgePair"] = edgePair.tolist()
        output["nodeCoord"] = nodeCoord.tolist()
        output["coordRange"] = dom.tolist()
        output["edgeCenter"] = edgeCoord.tolist()
        
        if(len(self.ioSpecs) > 0):
            objs = {}
            objs["count"] = len(self.ioSpecs)
            objs["modes"] = [io.outputJSON() for io in self.ioSpecs]
            output["objectives"] = objs
        
        output["time"] = time.asctime(time.localtime())

        if(copy):
            copyToClipboard(str(output), "topology info")

        return output

    def freeze(self):

        self.isFrozen = True

        self._Q(cacheResult=True, froceRecomp=True)
        self._jfts(cacheResult=True, froceRecomp=True)
        self._FT(cacheResult=True, forceRecomp=True)
        self._X(cacheResult=True, forceRecomp=True)

        msg = "Froze graph with %d kinematic loops and %d modes" %\
              (self.Q.shape[0], self.X.shape[0])

        return msg

    def unfreeze(self):

        self.isFrozen = False

        self.jfts = None
        self.jftsDeg = None
        self.FT = None
        self.X = None
        self.Q = None

    def config(self, setVal=None, vecForm=False):

        if(setVal is not None):

            if(isinstance(setVal, list) or isinstance(setVal, tuple)):
                setVal = np.asarray(setVal)

            if(setVal.size != self.edgeCount * SCREW_SPACE_DIM):
                raise Exception(MSG_SIZE_MISMATCH)
            elif(setVal.ndim == 1):
                setVal = setVal.reshape((self.edgeCount, SCREW_SPACE_DIM))

            self.E[:, EUC_SPACE_DIM:] = setVal
            self.unfreeze()
        
        rtn = self.E[:, EUC_SPACE_DIM:]

        if(vecForm):
            rtn = rtn.reshape((-1))
        
        return rtn

    def clone(self):

        A = np.copy(self.A)
        C = np.copy(self.C)
        N = np.copy(self.N) if isinstance(self.N, np.ndarray) else None
        E = np.copy(self.E) if isinstance(self.E, np.ndarray) else None

        ios = [io.clone() for io in self.ioSpecs]
        
        newGraph = CMGraph(A=A, C=C, N=N, E=E, ioSpecs=ios) # shared io specs
        newGraph.cons = [con.clone(newGraph) for con in self.cons]

        if(self.isFrozen): newGraph.freeze()

        return newGraph

    def goal(self, setVal=None):

        if(setVal is not None):
            self.N[3, EUC_SPACE_DIM + 3:] = setVal

        return self.N[3, EUC_SPACE_DIM + 3:]
    
    def state(self, setVal=None, useDouble=False):

        if(setVal is not None):
            if(useDouble):
                self.E[-2, EUC_SPACE_DIM:] = setVal[:SCREW_SPACE_DIM]
                self.E[-1, EUC_SPACE_DIM:] = setVal[SCREW_SPACE_DIM:]

            else:
                self.E[-1, EUC_SPACE_DIM:] = setVal
            
            self.unfreeze()
        
        if(useDouble):
            line1 = self.E[-2, EUC_SPACE_DIM:]
            line2 = self.E[-1, EUC_SPACE_DIM:]
            state = np.concatenate([line1, line2], axis=0)
        else:
            state = self.E[-1, EUC_SPACE_DIM:]

        return state

    def jft(self, id):

        line = self.E[id]

        return CMGraph._toJFT(line)

    def solveDOF(self, ground, target, rowEchForm=True):

        X = self.X if self.isFrozen else self._X()
        if(X.size == 0): return None

        P = self._P(ground, target)
        freedom = np.matmul(P.T, X.T).T
        freedom = Subspace.simplifySpace(freedom, removeZeroRows=True, normalize=True)
        freedom = Subspace(spans=freedom).spanSpace

        freedom[np.abs(freedom) < EPSILON] = 0
        freedom = sp.Matrix(freedom).rref()[0]
        freedom = np.asarray(freedom, dtype=np.float64)

        freedom = norm(freedom)

        return freedom

    def _specOutputCheck(self, spec, solConfigSpace):
        
        kinematics = spec.output

        X = self._X()

        # possible solution is present, check if bodies can move w/o motion
        # [P][X]
        isDone = True
        for km in kinematics:
            P = self._P(spec.groundId, km.id)
            A = np.matmul(P.T, X.T).T # A as in Ax = b
            actSpace = solConfigSpace.transformBy(A)

            isAchievable = actSpace.isPtOnSubspace(km.twist)
            isZero = actSpace.isPtOnSubspace(np.zeros(SCREW_SPACE_DIM))
            if(not isAchievable): # output motion is unattainable
                isDone = False
                break
            if(isZero and not km.isZero): # target is non-zero but zero motion solution is possible
                isDone = False
                break

        return isDone
    
    def Qloops(self):

        Q = self._Q()
        
        loops = []
        for loop in Q:
            tar = np.zeros(SCREW_SPACE_DIM)
            newLoop = loop, tar
            loops += [newLoop]
        
        return loops

    def specLoops(self):

        loops = []
        for spec in self.ioSpecs:
            loops += self._specLoop(spec)

        return loops

    def _specLoop(self, spec, addStageLoop=True):

        chainLoops, _ = self._specDeltaLoops(spec, mode="delta")

        if(addStageLoop):
            stageLooops, _ = self._specIOfreedom(spec)
            loops = chainLoops + stageLooops
        else:
            loops = chainLoops
        
        return loops

    def _jftFullRank(self, edgeId, useEdgePos=True):

        pivot = self.E[edgeId, :EUC_SPACE_DIM] if useEdgePos else np.zeros(EUC_SPACE_DIM)

        jft = fullSpace(pivot)

        return jft
    
    def _jftMasked(self, edgeId, useEdgePos=True):

        fullRanked = self._jftFullRank(edgeId, useEdgePos)

        mask = self.E[edgeId][EUC_SPACE_DIM:EUC_SPACE_DIM + SCREW_SPACE_DIM]
        mask = mask.reshape((-1, 1))
        masked = fullRanked * mask
        
        return masked

    def embedJFTtoLoop(self, loop, jftType="full"):
        # jft: "full" or "masked"
        
        assert len(loop) == self.edgeCount, MSG_UNIMPLEMENTED
        assert jftType == "full" or jftType == "masked", MSG_UNIMPLEMENTED

        degrees = np.zeros(self.edgeCount, dtype=np.int32) + SCREW_SPACE_DIM
        mask = np.repeat(loop, degrees) # expand per jft degree
        mask = np.expand_dims(mask, axis=-1) # create new dimension...
        mask = np.repeat(mask, SCREW_SPACE_DIM, axis=-1) # ...to match screw vectors

        stack = []
        for i in range(self.edgeCount):

            if(jftType == "full"): jft = self._jftFullRank(i)
            elif(jftType == "masked"): jft =  self._jftMasked(i)
            else: raise Exception(MSG_INVALID_INPUT)
            stack += [jft]
        
        fullJFT = np.concatenate(stack, axis=0)

        embedded = mask * fullJFT
        embedded[embedded == 0] = 0

        return embedded.T

    def embedJFTtoLoops(self, loops, jftType="full"):
        
        assert loops.shape[-1] == self.edgeCount, MSG_UNIMPLEMENTED
        assert jftType == "full" or jftType == "masked", MSG_UNIMPLEMENTED

        degrees = np.zeros(self.edgeCount, dtype=np.int32) + SCREW_SPACE_DIM
        mask = np.repeat(loops, degrees, axis=-1) # expand per jft degree
        mask = np.expand_dims(mask, axis=-2)
        mask = np.repeat(mask, SCREW_SPACE_DIM, axis=-2) # ...to match screw vectors
        mask = mask.astype(np.float32)

        embedded = mask * self.fullJFT()

        embedded = embedded.reshape(-1, embedded.shape[-1])

        return embedded

    def fullJFT(self):

        loop = np.ones(self.edgeCount)
        
        embedded = self.embedJFTtoLoop(loop)

        return embedded
    
    def fullJFTSquare(self):

        dim = self.edgeCount * SCREW_SPACE_DIM
        m = np.zeros((dim, dim))

        dofBlocks = self.fullJFT()
        for i in range(self.edgeCount):
            indexStart, indexEnd = i * SCREW_SPACE_DIM, (i + 1) * SCREW_SPACE_DIM  
            edgeBlock = dofBlocks[:, indexStart:indexEnd]

            m[indexStart: indexEnd, indexStart: indexEnd] = edgeBlock

        return m
    
    def embedSpeed(self, speed):

        if(speed.size == 0): return np.zeros(speed.shape)
        
        dofConfig = self.E[:, EUC_SPACE_DIM:].flatten()
        freedomIndices = np.where(dofConfig == 1)[0]
        indexer = np.zeros(speed.shape, dtype=np.int32)
        indexer[...,:] = freedomIndices
        

        newShape = list(speed.shape[:-1]) + [self.edgeCount * SCREW_SPACE_DIM]
        embedded = np.zeros(newShape)
        np.put_along_axis(embedded, indexer, speed, -1)
        
        return embedded

    def _procGen(self):

        config = self.E[:, EUC_SPACE_DIM:]
        configVec = config.flatten()

        procConfigs = []
        for i in range(len(configVec)):
            new = np.zeros(len(configVec))
            new[:i] = configVec[:i]
            new = new.reshape(config.shape)
            procConfigs += [new]
        
        return procConfigs

    def modelLinearSys(self, specId=None, printLoop=False):

        # Q loops
        AofQ, BofQ = self._modelLinearSysQ(printLoop)

        # for each spec, compute velocity space
        
        As, Bs = [], []
        if(specId is None):
            for spec in self.ioSpecs:
                AofSpec, BofSpec = self._modelLinearSysSpec(spec, printLoop)
                A = np.concatenate(AofQ + AofSpec, axis=0)
                B = np.concatenate(BofQ + BofSpec, axis=0)
                As += [A]
                Bs += [B]
        
        else:
            spec = self.ioSpecs[specId]
            AofSpec, BofSpec = self._modelLinearSysSpec(spec, printLoop)
            A = np.concatenate(AofQ + AofSpec, axis=0)
            B = np.concatenate(BofQ + BofSpec, axis=0)
            As += [A]
            Bs += [B]

        return As, Bs

    def _modelLinearSysQ(self, printLoop=False):

        Astack, Bstack = [], []
        Qloops = self.Qloops()
        loopCount = len(Qloops)
        if(loopCount == 0): return [], []
        
        for loop, B in Qloops: # as in Ax=B
            A = self.embedJFTtoLoop(loop, jftType="full")
            if(printLoop): print(loop, B)
            Astack += [A]
            Bstack += [B]
        
        qLoopA = np.concatenate(Astack, axis=0)
        qLoopB = np.concatenate(Bstack, axis=0)
        AofQ = [qLoopA]
        BofQ = [qLoopB]

        return AofQ, BofQ
    
    def _modelLinearSysSpec(self, spec, printLoop=False):

        specLoops, specBs = [], []
        for km in spec.iterIOs():
            loop = self.path(spec.groundId, km.id)
            B = km.twist
            if(printLoop): print(loop, B)
            specLoops += [loop]
            specBs  += [B]
        
        if(len(specLoops) == 0): return [], []

        specLoops = np.stack(specLoops, axis=0)


        AofSpec = [self.embedJFTtoLoops(specLoops, jftType="full")]
        BofSpec = [np.concatenate(specBs, axis=0)]
        
        return AofSpec, BofSpec

    def modelConsSys(self, addStageLoop=True):

        # Q loops
        Astack, Bstack = [], []
        for loop, B in self.Qloops(): # as in Ax=B
            A = self.embedJFTtoLoop(loop, jftType="full")
            Astack += [A]
            Bstack += [B]
        AofQ = np.concatenate(Astack, axis=0)
        BofQ = np.concatenate(Bstack, axis=0)


        syss = []
        for spec in self.ioSpecs:
            
            loops = self._specLoop(spec, addStageLoop)
            
            specAstack, specBstack = [], [] # delta constraint
            for loop, B in loops:
                A = self.embedJFTtoLoop(loop, jftType="full")
                specAstack += [A]
                specBstack += [B]
            
            AofSpec = np.concatenate(specAstack, axis=0)
            BofSpec = np.concatenate(specBstack, axis=0)
        
            specA = np.concatenate([AofQ, AofSpec], axis=0)
            specB = np.concatenate([BofQ, BofSpec], axis=0)

            syss += [(specA, specB)]

        return syss

    def solVelocitySpaces(self, addStageLoop=True):
        
        # Q loops
        qLoops = []
        for loop, B in self.Qloops(): # as in Ax=B
            A = self.embedJFTtoLoop(loop, jftType="full")
            qLoops += [A]
        A = np.concatenate(qLoops, axis=0)
        B = np.zeros(A.shape[0])
        AofQ = [A]
        BofQ = [B]

        # for each spec, compute velocity space
        specSols = []
        for spec in self.ioSpecs:
            loops = self._specLoop(spec, addStageLoop)
            stackA, stackB = [], [] # as in Ax=B
            for loop, B in loops:
                A = self.embedJFTtoLoop(loop, jftType="full")
                stackA += [A]
                stackB += [B]

            # speed vector combinations to solve IO spec while respecting kinematic compatibility
            A = np.concatenate(AofQ + stackA, axis=0)
            B = np.concatenate(BofQ + stackB, axis=0)
            specSolSpace = Subspace.solveLinearSys(A, B)
            
            specSols += [specSolSpace]

        return specSols

    def distToSolVelocitySpace(self, velSpace):

        curSpace = self.velocitySpace()
        dist = curSpace.minDistTo(velSpace)

        return dist

    def specZeroDeltaSpaces(self, addIO=True):

        # model level constraints
        qLoops = self.Qloops()
        qConsEqs = [(self.embedJFTtoLoop(loop), B) for loop, B in qLoops]

        designSystems = []
        for spec in self.ioSpecs:
            specIoLoops, ioIds = self._specIOfreedom(spec)
            specIoConsEqs = [(self.embedJFTtoLoop(loop), B) for loop, B in specIoLoops]
            
            # convert io stage eqs into a map
            specIoConsEqsMap = {}
            for i in range(len(ioIds)):
                id, eqs = ioIds[i], specIoConsEqs[i]
                specIoConsEqsMap[id] = eqs
                
            deltaLoops, deltaIds = self._specDeltaLoops(spec, mode="zero")
            specSystems = []
            for i in range(len(deltaIds)):
                startId, endId = deltaIds[i]

                # I/O pointer (end stage with zero)
                ioEqs = [specIoConsEqsMap[startId]]
                ioEqs += [(specIoConsEqsMap[endId][0], np.zeros(SCREW_SPACE_DIM))]

                # delta equates to zero
                loop, B = deltaLoops[i]
                A = self.embedJFTtoLoop(loop)
                deltaConsEqs = [(A, B)]

                # construct linear system
                if(addIO): stack = qConsEqs + ioEqs + deltaConsEqs
                else: stack = qConsEqs + deltaConsEqs
                sys = Subspace.solveEqsStack(stack)
                specSystems += [sys]
            
            designSystems += [specSystems]
        
        return designSystems
        
    def _specIOfreedom(self, spec):

        kms = spec.input + spec.output

        ids = []
        loops = []
        for km in kms:
            
            #if(km.isZero): continue
            ids += [km.id]
            path = self.path(spec.groundId, km.id)
            newLoop = path, km.twist
            loops += [newLoop]
        
        return loops, ids

    def _specDeltaLoops(self, spec, mode="delta"):

        # mode = "delta" or "zero"
        validModes = set(["delta", "zero"])
        if(mode not in validModes): raise Exception(MSG_UNIMPLEMENTED)

        kms = spec.input + spec.output
        if(mode == "zero"): kms = [km for km in kms if not km.isZero]

        if(mode == "delta"): enumerator = itertools.combinations
        elif(mode == "zero"): enumerator = itertools.permutations

        loops = []
        ids = []
        for start, end in enumerator(kms, 2):
            # skip criteria
            if(mode == "zero" and end.isZero): continue

            ids += [(start.id, end.id)]

            if(mode == "delta"): delta = end.twist - start.twist
            elif(mode == "zero"): delta =  -1 * start.twist # set b.twist == 0
            path = self.path(start.id, end.id, exclusion=[spec.groundId])
            loop = path, delta
            loops += [loop]
        
        return loops, ids

    def _specZeroDeltaActs(self, spec):

        kms = spec.input + spec.output

        zeroTwist = np.zeros(SCREW_SPACE_DIM)

        actSpaces = []
        for start, end in itertools.permutations(kms, 2):
            if(start.isZero or end.isZero): continue
            actS = self.findActuation(spec.groundId, start.id, start.twist)
            actE = self.findActuation(spec.groundId, end.id, zeroTwist)
            act = actS.intersect(actE)
            actSpaces += [act]
        
        return actSpaces

    def zeroDeltaSpeedCheck(self):

        curSpace = self.configSubspace()
        X = self._X()

        for spec in self.ioSpecs:
            spaces = self._specZeroDeltaActs(spec)
            for actSpace in spaces:
                if(actSpace is None): continue # zero actuation is impossible, no worries
                
                speedSpace = actSpace.transformBy(X)
                
                actSpeedRef = self.embedSpeed(speedSpace.refPt)
                actSpeedspans = self.embedSpeed(speedSpace.spanSpace)
                actSpeedSpace = Subspace(refPt=actSpeedRef, spans=actSpeedspans)

                solIntersection = curSpace.intersect(actSpeedSpace)
                if(solIntersection is not None):
                    return False
        
        return True

    def configSubspace(self):

        return CMGraph.configToSubspace(self)

    def stageDisp(self, edgeTwists):

        result = []
        for i in range(self.nodeCount):
            path = self.path(self.ioSpecs[0].groundId, i)
            factors = path.reshape(-1, 1)
            summed = np.sum(edgeTwists * factors, axis=-2)
            axis, rot, trans, pos = decompose(summed)
            result += [[axis.tolist(), rot, trans, pos.tolist()]]

        return result

    def edgeEndStage(self):

        ids = []
        for line in self.C:
            id = np.where(line == 1)[0]
            ids += [id]
        
        ids = np.concatenate(ids)

        return ids

    def costVec(self, condGround=False):

        # get base cost vector
        rotMask = np.ones((EUC_SPACE_DIM, EUC_SPACE_DIM)) - np.identity(EUC_SPACE_DIM)

        stack = []
        for eId in range(self.edgeCount):
            eCen = self.E[eId, :EUC_SPACE_DIM]
            nId = np.argwhere(self.C[eId] == 1)
            nCen = self.N[nId, :EUC_SPACE_DIM].flatten()
            pointer = eCen - nCen
            pointerMasked = rotMask @ np.diag(pointer)
            mag = np.linalg.norm(pointerMasked, axis=1)

            cost = np.ones(SCREW_SPACE_DIM)
            cost[:EUC_SPACE_DIM] = mag
            stack += [cost]
        
        costVec = np.concatenate(stack, axis=0)
        costVec[costVec <= EPSILON] = EPSILON # avoid 0
        
        # for each mode, calculate edge distance to ground to condition costs
        if(condGround):
            nodeDist = Graph.crossDist(self.A)
            edgeStartId = np.where(self.C == -1)[1]
            edgeEndId = np.where(self.C == 1)[1]
            
            conditioned = []
            for mode in self.ioSpecs:
                groundId= mode.groundId
                nodeDistToGround = nodeDist[groundId]
                startDistToGround =  nodeDistToGround[edgeStartId]
                endDistToGround = nodeDistToGround[edgeEndId]
                edgeDist = np.minimum(startDistToGround, endDistToGround)
                discount = np.interp(edgeDist, (np.min(edgeDist), np.max(edgeDist)), (1 - EDGE_DIST_DISCOUNT, 1))
                discount = np.repeat(discount, SCREW_SPACE_DIM)
                modeCost = discount * costVec
                conditioned += [modeCost]
            
            conditioned = np.stack(conditioned)
            costVec = np.max(conditioned, axis=-1)

            return conditioned

        costVec = costVec.reshape(-1, 1)

        return costVec

    def getDomain(self):

        nodePts = self.N[:,:EUC_SPACE_DIM]
        edgePts = self.E[:,:EUC_SPACE_DIM]
        allPts = np.concatenate([nodePts, edgePts], axis=0)
        
        if(len(allPts) == 0): return np.zeros((2, EUC_SPACE_DIM))

        minCoord = np.min(allPts, axis=0)
        maxCoord = np.max(allPts, axis=0)

        domain = np.stack([minCoord, maxCoord], axis=0)
        
        return domain

    def ioStageMask(self, combine=False):

        masks = []
        for spec in self.ioSpecs:
            
            mask = np.zeros(self.nodeCount, dtype=np.int32)

            mask[spec.groundId] = 1
            pivs = spec.enumIds()
            mask[pivs] = 1
            
            masks += [mask]
        
        mask = np.stack(masks, axis=0)

        if(combine): mask = np.sum(mask, axis=0)

        return mask

    def closestStage(self, coord):

        if(not isinstance(coord, np.ndarray)): coord = np.asarray(coord)

        nCoords = self.N[:,:EUC_SPACE_DIM]
        delta = nCoords - coord
        dist = np.linalg.norm(delta, axis=-1)
        closestId = np.argmin(dist)

        return closestId

    def linSysFromLoops(self, loops):

        As, Bs = [], []
        for loop, B in loops:
            A = self.embedJFTtoLoop(loop, jftType="full")
            As += [A]
            Bs += [B]

        A = np.concatenate(As, axis=0)
        B = np.concatenate(Bs, axis=0)

        return A, B

    def chain(self, specId): # TEMP

        spec = self.ioSpecs[specId]

        traversed = set()
        ids = []

        i = spec.input[0]
        o = spec.output[0]

        # inputs
        train = Graph.traverse(self.A, spec.groundId, i.id)
        for id in train:
            if(id not in traversed): 
                traversed.add(id)
                ids += [id]
        
        # gear train
        traversed.remove(ids[-1])
        train = Graph.traverse(self.A, i.id, o.id, exclusion=traversed)
        traversed.add(ids[-1])
        for id in train:
            if(id not in traversed): 
                traversed.add(id)
                ids += [id]
        
        # outputs
        traversed.remove(ids[0])
        traversed.remove(ids[-1])
        train = Graph.traverse(self.A, o.id, spec.groundId, exclusion=traversed)
        traversed.add(ids[0])
        traversed.add(ids[-1])
        for id in train:
            if(id not in traversed): 
                traversed.add(id)
                ids += [id]
        
        chain = ids + [ids[0]]

        return chain

    def specChains(self, specId, skipZero=False, trainOnly=False):

        spec = self.ioSpecs[specId]

        chains = []
        for i in spec.input:
            if(skipZero and i.isZero): continue
            for o in spec.output:
                if(skipZero and o.isZero): continue
                chain = self._specIoPairchain(i, o, spec.groundId, trainOnly)
                chains += [chain]
        
        return chains
        
    def _specIoPairchain(self, i, o, groundId, trainOnly=False):
        iId = i.id if isinstance(i, PrescribedMotion) else i
        oId = o.id if isinstance(o, PrescribedMotion) else o
        traversed = set()
        ids = []
        # input
        train = Graph.traverse(self.A, groundId, iId)
        for id in train:
            if(id not in traversed): 
                traversed.add(id)
                ids += [id]
        # gear train
        traversed.remove(iId)
        train = Graph.traverse(self.A, iId, oId, exclusion=traversed)
        if(trainOnly): return train
        traversed.add(iId)
        for id in train:
            if(id not in traversed): 
                traversed.add(id)
                ids += [id]
        # output
        traversed.remove(groundId)
        traversed.remove(oId)
        train = Graph.traverse(self.A, oId, groundId, exclusion=traversed)
        traversed.add(groundId)
        traversed.add(oId)
        for id in train:
            if(id not in traversed): 
                traversed.add(id)
                ids += [id]
        
        chain = ids + [ids[0]]

        return chain

    def splitEdge(self, pivId, ground=None):
        
        # copy original info
        eStart = np.copy(self.N[self.edgeStartId(pivId), :EUC_SPACE_DIM])
        eCen = np.copy(self.E[pivId, :EUC_SPACE_DIM])
        eEnd = np.copy(self.N[self.edgeEndId(pivId), :EUC_SPACE_DIM])

        # split topology
        addedGround = super().splitEdge(pivId, ground=ground)

        # update N
        newN = np.zeros((1, self.N.shape[-1]))
        newN[0, :EUC_SPACE_DIM] = eCen
        self.N = np.append(self.N, newN, axis=0)

        # modify E
        modCen = (eStart + eCen) * .5
        newCen = (eEnd + eCen) * .5
        self.E[pivId, :EUC_SPACE_DIM] = modCen

        # update other E
        newE = np.zeros((1, self.E.shape[-1]))
        newE[0, :EUC_SPACE_DIM] = newCen
        self.E = np.append(self.E, newE, axis=0)

        if(addedGround): # update grounded edge
            newE = np.zeros((1, self.E.shape[-1]))
            gCenter = (eCen + self.N[self.edgeStartId(-1), :EUC_SPACE_DIM]) * .5
            newE[0, :EUC_SPACE_DIM] = gCenter
            self.E = np.append(self.E, newE, axis=0)

        return pivId, self.edgeCount - 1, self.nodeCount - 1 # modded edge, new edge, new node id

    def edgeVec(self, id, unitize=False):

        startPt = self.N[self.edgeStartId(id), :EUC_SPACE_DIM]
        endPt = self.N[self.edgeEndId(id), :EUC_SPACE_DIM]

        vec = startPt - endPt
        if(unitize):
            vLen = np.linalg.norm(vec, axis=-1)
            vLen[vLen < EPSILON] = 1
            vec /= vLen.reshape(-1, 1)

        return vec
    
    def edgeLength(self, id):

        startPt = self.N[self.edgeStartId(id), :EUC_SPACE_DIM]
        endPt = self.N[self.edgeEndId(id), :EUC_SPACE_DIM]

        vec = startPt - endPt
        length = np.linalg.norm(vec, axis=-1)

        return length
        
    def edgeCenter(self, id):

        return self.E[id, :EUC_SPACE_DIM]
    
    def nodeCenter(self, id):

        return self.N[id, :EUC_SPACE_DIM]
    
    def newEdgeFromNodeIds(self, id1, id2):

        Ec = (self.nodeCenter(id1) + self.nodeCenter(id2)) * .5
        Ew = np.zeros(SCREW_SPACE_DIM)

        newE = np.concatenate([Ec, Ew], axis=-1)

        return newE

    def chainEdgesMask(self, skipZero=False, trainOnly=False):

        nominations = []
        for i in range(len(self.ioSpecs)):
            chains = self.specChains(i, skipZero, trainOnly)
            for chain in chains: 
                subMask = self.tracePath(chain)
                nominations += [subMask]
        
        nominations = np.stack(nominations, axis=0)
        nominations = np.any(nominations != 0, axis=0)
        
        mask = np.zeros(self.edgeCount, np.int32)
        mask[nominations] = 1
        
        return mask

    def configGroups(self):

        r = {}
        for i in range(len(self.ioSpecs)):
            io = self.ioSpecs[i]
            cId = io.configId
            r[cId] = r.get(cId, []) + [i]
        
        for key in r:
            r[key] = set(r[key])
        
        return r

    def moveNode(self, nId, newCoord):

        if(not isinstance(newCoord, np.ndarray)):
            newCoord = np.asarray(newCoord)
        
        self.N[nId, :EUC_SPACE_DIM] = newCoord

        eIds = self.adjEdges(nId)
        otherNodeIds = np.stack([self.edgeStartId(eIds), self.edgeEndId(eIds)]).T
        otherNodeIds = otherNodeIds[otherNodeIds != nId]

        otherNodeCoord = self.N[otherNodeIds, :EUC_SPACE_DIM]
        newEdgeCen = (otherNodeCoord + newCoord) * .5
        self.E[eIds, :EUC_SPACE_DIM] = newEdgeCen

    @staticmethod
    def _genConfigVar(ref, spans):

        configs = [np.copy(ref)]
        for i in range(1, len(spans) + 1):
            for ids in itertools.combinations(range(len(spans)), i + 1):
                vecs = np.take(spans, ids, axis=0)
                deltaMask = np.sum(vecs, axis=0)
                config = np.copy(ref)
                config[deltaMask > 0] = 1
                configs += [config]
        
        return configs

    @staticmethod
    def _specSolPivotIds(specSol):

        if(not isinstance(specSol, Subspace)): raise Exception(MSG_INVALID_INPUT)
        
        # find pivot dimensions to set to 0
        vecs = specSol.spanSpace.T # span vectors along dimensions (to eq zero)
        vecsNorm = np.linalg.norm(vecs, axis=-1)
        pivots = np.where(vecsNorm > EPSILON)[0] # entries that are adjustable
        stills = np.where(vecsNorm < EPSILON)[0] # entries that are not adjustable
        
        pivFilter, aligmentTriu = CMGraph._simplifyPivot(specSol, pivots)
        pivMap = CMGraph._createPivMap(specSol, pivots, stills, pivFilter, aligmentTriu)
        
        pivotsTrunc = pivots[pivFilter] # return truncated pivots

        return pivotsTrunc, pivMap

    @staticmethod
    def _simplifyPivot(specSol, pivots):
        
        pivA = specSol.spanSpace.T[pivots]
        pivB = specSol.refPt[pivots]

        # compare and remove vectors that are parallel (using dot product compare)
        pivVecs = np.concatenate([pivA, pivB.reshape((-1, 1))], axis=-1)
        pivVecsNorm = np.linalg.norm(pivVecs, axis=1).reshape((-1, 1))
        pivVecsNormed = pivVecs / pivVecsNorm
        dotted = np.matmul(pivVecsNormed, pivVecsNormed.T)
        dotted = np.abs(dotted)
        alignment = (np.abs(dotted - 1) < EPSILON).astype(np.int32) # parallel
        alignment = alignment
        aligmentTriu = np.triu(alignment)
        degree = np.sum(aligmentTriu, axis=0) # if > 1, entry pivots on a previous dimension
        pivFilter = np.where(degree <= 1)[0]

        return pivFilter, aligmentTriu

    @staticmethod
    def _createPivMap(specSol, pivots, stills, pivFilter, aligmentTriu):
        # construct pivot map, where the diagonal is the basis and each row is 
        # a dependency map (first entry is the main-pivot)

        # remap truncated to untruncated pivots. Truncated pivots will become 
        # a zero row (except for the diagonal of 1).
        pivMapTrunc = np.zeros((len(pivots), len(pivots)), dtype=np.int32)
        pivMapTrunc[pivFilter] = aligmentTriu[pivFilter]

        # remap pivots map back to the full dimension
        nz = np.where(pivMapTrunc != 0)
        nzRemapped = pivots[nz[0]], pivots[nz[1]]

        pivMap = np.zeros((specSol.dim, specSol.dim), dtype=np.int32)
        stillsVal = np.take(specSol.refPt, stills)
        stillsNz = (np.abs(stillsVal) > EPSILON).astype(np.int32)
        pivMap[nzRemapped] = 1 # assign pivots dependency
        pivMap[stills, stills] = stillsNz # assign sill dimension configuration
        pivMap[pivots, pivots] = 1 # assign pivots configuration

        return pivMap

    @staticmethod
    def configToSubspace(input):

        input = CMGraph._inp2cfgMtrx(input)
        input = input.flatten()

        spans = np.diag(input)
        mag = np.sum(spans, axis=-1)
        spans = spans[mag > EPSILON]

        space = Subspace(spans=spans)

        return space

    @staticmethod
    def configValidityCheck(input):

        input = CMGraph._inp2cfgMtrx(input)
        transValid = CMGraph._transCheck(input)

        return transValid
    
    @staticmethod
    def _inp2cfgMtrx(input):

        # type check and conversion
        if(isinstance(input, CMGraph)):
            input = input.config()
        elif(isinstance(input, list) or isinstance(input,tuple)):
            input = np.asarray(input)

        if(isinstance(input, np.ndarray)):
            if(input.ndim != 2 and input.shape[-1] != SCREW_SPACE_DIM):
                input = input.reshape((-1, SCREW_SPACE_DIM))
        else:
            raise Exception(MSG_UNIMPLEMENTED)
        
        return input

    @staticmethod
    def _transCheck(input):

        transCfg = input[:,EUC_SPACE_DIM:]
        summed = np.sum(transCfg, axis=-1)
        isValidTrans = np.abs(summed) < 3 # no more than 2 (included) trans parts
        
        summedAll = np.sum(input, axis=-1)
        is6dof = summedAll == 6

        edgeValid = np.logical_or(isValidTrans, is6dof)
        isValid = np.all(edgeValid)

        return isValid

    @staticmethod
    def _toJFT(spec, frame=None, refPtOverride=False):

        # determine input data
        sysOverride = isinstance(frame, Frame)
        refOverride = sysOverride and refPtOverride
        
        basis = frame.system if sysOverride else np.identity(EUC_SPACE_DIM)
        refPt = frame.origin if refOverride else spec[:EUC_SPACE_DIM]

        # precompute parts
        rotRef = np.cross(refPt, basis)
        transDirPart = np.zeros((EUC_SPACE_DIM, EUC_SPACE_DIM))

        # assemble full dof/c space
        dirPart = np.concatenate((basis, transDirPart), axis=0)
        posPart = np.concatenate((rotRef, basis), axis=0)
        dofs = np.concatenate((dirPart, posPart), axis=-1)
        dofs[dofs == 0] = 0 # remove -0

        # filter dof/c space
        dofFlag = spec[EUC_SPACE_DIM:EUC_SPACE_DIM + SCREW_SPACE_DIM]
        mask = np.nonzero(dofFlag)
        masked = dofs[mask]

        return masked

    @staticmethod
    def initDenseJSON(json):

        segX, segY, segZ = json["res"]
        axes = np.asarray(json["axes"], dtype=np.float64)
        base = json["base"]
        diag = json["diag"]

        # init node coords and ids
        nId = 0
        nCoord, nIdMap, nIdMapInv = {}, {}, {}
        for i in range(segX):
            for j in range(segY):
                for k in range(segZ):

                    id = (i, j, k)

                    fX = i / (segX - 1) if segX != 1 else .5
                    fY = j / (segY - 1) if segY != 1 else .5
                    fZ = k / (segZ - 1) if segZ != 1 else .5
                    factor = [fX, fY, fZ]
                    factor = np.asarray(factor)
                    disp = axes.T @ factor
                    
                    nCoord[id] = base + disp
                    nIdMap[id] = nId
                    nIdMapInv[nId] = id
                    nId += 1

        Nc = np.stack([nCoord[nIdMapInv[id]] for id in range(nId)], axis=0)
        Nbc = np.zeros((nId, 3))
        Ng = np.zeros((nId, SCREW_SPACE_DIM))
        
        edgePointer = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        if(diag >= 1):
            edgePointer += [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
        if(diag >= 2):
            edgePointer += [(1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]

        # init edges    
        Ec = []
        adjs = []

        for i in range(segX):
            for j in range(segY):
                for k in range(segZ):   
                    curId = (i, j, k)

                    for p in edgePointer:
                        tarId = (i + p[0], j + p[1], k + p[2])
                        if(tarId not in nCoord): continue
                        edgeCoord = (nCoord[curId] + nCoord[tarId]) / 2
                        Ec += [edgeCoord]
                        adjs += [(nIdMap[curId], nIdMap[tarId])]
        
        Ew = np.ones((len(adjs), SCREW_SPACE_DIM))
        
        # craete graph
        N = np.concatenate([Nc, Nbc, Ng], axis=-1)
        if(len(Ec) > 0):
            E = np.concatenate([Ec, Ew], axis=-1)
        else:
            E = np.zeros((0, EUC_SPACE_DIM + SCREW_SPACE_DIM), dtype=np.float64)
        C = pairs2matrix(adjs)

        return CMGraph(C=C, N=N, E=E)

    @staticmethod
    def fromJSON(json):
        
        topoTarget = json["topology"] if "topology" in json else json
        graph = CMGraph._topoFromJSON(topoTarget)

        if(not "objectives" in json): return graph

        graph.ioSpecs = CMGraph._objFromJSON(json["objectives"])
        
        return graph

    @staticmethod
    def _topoFromJSON(json):

        #if(not ("nodes" in json and "edges" in json)): return None

        if("parser" in json): # loading from design tool save 
            nodesJSON = json["parser"]["nodeCoord"]
            edgesJSON = json["parser"]["edgePair"]
            isCull = False
        else: # loading from design tool message
            nodesJSON = json["nodeCoord"] if "nodeCoord" in json else json["nodes"]
            edgesJSON = json["edgePair"] if "edgePair" in json else json["edges"]
            isCull = "nodesCull" in json

        # construct nodes array and filter
        nodes = np.asarray(nodesJSON)
        cullMask = np.zeros(len(nodes))
        if(isCull): cullMask[json["nodesCull"]] = 1
        
        # reconstruct C and cull invalid edges
        C = pairs2matrix(edgesJSON, len(nodes))
        if(len(C) != 0):
            adjCull = C * cullMask.reshape(1, -1)
            edgeFilter = np.linalg.norm(adjCull, axis=-1) < EPSILON
            cleanedC = C[edgeFilter]
            cleanedC = cleanedC[:, cullMask == 0]

            C = cleanedC

        # compose node feature matrix
        Nc = np.delete(nodes, json["nodesCull"], axis=0) if isCull else nodes
        Ng = np.zeros((len(Nc), SCREW_SPACE_DIM))
        Nbc = np.zeros((len(Nc), 3))
        N = np.concatenate([Nc, Nbc, Ng], axis=-1)

        # compose edge feature matrix
        Ec = (Nc[np.where(C == -1)[1]] + Nc[np.where(C == 1)[1]]) * .5
        Ew = np.zeros((len(Ec), SCREW_SPACE_DIM))
        E = np.concatenate([Ec, Ew], axis=-1)

        return CMGraph(C=C, N=N, E=E)

    @staticmethod
    def _objFromJSON(json):

        specs = []

        for m in json["modes"]:
            
            inp, out = [], []
            for io in m["pres"]:
                
                vel = np.asarray(io["vel"])
                piv = np.asarray(io["pivot"])
                full = fullSpace(piv).T
                twist = full @ vel
                p = PrescribedMotion(io["stageId"], twist, piv, vel)
                
                if(io["kind"] == "inp"): inp += [p]
                elif(io["kind"] == "out"): out += [p]
                
            spec = KinematicIO(m["groundId"], inp, out, m["configId"])
            specs += [spec]
        
        return specs

def pairs2matrix(pairs, nodeCount=None):
   
    pairCount = len(pairs)
   
    if(nodeCount is None):
        maxId = -1
        for pair in pairs:
            assert len(pair) == 2
            maxId = max(maxId, pair[0], pair[1])
    
        nodeCount = maxId + 1

    C = np.zeros((pairCount, nodeCount))
    for i in range(pairCount):
        pair = pairs[i]
        C[i, pair[0]] = -1
        C[i, pair[1]] = 1

    return C

def initDense(seg=None, coordRange=(0, 1), diag=0):

    # segment per dim
    if(seg is None):
        raise Exception(MSG_UNIMPLEMENTED)
    elif(isinstance(seg, int)):
        segX, segY, segZ = seg, seg, seg
    elif((isinstance(seg, list) or isinstance(seg, tuple)) and\
          len(seg) == EUC_SPACE_DIM):
        segX, segY, segZ = seg
    
    if(isinstance(coordRange, int) or isinstance(coordRange, float)):
        domain = 0, coordRange
        rangeX, rangeY, rangeZ = domain, domain, domain
    elif(isNdNumerical(coordRange, 1)):
        domain = coordRange[0], coordRange[1]
        rangeX, rangeY, rangeZ = domain, domain, domain
    elif(isNdNumerical(coordRange, 2)):
        rangeX = coordRange[0][0], coordRange[0][1]
        rangeY = coordRange[1][0], coordRange[1][1]
        rangeZ = coordRange[2][0], coordRange[2][1]
    else:
        raise Exception(MSG_UNIMPLEMENTED)
    
    # init node coords and ids
    nId = 0
    nCoord, nIdMap, nIdMapInv = {}, {}, {}
    for i in range(segX):
        for j in range(segY):
            for k in range(segZ):

                id = (i, j, k)
                xCoord = (i / (segX - 1)) * (rangeX[1] - rangeX[0]) + rangeX[0]
                yCoord = (j / (segY - 1)) * (rangeY[1] - rangeY[0]) + rangeY[0]
                zCoord = (k / (segZ - 1)) * (rangeZ[1] - rangeZ[0]) + rangeZ[0]
                nCoord[id] = np.asarray([xCoord, yCoord, zCoord])
                nIdMap[id] = nId
                nIdMapInv[nId] = id
                nId += 1
    
    Nc = np.stack([nCoord[nIdMapInv[id]] for id in range(nId)], axis=0)
    Nbc = np.zeros((nId, 3))
    Ng = np.zeros((nId, SCREW_SPACE_DIM))

    edgePointer = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if(diag >= 1):
        edgePointer += [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
    if(diag >= 2):
        edgePointer += [(1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]

    # init edges    
    Ec = []
    adjs = []

    for i in range(segX):
        for j in range(segY):
            for k in range(segZ):   
                curId = (i, j, k)

                for p in edgePointer:
                    tarId = (i + p[0], j + p[1], k + p[2])
                    if(tarId not in nCoord): continue
                    edgeCoord = (nCoord[curId] + nCoord[tarId]) / 2
                    Ec += [edgeCoord]
                    adjs += [(nIdMap[curId], nIdMap[tarId])]
    
    Ew = np.ones((len(adjs), SCREW_SPACE_DIM))
    
    # craete graph
    N = np.concatenate([Nc, Nbc, Ng], axis=-1)
    E = np.concatenate([Ec, Ew], axis=-1)
    C = pairs2matrix(adjs)

    return CMGraph(C=C, N=N, E=E)

if __name__ == "__main__":
    pass