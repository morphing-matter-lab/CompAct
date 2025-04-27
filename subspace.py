import numpy as np
import sympy as sp
import copy
from scipy.linalg import null_space
from messages import *

from parameters import *
from utils import fixFloat, solve, nullspace

np.set_printoptions(precision=NUM_PREC, suppress=True, linewidth=100000, edgeitems=30)

class Subspace(object):
    def __init__(self, normals=None, refPt=None, parameters=None, spans=None, dim=None):
        # null space as row vector (N, M), N: number of null spaces, M: dimension
        # ref pt is a point on the subspace (D): dimensions
        # parameters are the variables(symbols) of the subspace (M), M: number of parameters

        self.dim = None

        self.normalSpace = None
        self.spanSpace = None
        self.refPt = None

        self.parameters = None
        self.isPoint = None
        self.isFull = None
        self.normalRank = 0
        self.spanRank = 0

        self.subsetSpaces = [] # for multimodal systems

        self._initialize(normals, spans, refPt, parameters, dim)
        
    def __eq__(self, other):
        
        # atrribute check
        if(not isinstance(other, Subspace)): return False
        if(self.dim != other.dim): return False
        if(self.normalRank != other.normalRank): return False
        if(self.spanRank != other.spanRank): return False

        # check points (shortcut)
        if(self.isPoint):
            dist = abs(np.linalg.norm(self.refPt - other.refPt))
            return dist <= EPSILON
        
        # check if multually share position
        selfPtOnOther = other.isPtOnSubspace(self.refPt)
        otherPtOnSelf = self.isPtOnSubspace(other.refPt)
        if((not selfPtOnOther) or (not otherPtOnSelf)): return False

        # check if normal/span sapces are mutually complementary
        complementary = Subspace.isComplementary(self.normalSpace, other.spanSpace) and\
                        Subspace.isComplementary(self.spanSpace, other.normalSpace)
        if(not complementary): return False

        return True
        
    def __hash__(self):

        infoList = [self.dim, self.normalRank, self.spanRank]
        infoList += list(self.normalSpace.reshape(-1)) + list(self.spanSpace.reshape(-1))
        return hash(infoList)
    
    def __repr__(self):

        msg = "SubSpace%dD(Normals:%d, Spans:%d)" %\
              (self.dim, self.normalRank, self.spanRank)

        return msg

    def _initialize(self, normals, spans, refPt, parameters, dim):
        
        normals, spans, refPt, parameters = self._formatInput(normals, spans, refPt, parameters)

        # only dim was supplied, initialize the entire dimensional space
        if(dim is not None): 
            self.dim = dim
            self._initFullSpace()
        
        # both normals and spans were supplied, initialize by check and copy
        if(normals is not None and spans is not None):
            assert normals.shape[-1] == spans.shape[-1], MSG_UNIMPLEMENTED
            self.dim = normals.shape[-1]
            self._initNormalAndSpan(normals, spans)

        # normals was supplied but not spans, initialize by solving spans
        elif(normals is not None and spans is None):
            self.dim = normals.shape[-1]
            self._initByNormals(normals)

        # spans was supplied but not normals, initialize by solving normals
        elif(normals is None and spans is not None):
            self.dim = spans.shape[-1]
            self._initBySpans(spans)

        # neither was supplied, initialize as a point
        elif(normals is None and spans is None):
            self.dim = refPt.shape[-1]
            self._initAsPoint()
        
        # initialize reference point
        if(refPt is not None): 
            refPt = fixFloat(refPt)
            refPt[refPt == 0] = +0
            self.refPt = refPt
        else: self.refPt = np.zeros(self.dim)

        self._initPrms(parameters)
        
        self._updateInfo()

    def _formatInput(self, normals, spans, refPt, parameters):

        # normals
        if(isinstance(normals, np.ndarray) and normals.size != 0):
            normals = normals.reshape((-1, normals.shape[-1]))
        elif(isinstance(normals, list) or isinstance(normals, tuple)):
            normals = np.asarray(normals)
        else:
            normals = None
        
        # spans 
        if(isinstance(spans, np.ndarray) and spans.size != 0):
            spans = spans.reshape((-1, spans.shape[-1]))
        elif(isinstance(spans, list) or isinstance(spans, tuple)):
            spans = np.asarray(spans)
        else:
            spans = None
        
        # reference point
        if(isinstance(refPt, np.ndarray)):
            refPt = refPt.reshape(-1)
        elif(isinstance(refPt, list) or isinstance(refPt, tuple)):
            refPt = np.asarray(refPt)
        else:
            refPt = None

        # parameters
        if(isinstance(parameters, np.ndarray)):
            parameters = parameters.reshape(-1)
        elif(isinstance(parameters, list) or isinstance(parameters, tuple)):
            parameters = np.asarray(parameters)
        else:
            parameters = None

        return normals, spans, refPt, parameters

    def _initFullSpace(self):
        
        self.normalSpace = np.asarray([[]])
        self.spanSpace = np.identity(self.dim)
        
    def _initNormalAndSpan(self, normals, spans):

        normals = Subspace.fold(normals)
        normals = Subspace.simplifySpace(normals, True, True)
        normalRank = normals.shape[0]

        spans = Subspace.fold(spans)
        spans = Subspace.simplifySpace(spans, True, True)
        spanRank = spans.shape[0]
        checker = np.matmul(normals, spans.T)
        checkerNorm = np.linalg.norm(checker)

        assert checkerNorm < EPSILON, MSG_UNIMPLEMENTED # orthogonal
        assert normalRank + spanRank == self.dim, MSG_UNIMPLEMENTED # complete
        
        self.normalSpace = normals
        self.spanSpace = spans

    def _initByNormals(self, normals):
        
        if(normals.size == 0 or np.linalg.norm(normals) < EPSILON):
            self.normalsSpace = np.asarray([[]])
            self.spanSpace = np.identity(self.dim)
            return
        
        if(SUBSPACE_INIT_POSTPROC):
            normals = Subspace.simplifySpace(normals, True, True)
            normals = Subspace.fold(normals)
            #normals = Subspace.simplifySpace(normals, True, True)

        spans = nullspace(normals)
        
        if(SUBSPACE_INIT_POSTPROC):
            spans = Subspace.simplifySpace(spans, True, True)

        # rank check
        normalRank = normals.shape[0]
        spanRank = spans.shape[0]
        if(normalRank + spanRank != self.dim):# complete
            if(spans.size == 0):
                normals = np.identity(spans.shape[1])
            else:
                normals = nullspace(spans)
            #print(normalRank, spanRank, self.dim)
            #raise Exception(MSG_UNIMPLEMENTED)

        self.normalSpace = normals
        self.spanSpace = spans

    def _initBySpans(self, spans):
        
        if(spans.size == 0 or np.linalg.norm(spans) < EPSILON):
            self.spanSpace = np.asarray([[]])
            self.normalSpace = np.identity(self.dim)
            return
        
        if(SUBSPACE_INIT_POSTPROC):
            spans = Subspace.simplifySpace(spans, True, True)
            spans = Subspace.fold(spans)
            #spans = Subspace.simplifySpace(spans, True, True)
        
        normals = nullspace(spans)
        
        if(SUBSPACE_INIT_POSTPROC):
            normals = Subspace.simplifySpace(normals, True, True)
            
        if(normals.size == 0): # if span is full ranked, convert it into identity matrix
            spans = np.identity(self.dim)

        spanRank = spans.shape[0]
        normalRank = normals.shape[0]
        
        if(normalRank + spanRank != self.dim):# complete
            spans = nullspace(normals)
            #print(normalRank, spanRank, self.dim)
            #raise Exception(MSG_UNIMPLEMENTED)

        self.normalSpace = normals
        self.spanSpace = spans

    def _initAsPoint(self):

        self.spanSpace = np.asarray([[]])
        self.normalSpace = np.identity(self.dim)

    def _initPrms(self, parameters):

        # parameters
        if(not isinstance(parameters, np.ndarray)):
            names = ' '.join([chr(ord("a") + i) for i in range(self.dim)])
            self.parameters = np.asarray(sp.symbols(names))
        else:
            self.parameters = parameters
        
    def _getDataFrom(self, other):

        self.dim = other.dim

        self.normalSpace = other.normalSpace
        self.spanSpace = other.spanSpace
        self.parameters = other.parameters
        self.refPt = other.refPt

        self._updateInfo()

    def _sanityCheck(self):

        # check normals and spans
        assert Subspace.isComplementary(self.normalSpace, self.spanSpace),\
               "normal and span space are not orthogonal"
        
        # check ref point on space
        assert self.isPtOnSubspace(self.refPt), "refPt not on plane"

        # check if subspace has parameters
        assert self.parameters.size == self.dim, "parameters not right"

        # check all vectors are unitized
        bases = self.getBases()
        length = np.linalg.norm(bases, axis=-1)
        deviation = abs(length) - 1
        assert np.all(deviation <= EPSILON), "bases are not unitized"

        # check all vectors are orthogonal
        for i in range(self.dim):
            for j in range(self.dim):
                if(i < j):
                    dotProd = np.dot(bases[i], bases[j])
                    assert abs(dotProd) <= EPSILON, "bases not orthogonal"

    def _updateInfo(self):
        
        self.spanRank = int(self.spanSpace.size / self.dim)
        self.normalRank = int(self.normalSpace.size / self.dim)
        self.isPoint = self.spanRank == 0
        self.isFull = self.normalRank == 0

        if(SANITY_CHECK): self._sanityCheck()
    
    def _intersectNonVectorized(self, other):
        # note: other allways has a lower or equal solution space rank than self
        
        if(self.isParallel(other)): # two subspaces are parallel
            return None
        elif(self.includesSubspace(other)):
            return other
        
        # find intersection subspace by expanding normals
        normals = self.normalSpace
        newNormals = []
        for vec in other.normalSpace:
            # expand normal
            expanded = Subspace.expand(normals, vec)
            if(expanded.shape == normals.shape): # nothing happened
                continue
            newNormals += [expanded[-1]]
            normals = expanded
        
        newNormals = np.stack(newNormals)
        
        if(self.isPtOnSubspace(self.refPt) and other.isPtOnSubspace(self.refPt)):
            refPt = self.refPt
        else:
            symbs = sp.symbols(' '.join(["t%d" % i for i in range(len(newNormals))]))
            symbs = np.asarray(symbs).reshape(-1)
            correction = np.sum(newNormals * symbs.reshape((-1, 1)), axis=0)
            corrected = self.refPt + correction
            pointerVec = other.refPt - corrected
            otherNormalDot = np.dot(other.normalSpace, pointerVec)
            eqs = []
            for dotProd in otherNormalDot:
                if(len(dotProd.free_symbols) == 0):
                    if(abs(dotProd) <= EPSILON): continue
                    else: return None
                else:
                    eqs += [sp.Eq(dotProd, 0)]

            factor = solve(eqs, symbs)
            refPt = self.refPt + np.sum(newNormals * factor.reshape(-1, 1), axis=0)
            
            if(SANITY_CHECK):
                assert (self.isPtOnSubspace(refPt) and\
                        other.isPtOnSubspace(refPt)), "implementation error"
            
        # constrcut new subspace
        intersection = Subspace(normals=normals, 
                                refPt=refPt, 
                                parameters=self.parameters)
        
        return intersection
    
    def _intersectVectorized(self, other):
        # note: other allways has a lower or equal solution space rank than self

        A = np.concatenate([self.normalSpace, other.normalSpace], axis=0)
        selfB = np.matmul(self.normalSpace, self.refPt)
        otherB = np.matmul(other.normalSpace, other.refPt)
        B = np.concatenate([selfB, otherB], axis=0)
        sol = Subspace.solveLinearSys(A, B)

        return sol
    
    def _intersect(self, other):

        if(SUBSPACE_VECTORIZED_INTERSECT):
            return self._intersectVectorized(other)
        else:
            return self._intersectNonVectorized(other)

    def _isParallelVec(self, span):

        if(self.isFull):
            return False
        
        for vec in span:
            dotProd = np.dot(self.normalSpace, vec)
            if(np.any(abs(dotProd) > EPSILON)):
                return False

        return True

    def _isParallelSubspace(self, other):

        pointer = other.refPt - self.refPt
        # get all span vectors
        catList = []
        if(self.spanRank > 0): catList += [self.spanSpace]
        if(other.spanRank > 0): catList += [other.spanSpace]
        if(len(catList) == 0): # shortcut: both are points, evaluate distance
            return np.linalg.norm(pointer) > EPSILON
        spans = np.concatenate(catList, axis=0)

        spansOrtho = np.asarray([[]])
        for vec in spans: spansOrtho = Subspace.expand(spansOrtho, vec)

        correction = Subspace.project(pointer, spansOrtho)
        residual = pointer - correction
        isParallel = np.any(abs(residual) > EPSILON) # there are components that cannot be eliminated

        return isParallel

    def _cpPoint(self, pt):

        if(self.isFull): return pt # full space special case
        
        vec = self.refPt - pt
        scale = np.dot(self.normalSpace, vec).reshape((-1, 1))
        corrections = self.normalSpace * scale
        correction = np.sum(corrections, axis=0)
        
        closestPt = pt + correction

        if(SANITY_CHECK):
            assert self.isPtOnSubspace(closestPt), "implementation error"

        return closestPt

    def _cpSubspace(self, other):

        if(self.isFull): # handle edge cases
            if(other.isPoint): return other.refPt
            else: return other
        
        if(not self.isParallel(other)): # two subspaces would intersect
            return self.intersect(other)
        else: # tow subspaces are parallel
            # step 1: pull other to self
            cloned = Subspace.clone(other)
            pointer = self.refPt - other.refPt
            # eliminate the "normal" part from both subspaces
            spansCollected = self.spanSpace

            for vec in other.spanSpace:
                spansCollected = Subspace.expand(spansCollected, vec)
            
            if(spansCollected.size != 0):
                spansProj = Subspace.project(pointer, spansCollected)
            else:
                spansProj = np.zeros(pointer.shape)

            correction = pointer - spansProj
            cloned.move(correction)

            # find intersection
            closest = self.intersect(cloned)

            return closest

    def move(self, vec):

        self.refPt = self.refPt + vec

    def reCenter(self, coord, ignoreSpanSpace=False):

        if(ignoreSpanSpace):
            self.refPt = coord
        
        else:
            newPt = self.closestPoint(coord)
            newPt = fixFloat(newPt)
            self.refPt = newPt

    def expandSpan(self, vec):
        
        # check if vectorhas zero length
        if(np.linalg.norm(vec) <= EPSILON):
            return # no action needed

        expanded = Subspace.expand(self.spanSpace, vec)
        if(self.spanSpace.size == expanded.size): # no changes, skip
            pass
        
        isFull = expanded.size / self.dim == self.dim
        if(isFull): normals = np.asarray([[]])
        else: normals = null_space(expanded).T
        
        self.spanSpace = expanded
        self.normalSpace = normals
        self._updateInfo()

    def expandNormal(self, vec):

        expanded = Subspace.expand(self.normalSpace, vec)
        if(self.normalSpace.shape == expanded.shape): # no changes, skip
            pass
        
        isFull = expanded.size / self.dim == self.dim
        if(isFull): spans = np.asarray([[]])
        else: spans = null_space(expanded).T

        self.spanSpace = spans
        self.normalSpace = expanded
        self._updateInfo()

    def removeSpan(self, vec):

        removed = Subspace.remove(self.spanSpace, vec)
        if(self.spanSpace.shape == removed.shape):
            pass

        isZero = removed.size == 0
        if(isZero): normals = np.eye(self.dim)
        else: normals = null_space(removed).T

        self.spanSpace = removed
        self.normalSpace = normals
        self._updateInfo()

    def removeNormal(self, vec):
        
        removed = Subspace.remove(self.normalSpace, vec)
        if(self.normalSpace.shape == removed.shape):
            pass

        isZero = removed.size == 0
        if(isZero): spans = np.eye(self.dim)
        else: spans = null_space(removed).T

        self.spanSpace = spans
        self.normalSpace = removed
        self._updateInfo()

    def transformBy(self, transformation, pivot="spans"):

        # check if pivot is valid
        if(pivot != "normals" and pivot != "spans"):
            raise Exception(MSG_UNIMPLEMENTED)
        
        # process reference point
        newRefPt = np.matmul(transformation.T, self.refPt)
        
        # transform spans
        tarSpace = self.normalSpace if pivot == "normals" else self.spanSpace
        if(tarSpace.size == 0): 
            vecs = np.asarray([[]]).astype(np.float64)
        else:
            vecs = np.matmul(transformation.T, tarSpace.T).T
            vecs = Subspace.simplifySpace(vecs, True, True) # added True True
        
        if(pivot == "normals"):
            transformed = Subspace(refPt=newRefPt, normals=vecs)
        elif(pivot == "spans"):
            transformed = Subspace(refPt=newRefPt, spans=vecs)

        return transformed

    def printInfo(self, fullReport=False, printSpan=False, printPrm=False, \
                  printNormal=False, printRefPt=False, printTitle=False,
                  echelonForm=False):

        print("Subspace entity (Dim: %d, Normals: %d, Spans: %d)" %\
              (self.dim, self.normalRank, self.spanRank))
        """
        if(fullReport or printPrm):
            print("Parameters:")
            print(self.parameters)
        """
        if(fullReport or printRefPt):
            print("point on subspace:")
            print(self.refPt)

        if(fullReport or printSpan):
            spans = self.spanSpace
            if(echelonForm): spans = Subspace.simplifySpace(spans)
            print("Spans:")
            print(spans)

        if(fullReport or printNormal):
            spans = self.normalSpace
            if(echelonForm): spans = Subspace.simplifySpace(spans)
            print("Normals:")
            print(spans)
        
        print()

    def includesSubspace(self, other):

        if(self.isFull):
            return True
        elif(self.isPoint and other.isPoint):
            return np.linalg.norm(self.refPt - other.refPt) <= EPSILON
        # check if span is included
        for vec in other.spanSpace:
            dotProd = np.dot(self.normalSpace, vec)
            if(np.any(abs(dotProd) > EPSILON)): return False
        
        # check if other conincide with self
        isOn = self.eval(other.refPt) <= EPSILON

        return isOn

    def isSubspaceOf(self, other):

        return other.includesSubspace(self)

    def isParallel(self, other):
        # parallel: two objects do not intersect

        if(isinstance(other, Subspace)):
            span = other.spanSpace
            vecInput = False
        else: # a vector
            span = other.reshape((-1, self.dim))
            vecInput = True
        
        if(vecInput):
            return self._isParallelVec(span)
        else:
            return self._isParallelSubspace(other)

    def isPtOnSubspace(self, pt):
        
        if(self.normalRank != 0):
            #dist = abs(self.eval(pt))
            vec = pt - self.refPt
            dot = self.normalSpace @ vec
            onPlane = np.linalg.norm(dot) < EPSILON
        else:
            # normalRank = 0, full space
            onPlane = True
            #dist = np.linalg.norm(self.refPt - pt)
        
        #    onPlane = dist <= EPSILON

        return onPlane

    def closestPoint(self, other):
        
        if(isinstance(other, np.ndarray)):
            return self._cpPoint(other)
        elif(isinstance(other, Subspace)):
            return self._cpSubspace(other)

    def minDistTo(self, other, ord=2):
        
        if(isinstance(other, np.ndarray)):
            return self.ptDist(other, ord) # abs(self.eval(other))
        elif(isinstance(other, Subspace)):
            cp1 = self.closestPoint(other)
            cp2 = other.closestPoint(self)
            if(cp1 == cp2): # intersection found
                return 0
            else:
                return abs(cp1.eval(cp2.refPt))
        
    def eval(self, params):
        
        dotProd = np.dot(self.normalSpace, params)
        evalDist = np.linalg.norm((dotProd.reshape(-1)))
        offSetDotProd = np.dot(self.normalSpace, self.refPt)
        offset = np.linalg.norm((offSetDotProd.reshape(-1)))

        result = evalDist - offset

        return result
    
    def ptDist(self, pt, ord=2):
        # assuming self is otho-unitized

        delta = pt - self.refPt
        normalProj = np.matmul(self.normalSpace, delta)
        dist = np.linalg.norm(normalProj, ord=ord)

        return dist

    def eqs(self):

        equations = []
        for normal in self.normalSpace:
            lhs = np.dot(normal, self.parameters)
            rhs = np.dot(normal, self.refPt)
            eq = sp.Eq(lhs, rhs)
            equations += [eq]
        
        return equations

    def intersect(self, other, debug=False):
        
        # make sure self has higher span rank than other
        if(self.spanRank < other.spanRank):
            return other.intersect(self)

        # cases dispathcer
        if(self.isFull): # self is full rank, simply return the other
            if(debug): print("intersect: full")
            return other
            """
        elif(self == other): # special case, self and other are identical
            if(debug): print("intersect: identical")
            return self
            """
        elif(self.isPoint and other.isPoint): # both are points
            if(debug): print("intersect: two points")
            dist = np.linalg.norm(self.refPt - other.refPt)
            if(dist <= EPSILON): return self
        elif((not self.isPoint) and other.isPoint):
            if(debug): print("intersect: one point")
            if(self.isPtOnSubspace(other.refPt)): return other
        else:
            if(debug): print("intersect: subspaces")
            return self._intersect(other)
        
        # float point edge case
        if(self.minDistTo(other) <= EPSILON):
            # use cp
            return other.closestPoint(self)

        return None
    
    def outputRefPtSpan(self, spanOnly=False):

        result = [] if spanOnly else [self.refPt]

        for span in self.spanSpace:
            if (len(span) != self.dim):
                continue
            result += [span]
        
        result = np.stack(result)

        return result

    def getBases(self):

        if(self.isFull):
            bases = self.spanSpace
        elif(self.isPoint):
            bases = self.normalSpace
        else:
            bases = np.concatenate((self.spanSpace, self.normalSpace), axis=0)

        return bases

    def antiSpace(self):

        refPt= np.copy(self.refPt)
        spans = np.copy(self.normalSpace)

        return Subspace(spans=spans, refPt=refPt)

    def copy(self):

        if(self.spanSpace.size != 0):
            new = Subspace(spans=np.copy(self.spanSpace), refPt=np.copy(self.refPt))
        else:
            new = Subspace(normals=np.copy(self.normalSpace), refPt=np.copy(self.refPt))
        
        return new

    def save(self, path):

        np.savez(path, 
                 refPt=self.refPt, 
                 spans=self.spanSpace, 
                 normals=self.normalSpace)

    @staticmethod
    def project(vec, space):

        spaceNorm = np.linalg.norm(space, axis=-1).reshape((-1, 1))
        spaceUnit = space / spaceNorm
        dotProd = np.dot(spaceUnit, vec).reshape((-1, 1))
        vecs = spaceUnit * dotProd
        proj = np.sum(vecs, axis=0)

        return proj

    @staticmethod
    def fold(vecs):

        result = None
        for vec in vecs:
            result = Subspace.expand(result, vec)
        
        return result

    @staticmethod
    def expand(space, vec):

        vecNorm = np.linalg.norm(vec)
        if(abs(np.linalg.norm(vec)) <= EPSILON): 
            return space
        if(space is None or space.size == 0):
            return vec.reshape((1, -1))
        
        vecNormed = vec / vecNorm
        correction = Subspace.project(vecNormed, space)
        residual = vecNormed - correction
        if(np.all(abs(residual) <= EPSILON)): return space

        newVec = residual / np.linalg.norm(residual)
        catList = (space, newVec.reshape((1, -1)))
        
        return np.concatenate(catList, axis=0)
            
    @staticmethod
    def remove(space, vec):

        if(abs(np.linalg.norm(vec)) <= EPSILON): 
            return space
        if(space.size == 0):
            return space

        dotProd = np.dot(space, vec)
        if(np.all(abs(dotProd) <= EPSILON)): # perpendicular to all
            return space
        else:
            normalSpace = null_space(space).T
            normalExpanded = Subspace.expand(normalSpace, vec)
            if(normalExpanded.size == normalSpace.size): return space
            removedSpace = null_space(normalExpanded).T

            return removedSpace

    @staticmethod
    def isComplementary(space1, space2):
        
        if(space1.size == 0):
            rank = np.linalg.matrix_rank(space2)
            return rank == space2.shape[-1]
        elif(space2.size == 0):
            rank = np.linalg.matrix_rank(space1)
            return rank == space1.shape[-1]
        else:
            for vec in space2:
                dotProd = np.dot(space1, vec)
                if(np.any(abs(dotProd) > EPSILON)): return False
            
            s1NullRank = null_space(space1).size / space1.shape[-1]
            s2NullRank = null_space(space2).size / space2.shape[-1]
            rank = s1NullRank + s2NullRank
            isFullRank = rank == space1.shape[-1]

            return isFullRank

    @staticmethod
    def cullDuplicates(subspaces):

        cleaned = []
        for item in subspaces:
            if(not isinstance(item, Subspace)):
                cleaned += [item]
            elif(item not in cleaned):
                cleaned += [item]
        
        return subspaces

    @staticmethod
    def solve(subspaces, printInfo=SANITY_CHECK):

        dim = subspaces[0].dim

        # check if planes share an intersection
        # method: solve cummulative intersection
        solutionSpace = Subspace(dim=dim, spans=np.identity(dim))
        if(printInfo): print(solutionSpace)
        for subspace in subspaces:
            if(printInfo): print("\t", subspace)
            intersection = solutionSpace.intersect(subspace)
            if(printInfo): print(intersection)
            if(intersection == None):
                return None
            solutionSpace = intersection
        if(printInfo): print()
        
        return solutionSpace

    @staticmethod
    def clone(source):

        return copy.deepcopy(source)

    @staticmethod
    def checkParallel(first, second):
        if(isinstance(first, Subspace)):
            return first.isParallel(second)
        elif(isinstance(second, Subspace)):
            return second.isParallel(first)
        else:
            return Subspace(spans=first).isParallel(second)

    @staticmethod
    def difference(first, second, returnSubspace=True):

        if(not isinstance(first, Subspace) and isinstance(first, np.ndarray)):
            first = Subspace(spans=first)
        if(not isinstance(second, Subspace) and isinstance(second, np.ndarray)):
            first = Subspace(spans=second)
        
        # not(union(not A, B))
        diffInverse = np.concatenate([first.normalSpace, second.spanSpace])
        diffSpace = Subspace(normals=diffInverse)
        
        if(returnSubspace):
            return diffSpace
        else:
            return diffSpace.spanSpace

    @staticmethod
    def simplifySpace(space, removeZeroRows=False, normalize=False, yieldPivots=False):

        if(not isinstance(space, np.ndarray)): space = np.asarray(space)

        inpType = space.dtype
        
        space = fixFloat(space)
        simplified, pivotIds = sp.Matrix(space).rref()
        simplified = np.asarray(simplified, dtype=np.float64)
        
        if(removeZeroRows):
            rowNorm = np.linalg.norm(simplified, axis=-1)
            simplified = simplified[rowNorm > EPSILON]
            
        if(normalize):
            rowNorm = np.linalg.norm(simplified, axis=-1)
            rowNorm[rowNorm < EPSILON] = 1
            rowNorm = rowNorm.reshape((-1, 1))
            simplified = simplified / rowNorm
            
        simplified = fixFloat(simplified)
        simplified[simplified == 0] = +0

        simplified = simplified.astype(inpType)

        if(yieldPivots): return simplified, pivotIds
        else:    return simplified

    @staticmethod
    def isOrthogonal(space):

        if(not isinstance(space, np.ndarray)):
            space = np.asarray(space)
        
        mul = np.matmul(space, space.T)
        np.fill_diagonal(mul, 0)
        
        isOrtho = np.all(np.abs(mul) < EPSILON)
        
        return isOrtho

    @staticmethod
    def makeOrtho(space):

        if(not isinstance(space, np.ndarray)):
            space = np.asarray(space)
        
        cleaned = None
        for vec in np.flip(space, axis=0):
            if(cleaned is None):
                vec = np.asarray(vec)
                vec = vec / np.linalg.norm(vec)
                cleaned = vec.reshape((1, -1))
            else:
                dotProd = np.matmul(cleaned, vec).reshape((-1, 1))
                proj = dotProd * cleaned # cleaend is ortho
                summed = np.sum(proj, axis=0)
                diff = vec.reshape((1, -1))  - summed
                diff = diff / np.linalg.norm(diff)
                cleaned = np.concatenate([cleaned, diff.reshape((1, -1))],
                                         axis=0)
        
        cleaned = np.flip(cleaned, axis=0)

        return cleaned

    @staticmethod
    def isSpannedBy(vec, space):

        if(not Subspace.isOrthogonal(space)):
            space = Subspace.makeOrtho(space)
        
        # sys: m*n, vec: n*1, n: vector dimension
        # sys: each row is a unit vec

        vec = vec.reshape((-1, 1))
        sys = space / np.linalg.norm(space, axis=1).reshape((-1, 1))

        if(sys.size == 0):
            return np.linalg.norm(vec) <= EPSILON # sys and vec both zero

        dotProd = np.matmul(sys, vec)
        proj = dotProd * sys
        summed = np.sum(proj, axis=0)
        diff = vec.reshape((1, -1))  - summed

        isSpanned = np.linalg.norm(diff) <= EPSILON

        return isSpanned

    @staticmethod
    def solveLinearSys(A: np.ndarray, B: np.ndarray, check=True):
        """
        # solves x for Ax=b as a solution space.
        see stackoverflow post:
        https://stackoverflow.com/questions/46377331/solving-ax-b-for-a-non-square-matrix-a-using-python

        d: the dimension of the linear space
        n: n < d is the under-ranked basis vectors (needn't to be orthogonal)
        A: shape (d, n), row-major matrix
        B: shape (d, 1), col-major form
        """

        if(np.linalg.norm(B) < EPSILON):
            sol_lstsq = np.zeros(A.shape[-1])
        else:
            A_pinv = np.linalg.pinv(A)
            sol_lstsq = np.matmul(A_pinv, B).T
            
            if(check):
                x_lstsq = np.matmul(A, sol_lstsq)
                diff = x_lstsq - B
                diff_norm = np.linalg.norm(diff)
                if(diff_norm > EPSILON): return None
            
        A_null = nullspace(A)
        
        sol = Subspace(spans=A_null, refPt=sol_lstsq)

        return sol

    @staticmethod
    def multiIntersect(subspaces):

        result = None
        for space in subspaces:
            if(result is None): 
                result = space
                continue

            intersect = result.intersect(space)

            if(intersect is None):
                result = None
                break

            result = intersect
        
        return result

    @staticmethod
    def load(path):

        if(not path.endswith(".npz")): path = path + ".npz"

        file = np.load(path)

        refPt = file["refPt"]
        spans = file["spans"]
        normals = file["normals"]

        space = Subspace(refPt=refPt, 
                         normals=normals, 
                         spans=spans)

        return space

    @staticmethod
    def solveEqsStack(stack):
        # stack of format [(A, B), ...]

        stackA = [A for A, _ in stack]
        stackB = [B for _, B in stack]
        A = np.concatenate(stackA, axis=0)
        B = np.concatenate(stackB, axis=0)

        solution = Subspace.solveLinearSys(A, B)

        return solution

if __name__ == "__main__":
    pass