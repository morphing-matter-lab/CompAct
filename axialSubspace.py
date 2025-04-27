import numpy as np
import copy
from scipy.linalg import null_space
from messages import *
from subspace import Subspace
from screwVectors import screwVec
from parameters import *

class AxialSubspace(Subspace):
    def __init__(self, axis, refPt, spanVecs, isPrinciple=True):

        self.axis = None
        self.degree = None

        self.isTrans = None
        self.isPrinciple = isPrinciple

        self._initAxialSubspace(axis, refPt, spanVecs)

    def __eq__(self, other):

        if(not isinstance(other, AxialSubspace)): return False

        subspaceSame = super().__eq__(other)
        if(not subspaceSame): return False

        # check axis
        if(self.isFull):
            return other.isFull
        else:
            axisNull = null_space(self.axis).T
            return Subspace.isComplementary(axisNull, other.axis)

    def __repr__(self):

        axis = "%d-Form" % len(self.axis)
        form = super().__repr__()

        return "AxialSubspace(%s, %s)" %(axis, form)

    def _initAxialSubspace(self, axis, refPt, spanVecs):

        if(not isinstance(axis, np.ndarray)): axis = np.asarray(axis)
        axis = axis.reshape((-1, EUC_SPACE_DIM))

        if(not isinstance(spanVecs, np.ndarray)): 
            spanVecs = np.asarray(spanVecs)
        
        self.axis = axis
        self.degree = len(self.axis)
        self.isTrans = np.all(abs(axis) <= EPSILON)
        if(self.isTrans):
            spanVecsDirected = np.asarray([[]])
        else:
            if(self.isPrinciple):
                # axis is always aligned to the basis vectors
                spanVecsDirected = axis
            else:
                spanVecsDirected = spanVecs
        
        if(self.isPrinciple):
            for vec in spanVecs:
                spanVecsDirected = Subspace.expand(spanVecsDirected, vec)
        
        super().__init__(spans=spanVecsDirected, refPt=refPt)
    
    def printInfo(self):

        print(self.__repr__())
        print("Axii: ")
        print(self.axis)

        super().printInfo(printRefPt=True, printSpan=True)

    def axisNormal(self):

        return null_space(self.axis).T

    def getSubspace(self):

        return self.copy()

    def intersect(self, other, isFreedom=True):

        assert isinstance(other, AxialSubspace),\
               "Type error: other must be a AxialSubspace"

        selfSubspace = self.getSubspace()
        otherSubspace = other.getSubspace()
        
        intersection = selfSubspace.intersect(otherSubspace)
        
        #intersection = self.intersect(other)


        if(intersection == None): return None

        if(isFreedom):
            axisCombined = self.axis
            for axis in other.axis:
                axisCombined = Subspace.expand(axisCombined, axis)
        else:
            if(self.degree > self.spanRank):
                axisCombined = other.axis
            else:
                axisCombined = intersection.spanSpace

        intAxialSubspace = AxialSubspace(axisCombined, 
                                         intersection.refPt, 
                                         intersection.spanSpace,
                                         isPrinciple=False)
        
        return intAxialSubspace

    def curScrewSpace(self, flexures):

        goalSpace = self.outputScrewSpace()

        # get intersections
        allSVs = []
        for f in flexures:
            fAS = f.axialSubspace()
            intersection = self.intersect(fAS, isFreedom=False)
            intSVs = intersection.outputScrewSpace()
            allSVs += [intSVs]
        
        if(len(allSVs) != 0):
            allSVs = np.concatenate(allSVs, axis=0)
        
        cur = np.zeros(0)
        for vec in allSVs:
            cur = Subspace.expand(cur, vec)

        return cur

    def reposition(self, point):
        
        if(not isinstance(point, np.ndarray)):
            point = np.asarray(point)
        
        if(self.isTrans): # translation is spatailly invariant
            self.refPt = point
        else: # rotation is position-dependent
            cp = self.closestPoint(point)
            
            self.refPt = cp

    def isAxisIncludedBy(self, other):

        if(other.degree == other.dim):
            # other's axis is full-ranked
            return True
        else:
            # none full rank cases

            for vec in self.axis:
                correction = Subspace.project(vec, other.axis)
                residual = vec - correction
                if(np.any(abs(residual) > EPSILON)): return False
            
            return True
    
    def isSpaceIncludedBy(self, other):
        # other has more axis and lower dimensional subspace than self
        
        # step 1: make copies to aovid aliasing
        selfDummy = copy.deepcopy(self) # avoid aliasing
        otherDummy = copy.deepcopy(other)
        
        # step 2: remove principle axis from self
        #         move refPt position
        
        if(selfDummy.isPrinciple):
            for vec in selfDummy.axis: # move everytime unwedge
                pointer = otherDummy.refPt - selfDummy.refPt
                pointerProj = Subspace.project(pointer, vec.reshape(1, -1))
                
                selfDummy.move(pointerProj)
                #selfDummy.removeSpan(pointerProj)
                selfDummy.removeSpan(vec)
        
        # step 3: remove principle axis from other
        #         do not move refPt position
        if(otherDummy.isPrinciple):
            for vec in otherDummy.axis:
                otherDummy.removeSpan(vec)

        # step 4: check if erased subspace is included by other
        isSubset = selfDummy.isSubspaceOf(otherDummy)
        
        return isSubset

    def isIncludedBy(self, other):
        
        # check if axis is included by other
        if(not self.isAxisIncludedBy(other)): 
            return False
        # at this point, self will have equal or fewer axii than other
        
        # check if subspace is included by other
        if(not self.isSpaceIncludedBy(other)): 
            return False
        
        return True

    def alignSpace(self, bases):

        # align axii
        factor = np.zeros(len(bases))

        if(not self.isPoint):
            for vec in self.spanSpace:
                dotProd = np.dot(bases, vec)
                factor += abs(dotProd)
            
            aligned = []
            for i in range(len(bases)):
                if(factor[i] > EPSILON):
                    aligned += [bases[i]]
            
            aligned = np.stack(aligned)

            self.spanSpace = aligned

    def output(self):
        
        motionType = 't' if self.isTrans else 'r'
        axis = tuple([tuple(vec) for vec in self.axis])
        spans = tuple([tuple(vec) for vec in self.spanSpace])
        try:
            refPt = tuple(self.refPt)
        except:
            refPt = (0, 0, 0)
        report = (motionType, axis, refPt, spans)

        return report
    
    def outputScrewSpace(self):

        if(self.spanRank != 0):
            anchors = self.refPt + self.spanSpace
        else:
            anchors = self.refPt.reshape((1, -1))
        
        screws = np.zeros(0)
        for axis in self.axis:
            base = screwVec(axis, self.refPt)
            screws = Subspace.expand(screws, base)
            for anchor in anchors:
                screw = screwVec(axis, anchor)
                screws = Subspace.expand(screws, screw)
                
        return screws
