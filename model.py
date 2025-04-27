import numpy as np
from itertools import combinations

from parameters import EUC_SPACE_DIM, SCREW_SPACE_DIM, EPSILON
from screwVectors import Frame, lineToScrewVec, screwTransform, screwVec, getAxis
from utils import fixFloat, str2num
from DOFsystem import DOFSystem
from subspace import Subspace
from messages import *
from joints import Rod, Blade

class CompliantJoint(object):
    def __init__(self, dofTwists, viewCenter, id=None, flexures=None):
        
        self.id = id

        # flexures
        self.flexures = []

        # spaces
        self.center = viewCenter
        self.freedomSpace = dofTwists
        self.constraintSpace = None
        self.flexureSpace = None
        self.dof = dofTwists.shape[0]
        self.doc = SCREW_SPACE_DIM - self.dof

        self.sys = None
        
        # completion check
        self.isComplete = False

        # messages
        self.msgModeOverall = ''
        self.msgMode = []
        self.msgRod = ''

        if(flexures is not None):
            self._addFlexures(flexures)
        
        self._model()
    
    def _model(self):

        self.constraintSpace = DOFSystem.freeToCon(self.freedomSpace)
        self.sys = DOFSystem(self.constraintSpace, True)
        self.sys.recenterSpaces(self.center)
        self.bases = self.sys.outputBases()
                # convert into constraint space
        if(len(self.flexures) > 1):
            screwVecs = np.concatenate([flexure.conSpace for flexure in self.flexures],
                                       axis=0)
            self.flexureSpace = Subspace(spans=screwVecs).spanSpace
        elif(len(self.flexures) == 1):
            screwVecs = self.flexures[0].conSpace
            self.flexureSpace = Subspace(spans=screwVecs).spanSpace
        else:
            self.flexureSpace = np.zeros((0,0))
    
    def _addFlexures(self, flexures):
        
        for i in range(len(flexures)):
            info = flexures[i]
            if(info["type"] == "rod"):
                flexure = Rod(info)
            elif(info["type"] == "blade"):
                flexure = Blade(info)
            else:
                raise Exception(MSG_UNIMPLEMENTED)

            self.flexures += [flexure]
    
    def genInfo(self):

        msg = [Msngr.showNeeded]

        # check overconstraints
        msgOC, ocFlexures = self._overConstraint()
        noOC = len(ocFlexures) == 0
        
        # check completion
        msgComp = self._completionCheck()

        # show completion if not over-constrained
        if(noOC):
            msg += [msgComp]
        else:
            msg += msgOC
        
        consSpacesInfo = self.sys.output()
        constraints = tuple(consSpacesInfo)
        
        msg += Msngr.describeRodSpace(constraints)
        msg = '\n'.join(msg)

        result = {}
        result["message"] = msg
        result["spaces"] = consSpacesInfo
        result["bases"] = self.bases
        result["center"] = list(self.center)
        result["OC"] = ocFlexures
        result["freedoms"] = self.freedomSpace.tolist()
        result["id"] = self.id
        result["flexures"] = [f.json() for f in self.flexures]
        
        return result

    def _overConstraint(self):

        targSpace = self.constraintSpace
        msg = []

        # check calidity of each rod
        isValid = True
        invalidFlexures = []
        for flexure in self.flexures:
            isAllowed = flexure.isAllowed(targSpace)
            if(not isAllowed):
                invalidFlexures += [flexure]
                isValid = False
        
        if(isValid):
            msg += [Msngr.noOCRod]
        else:
            msg += [Msngr.hasOCRod]
            msg += Msngr.ocIssue(invalidFlexures, targSpace, "Flexure")
        
        ocFlexures = tuple([flexure.id for flexure in invalidFlexures])
        
        return msg, ocFlexures

    def _completionCheck(self):

        tarSpace = self.constraintSpace
        curSpace = self.flexureSpace
        
        tarSys = DOFSystem(tarSpace, True)
        curSys = DOFSystem(curSpace, True)
        print(curSys.tVecs)
        tarAxisDeg, tarSpanDeg = tarSys.rotDeg, tarSys.transDeg
        curAxisDeg, curSpanDeg = curSys.rotDeg, curSys.transDeg

        msg = Msngr.completion(tarAxisDeg, tarSpanDeg, curAxisDeg, curSpanDeg)

        completion = tarAxisDeg == curAxisDeg and\
                     tarSpanDeg == curSpanDeg
        
        status = Msngr.rodSpaceComplete if completion else Msngr.rodSpaceIncomplete
        msg = '\n'.join([status, msg])

        self.isComplete = completion
        
        return msg

    @staticmethod
    def fromJSON(info):
        
        freedoms = np.asarray(info["freedoms"])
        center = np.asarray(info["center"])
        id = info["id"]
        flexures = info["flexures"]
        joint = CompliantJoint(freedoms, center, id, flexures)

        return joint

class Msngr(object):
    showNeeded = MSG_SHOW_NEEDED
    noOCRod = MSG_NO_OC_ROD
    hasOCRod = MSG_HAS_OC_ROD
    rodComplete = MSG_ROD_COMPLETE
    rodIncomplete = MSG_ROD_INCOMPLETE
    rodSpaceComplete = MSG_ROD_SPACE_COMPLETE
    rodSpaceIncomplete = MSG_ROD_SPACE_INCOMPLETE

    def __init__(self):
        pass

    @staticmethod
    def describeSubspace(motion, name=None, isFreedom=True):
        
        moType = motion[0]
        axisDeg = 0
        for axis in motion[1]:
            if(len(axis) == 3):
                axisDeg += 1
        spaceDeg = 0
        for axis in motion[3]:
            if(len(axis) == 3):
                spaceDeg += 1
        
        # space type
        mode = None
        if(moType == 'r' and isFreedom):
            spaceType = "rotation"
            mode = 'r'
        elif(moType == 't' and isFreedom):
            spaceType = "translation"
            axisDeg = spaceDeg
            mode = 't'
        elif(moType == 'r' and not isFreedom):
            spaceType = "wrench/wire"
            mode = 'w'
        
        # name override
        if(name != None):
            spaceType = name
        
        # composition
        axisNote = ''
        if(axisDeg == 1):
            direction = "that aligns with the displayed direction"
        elif(axisDeg == 2):
            direction = "that is parallel to the plane(s)"
        elif(axisDeg == 3):
            direction = "in 3D space"
            axisNote = "Note: axes must not all lie on the same plane."
        
        spaceNote = ''
        # location):
        if(spaceDeg == 0):
            position = "passes through the center point"
        elif(spaceDeg == 1):
            position = "passes through this line at some point"
        elif(spaceDeg == 2):
            if(axisDeg <= 2):
                position = "lies on this plane"
            else:
                position = "intersects with this plane at some point"
            if(axisDeg == 2 and mode != 't'):
                spaceNote = "Note: the axes must not intersect at the same point."
        elif(spaceDeg == 3):
            if(axisDeg == 1):
                position = "lies anywhere in space"
                if(mode != 't'):
                    spaceNote = "Note: axes must not all lie on a same plane."
            elif(axisDeg == 2):
                position = "lies on any parallel plane"
                if(mode != 't'):
                    spaceNote = "Note: axes must not all lie on the same plane."
            elif(axisDeg == 3):
                position = "lies anywhere in space"
                if(mode != 't'):
                    spaceNote = "Note: axes must not all lie on a same plane nor pass through the same point."
        
        if(moType == 'r'):
            msg = "Any axis %s and %s is an allowed %s axis. " % (direction, position, spaceType)
        elif(moType == 't'):
            msg = "Any axis %s is an allowed %s axis. " % (direction, spaceType)
        
        
        msg += axisNote + ' ' + spaceNote
        
        return msg

    @staticmethod
    def describeFreedomSpace(sys,motions):
        
        msgs = []
        # report number of freedoms
        msgs += ["This mode has %d rotational and %d translational DOF." % (sys.rotDeg, sys.transDeg)]
        
        # check viability
        msgs += [Msngr.viabilityCheck(sys, "Current mode")]
        
        
        # describe the direction and location of subspaces
        for i in range(len(motions)):
            motion = motions[i]
            msgs += ["Motion subspace %d:" % (i + 1)]
            msgs += [Msngr.describeSubspace(motion)]

        return '\n'.join(msgs)

    @staticmethod
    def describeRodSpace(motions):
        
        # describe direction and position
        msg = []

        for i in range(len(motions)):
            con = motions[i]
            msg += ["Constraint subsapce %d:" % (i + 1)]
            msg += [Msngr.describeSubspace(con, "rod", False)]
        
        return msg
    
    @staticmethod
    def viabilityCheck(sys, name="Design"):
        
        if(sys.isConstraint):
            isViable = sys.rotDeg > 0 # has directional component
        else:
            # is not unconstrained & has directional component
            isViable = (sys.rotDeg + sys.transDeg < 6) and (sys.transDeg < 3)
        
        viability = "viable" if isViable else "unviable"
        msg = "%s is %s." % (name, viability)

        return msg

    @staticmethod
    def ocIssue(elems, space, name):

        msgs = []
        for elem in elems:
            validDir, validPos = elem.dirPosAllowed(space)
            if(not validDir and validPos):
                violation = "direction"
            elif(validDir and validPos):
                violation = "position"
            else:
                violation = "direction and position"

            msg = "The highlighted %s has an invalid %s." % (name + " %d" % elem.id, violation)
            msgs += [msg]
        
        return msgs

    @staticmethod
    def completion(*args):

        return MSG_CHECKLIST(*args)

if __name__ == "__main__":

    m = CompMechSimp()
    info = m.computeDesign()
    messages = m.outputMsg()
    bases = m.outputBases()