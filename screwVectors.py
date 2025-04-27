import numpy as np
from messages import MSG_UNIMPLEMENTED, MSG_INPUT_NOT_SCREW
from parameters import SCREW_SPACE_DIM, EUC_SPACE_DIM, EPSILON, TAU
from subspace import Subspace
from utils import fixFloat, p2t, nullspace
import sympy as sp
import math as m

class Frame(object):
    def __init__(self, origin, sys):
        self.origin = origin
        self.system = sys / np.linalg.norm(sys, axis=1).reshape((-1, 1))
        self.x = sys[0]
        self.y = sys[1]
        self.z = sys[2]
        self.dim = len(self.origin)

    def __repr__(self):

        msg = "O: %s X: %s Y: %s Z: %s" % (str(self.origin), str(self.x), str(self.y), str(self.z))

        return msg

    def transform(self, screw):

        # transform vectors
        origin = screwTransform(self.origin, screw)
        sys = np.stack([screwTransform(vec, screw, isPt=False) for vec in self.system])
        newFrame = Frame(origin, sys)

        return newFrame

    def output(self):
        
        o = p2t(self.origin)
        x = p2t(self.x)
        y = p2t(self.y)
        z = p2t(self.z)

        return tuple([o, x, y, z])

    def AdToWorld(self):

        Ad = adFromFrames(self, Frame.spatial())

        return Ad

    def reSpan(self, source):
        # reorientt the source span into vectors aligned with the currrent
        # system if possible

        if(not isinstance(source, np.ndarray)):
            source = np.asarray(source)

        if(source.size == 0):
            return source
        
        inputRank = np.linalg.matrix_rank(source)
        
        pivots = []
        for vec in self.system:

            if(isSpannedBy(vec, source)):
                pivots += [vec]

        pivots = np.asarray(pivots)

        if(pivots.shape[0] == inputRank):
            # full rank reorientation, return concat
            return pivots
        
        for vec in source:
            dotProd = np.matmul(pivots, vec)
            proj = dotProd * pivots
            summed = np.sum(proj, axis=0)
            residual = vec - summed
            if(np.linalg.norm(residual) > EPSILON):
                newPivot = residual / np.linalg.norm(residual)
                pivots = np.concatenate((pivots, newPivot.reshape((1, -1))), axis = 0)
            
            if(pivots.shape[0] == inputRank):
                break

        return pivots

    def clone(self):

        ori = np.copy(self.origin)
        sys = np.copy(self.system)

        return Frame(ori, sys)

    @staticmethod
    def spatial():
        return Frame(np.zeros(EUC_SPACE_DIM), np.identity(EUC_SPACE_DIM))

    @staticmethod
    def fromList(inp):
        o = np.asarray(inp[0])
        sys = np.asarray(inp[1:])

        return Frame(o, sys)

    @staticmethod
    def xAligned(vec):

        if(not isinstance(vec, np.ndarray)):
            vec = np.asarray(vec)
        
        vec = vec.reshape((1, -1))
        complement = nullspace(vec)
        system = np.concatenate([vec, complement], axis=0)

        return system

class FrameCyl(object):
    def __init__(self, origin, x_vec):
        self.origin = origin
        self.x = x_vec

    def __repr__(self):

        msg = "O: %s X: %s" % (str(self.origin), str(self.x))

        return msg

def swapOperator():

    swap = np.eye(SCREW_SPACE_DIM)
    swap = np.concatenate((swap[EUC_SPACE_DIM:, :], 
                           swap[:EUC_SPACE_DIM, :]), axis=0)
    
    return swap

def screwVec(axis=[0, 0, 0], ref=[0, 0, 0], trans=[0, 0, 0]):

    if(not isinstance(axis, np.ndarray)):
        axis = np.array(axis)
    if(not isinstance(ref, np.ndarray)):
        ref = np.array(ref)
    if(not isinstance(trans, np.ndarray)):
        trans = np.array(trans)

    s = np.cross(ref, axis) + trans
    
    result = np.concatenate((axis, s))

    return result

def lineToScrewVec(motion):

    # chr,      2 points
    motionType, p0, p1 = motion
    p0, p1 = np.asarray(p0), np.asarray(p1)
    vec = p1 - p0

    if(motionType == 'r' or motionType == 'w'): # rotation & wrench
        sv = screwVec(axis=vec, ref=p0)
    elif(motionType == 't'): # translation
        sv = screwVec(trans=vec)
    else:
        assert False, "function not implemented"

    sv = fixFloat(sv)
    return sv

def skewSymmetricMatrix(vec):

    x, y, z = vec
    m = np.asarray([[ 0, -z,  y],
                    [ z,  0, -x],
                    [-y,  x,  0]])

    return m

def getAxis(vec, unitize=False):

    vecNorm = np.linalg.norm(vec[:EUC_SPACE_DIM])

    if(vecNorm > EPSILON):
        if(unitize):
            return vec[:EUC_SPACE_DIM] / vecNorm
        else:
            return vec[:EUC_SPACE_DIM]
    else:
        if(unitize):
            return vec[EUC_SPACE_DIM:] / np.linalg.norm(vec[EUC_SPACE_DIM:])
        else:
            return vec[EUC_SPACE_DIM:]

def decompose(vec, debugPrint=False):

    if(not isinstance(vec, np.ndarray)):
        vec = np.asarray(vec, dtype=np.float64)

    if(vec.ndim >= 2): return _decomposeND(vec) # ND mode
    else: return _decompose1D(vec, debugPrint)

def _decompose1D(vec, debugPrint=False):

    # scheck for 0 screw vector
    vecNorm = np.linalg.norm(vec)
    if(vecNorm > EPSILON): vecScaled = vec / vecNorm
    else: vecScaled = np.zeros(SCREW_SPACE_DIM)

    # mag analysis
    rotMag = np.linalg.norm(vec[:EUC_SPACE_DIM])
    transMag = np.linalg.norm(vec[EUC_SPACE_DIM:])

    if(rotMag > EPSILON): # rotation
        if(debugPrint): print("rot")
        vecUnitized = vec / rotMag
        axisUnit = vecUnitized[:EUC_SPACE_DIM]
        transLen = np.dot(axisUnit, vec[EUC_SPACE_DIM:])
        refPt = np.cross(axisUnit, vecUnitized[EUC_SPACE_DIM:])
    elif(transMag > EPSILON): # pure translation
        if(debugPrint): print("trans")
        vecUnitized = vec / transMag
        axisUnit, rotMag = vecUnitized[EUC_SPACE_DIM:], 0
        transLen = transMag
        refPt = np.zeros(EUC_SPACE_DIM)
    else: # all 0
        if(debugPrint): print("zero")
        axisUnit = np.zeros(EUC_SPACE_DIM)
        rotMag = 0
        transLen = 0
        refPt = np.zeros(EUC_SPACE_DIM)
        
    # mag = fixFloat(mag)
    # transLen = fixFloat(transLen)

    # # remove components that are insignificant
    # axisMax = np.max(np.abs(axisUnit))
    # if(axisMax > EPSILON): axisUnit[np.abs(axisUnit / axisMax) < TAU] = 0

    # refMax = np.max(np.abs(refPt))
    # if(refMax > EPSILON): refPt[np.abs(refPt / refMax) < TAU] = 0

    return axisUnit, rotMag, transLen, refPt

def _decomposeND(vec):

    # find pivotal cmoponent
    dir, pos = vec[...,:EUC_SPACE_DIM], vec[..., EUC_SPACE_DIM:]
    dirNorm = np.expand_dims(np.linalg.norm(dir, axis=-1), axis=-1)
    dirNorm[dirNorm < EPSILON] = 0
    posNorm = np.expand_dims(np.linalg.norm(pos, axis=-1), axis=-1)
    posNorm[posNorm < EPSILON] = 0

    # remix pivot axis and magnitude
    useRotMask = dirNorm >= EPSILON
    axis = dir * useRotMask + pos * np.logical_not(useRotMask)
    axisNorm = dirNorm * useRotMask + posNorm * np.logical_not(useRotMask)
    axisNorm[axisNorm < EPSILON] = 1 # avoid div 0
    
    # get output components
    axisUnit = axis / axisNorm
    
    axisNormer = np.copy(axisNorm)
    axisNormer[axisNormer == 0] = 1 # avoid div-0 

    posNormed = pos / axisNormer
    transNorm = np.expand_dims(np.diagonal(axisUnit @ pos.T), axis=-1)
    refPt = np.cross(axisUnit, posNormed)
    

    return axisUnit, dirNorm, transNorm, refPt

def screwTransform(vec, screwVec, isPt=True):
    
    if(not isinstance(vec, np.ndarray)):
        vec = np.asarray(vec)

    if(vec.shape[-1] == EUC_SPACE_DIM): # point vector mode
        axis, rotMag, transLen, refPt = decompose(screwVec)
        
        R = screwRot(axis, rotMag)
        if(isPt): # point
            transformed = refPt + np.matmul(R, vec - refPt) + transLen * axis
        else: # vec
            transformed = np.matmul(R, vec)

    else: # screw vector mode
        vecAxis, vecMagnitude, vecTransLen, vecRefPt = decompose(vec)
        axisNew = screwTransform(vecAxis, screwVec, isPt=False) # unit
        refPtNew = screwTransform(vecRefPt, screwVec, isPt=True)
        transformed = screwVec(axisNew * vecMagnitude, 
                               refPtNew, 
                               vecTransLen * axisNew)
    
    return transformed

def screwTransMat(d, R):
    D = skewSymmetricMatrix(d)
    m = np.zeros((SCREW_SPACE_DIM, SCREW_SPACE_DIM)).astype(d.dtype)
    m[:3, :3] = R
    m[3:, :3] = np.matmul(D, R)
    m[3:, 3:] = R

    return m

def findTrans(source, target):

    # Ax = B
    # A = Bx^-1
    # A: to find, B: target X: source

    #trans = np.matmul(target, source.T)
    trans = np.matmul(target, np.linalg.inv(source))

    return trans

def screwRot(inputVec, rotMag=None):
    
    if(len(inputVec) == SCREW_SPACE_DIM): # screw mode, decompose
        axis, rotMag, _, _ = decompose(inputVec)
    elif(len(inputVec) == EUC_SPACE_DIM and rotMag != None): # axis and rotMag
        axis = inputVec
    K = skewSymmetricMatrix(axis)
    # watch out for types casting here
    R = np.identity(3) +\
        np.sin(rotMag) * K +\
        (1 - np.cos(rotMag)) * np.matmul(K, K)
        
    return R

def inverseAd(Ad):
    R = Ad[:EUC_SPACE_DIM, :EUC_SPACE_DIM]
    DR = Ad[EUC_SPACE_DIM:, :EUC_SPACE_DIM]
    D = np.matmul(DR, R.T) # r is rotation matrix -> transpose = inverse

    adInv = np.zeros((SCREW_SPACE_DIM, SCREW_SPACE_DIM))
    adInv[:EUC_SPACE_DIM, :EUC_SPACE_DIM] = R.T
    adInv[EUC_SPACE_DIM:, :EUC_SPACE_DIM] = -1 * np.matmul(R.T, D)
    adInv[EUC_SPACE_DIM:, EUC_SPACE_DIM:] = R.T

    return adInv

def trMatrix(screw):

    R = screwRot(screw)
    Tr = np.zeros((SCREW_SPACE_DIM, SCREW_SPACE_DIM))
    Tr[EUC_SPACE_DIM:, EUC_SPACE_DIM:] = R
    
    return Tr

def screwVecLen(vec):

    axis = vec[:EUC_SPACE_DIM]
    axisLen = np.linalg.norm(axis)
    if(axisLen > EPSILON): # rotation axis exists
        vecLen  = axisLen
    else:
        vecLen = np.linalg.norm(vec[EUC_SPACE_DIM:])
    
    return vecLen

def normScrew(vec, fix=True):

    vecLen = screwVecLen(vec)
    
    if(vecLen <= EPSILON): return np.zeros(SCREW_SPACE_DIM) # shortcut
    
    normed = vec / vecLen
    if(fix):
        normed = fixFloat(normed)

    return normed

def norm(vecs, fix=True):

    result = []
    for vec in vecs:
        vec = normScrew(vec, fix=fix)
        result += [vec]
    
    if(isinstance(vecs, np.ndarray)):
        return np.stack(result, axis=0)
    else:
        return result

def rotMatrixAxial(axis, angle, unit='r'):
    
    if(unit=='d'):
        angle = m.radians(angle)

    if(not isinstance(axis, np.ndarray)):
        axis = np.asarray(axis)

    axisSkew = skewSymmetricMatrix(axis)
    outer = np.outer(axis, axis)
    id = np.identity(EUC_SPACE_DIM)

    cosAng = sp.cos(angle)
    sinAng = sp.sin(angle)

    R = cosAng * id + sinAng * axisSkew + (1 - cosAng) * outer
    
    return R

def adFromFrames(source, target, flipD=False):

    if(flipD):
        d = target.origin - source.origin
    else:
        d = source.origin - target.origin
    
    D = sp.Matrix(skewSymmetricMatrix(d))
    R = sp.Matrix(findTrans(source.system, target.system))
    
    ad = sp.Matrix(np.zeros((6,6)))
    ad[:3, :3] = R
    ad[3:, :3] = D * R
    ad[3:, 3:] = R
    
    ad_inv = sp.Matrix(np.zeros((6,6)))
    ad_inv[:3, :3] = R.T
    ad_inv[3:, :3] = -1 * R.T * D
    ad_inv[3:, 3:] = R.T
    
    return ad, ad_inv

def isTrans(motion):

    motion = motion.astype(np.float64)

    if(motion.ndim == 1):
        rotAxis = motion[:EUC_SPACE_DIM]
        return np.linalg.norm(rotAxis) <= EPSILON
    
    rotAxes = motion[:, :EUC_SPACE_DIM]
    return np.linalg.norm(rotAxes, axis=-1) <= EPSILON

def isRot(motion):

    motion = motion.astype(np.float64)
    
    if(motion.ndim == 1):    
        rotAxis = motion[:EUC_SPACE_DIM]
        return np.linalg.norm(rotAxis) > EPSILON
    
    rotAxes = motion[:, :EUC_SPACE_DIM]
    return np.linalg.norm(rotAxes, axis=-1) > EPSILON

def smallestRot(angles):

    nums = []
    for n in angles:
        if n < 0:
            nums += [n + (np.pi * 2)]
        else:
            nums += [n]

    return sorted(nums)[0]

def isSpannedBy(vec, sys):
    # sys: m*n, vec: n*1, n: vector dimension
    # sys: each row is a unit vec

    vec = vec.reshape((-1, 1))
    sys = sys / np.linalg.norm(sys, axis=1).reshape((-1, 1))

    if(sys.size == 0):
        return np.linalg.norm(vec) <= EPSILON # sys and vec both zero

    dotProd = np.matmul(sys, vec)
    proj = dotProd * sys
    summed = np.sum(proj, axis=0)
    diff = vec.reshape((1, -1))  - summed

    isSpanned = np.linalg.norm(diff) <= EPSILON

    return isSpanned

def freedomToHot(space):
    
    hot = np.sum(space, axis=0)
    
    for i in range(len(hot)):
        entry = hot[i]
        if(entry == 0 or entry == 1):
            continue
        raise Exception(MSG_UNIMPLEMENTED)
    
    return hot

def fixSV(vec):
    
    vecNorm = np.linalg.norm(vec)
    if(vecNorm < EPSILON): return np.zeros(SCREW_SPACE_DIM)
    
    vec /= vecNorm # normalized
    vec = fixFloat(vec)

    rotPart = vec[:EUC_SPACE_DIM]
    posPart = vec[EUC_SPACE_DIM:]
    rotNorm = np.linalg.norm(rotPart)
    isRot = rotNorm > EPSILON

    if(isRot):
        
        axis = rotPart
        axisUnit = axis / rotNorm
        # remove components that are insignificant
        axisMax = np.max(axisUnit)
        if(axisMax > EPSILON): axisUnit[axisUnit / axisMax < TAU] = 0
        axisUnit = axisUnit / np.linalg.norm(axisUnit)
        axisFixed = axisUnit * fixFloat(rotNorm)

        pitch = np.dot(axisUnit, posPart)
        
        crossProd = posPart - axisUnit * pitch
        refPt = np.cross(axisUnit, crossProd / rotNorm)
        refPtFixed = fixFloat(refPt)
        if(pitch < EPSILON): # pure rotation
            vec = screwVec(axis=axisFixed, ref=refPtFixed)
        else:
            trans = axisUnit * fixFloat(pitch)
            vec = screwVec(axis=axisFixed, ref=refPtFixed, trans=trans)
    else:
        axis = posPart
        posNorm = np.linalg.norm(axis)
        axisUnit = posPart / posNorm

        # remove components that are insignificant
        axisMax = np.max(axisUnit)
        if(axisMax > EPSILON): axisUnit[axisUnit / axisMax < TAU] = 0

        transVec = axisUnit * fixFloat(posNorm)
        vec = screwVec(trans=transVec)

    vec *= vecNorm

    return vec

def scaleTo(vec, vLen):

    dirPart = vec[:EUC_SPACE_DIM]
    dirNorm = np.linalg.norm(dirPart)

    if(dirNorm > EPSILON): # rotation vector
        factor = vLen / dirNorm
    else: # translation vector
        posPart = vec[EUC_SPACE_DIM:]
        posNorm = np.linalg.norm(posPart)
        factor = vLen / posNorm
    
    scaled = vec * factor

    return scaled

def zero(shape=SCREW_SPACE_DIM):

    return np.zeros(shape)

def travelDist(pt, twist):
    
    axis, rotMag, transMag, refPt = decompose(twist)

    if(isinstance(rotMag, np.ndarray)): rotMag = rotMag.reshape(-1)
    if(isinstance(transMag, np.ndarray)): transMag = transMag.reshape(-1)

    delta = refPt - pt
    orthoDelta = np.cross(delta, axis)
    radius = np.linalg.norm(orthoDelta, axis=-1)
    arcLen = radius * rotMag
    dist = np.sqrt((np.square(arcLen) + np.square(transMag)))
    
    return dist

def unitize(vecs):
    # last dim is screw

    if(not isinstance(vecs, np.ndarray)): vecs = np.asarray(vecs)

    if(not vecs.shape[-1] == SCREW_SPACE_DIM): raise Exception(MSG_INPUT_NOT_SCREW)

    unitized = _unitize1D(vecs) if vecs.ndim == 1 else _unitizeND(vecs)

    return unitized

def _unitizeND(vecs):

    axis, rotMag, transMag, refPt = decompose(vecs)
 
    isTrans = rotMag < EPSILON
    axisFactored = np.copy(axis)
    axisFactored[isTrans] *= 0 # so that the directional component is zero

    rotMagFactored = np.copy(rotMag)
    rotMagFactored[isTrans] = transMag[isTrans] # avoid div-0

    unitized = screwVec(axisFactored, refPt, axis * (transMag / rotMagFactored))

    return unitized

def _unitize1D(vec):

    axis, rotMag, transMag, refPt = decompose(vec)
 
    isTrans = rotMag < EPSILON
    if(isTrans): return screwVec(trans=axis)

    unitized = screwVec(axis, refPt) * rotMag + screwVec(trans=transMag * axis)
    return unitized

def tangent(vecs, pt, unitize=False):

    axes, rotMag, transMag, refPt = decompose(vecs)

    pointer = pt - refPt
    rotTangent = np.cross(pointer, axes)
    tangent = rotTangent * rotMag + axes * transMag

    if(unitize):
        norm = np.linalg.norm(tangent)
        norm[norm < EPSILON] = 1
        tangent /= norm
    
    tangent[np.abs(tangent) < EPSILON] = 0

    return tangent

def fullSpace(pivot):

    if(not isinstance(pivot, np.ndarray)): pivot = np.asarray(pivot)

    jft = np.identity(SCREW_SPACE_DIM)

    rotPos = np.cross(pivot, np.identity(EUC_SPACE_DIM))
    jft[:EUC_SPACE_DIM, EUC_SPACE_DIM:] = rotPos

    return jft

def cleanDOFs(vecs, shift):

    # ignore positional component or things may get nasty
    cleanedVecs = []
    for vec in vecs:
        axis, rotMag, transMag, _ = decompose(vec)
        cleanedVec = screwVec(axis, [0, 0, 0]) * rotMag + screwVec(trans=axis) * transMag
        cleanedVecs += [cleanedVec]
    cleanedVecs = np.stack(cleanedVecs, axis=0)

    cleaned = Subspace.simplifySpace(nullspace(cleanedVecs), normalize=True)
    
    result = []
    for vec in cleaned:
        axis, rotMag, transMag, refPt = decompose(vec)
        newVec = screwVec(axis, refPt + shift) * rotMag + screwVec(trans=axis) * transMag
        result += [newVec]
    
    result = np.stack(result, axis=0)

    return result

if __name__ == "__main__":
    
    v1 = np.asarray([0.01257, 0, 0, -0.5, 0.37704, -0.62839])
    v2 = np.asarray([0, 0, 0, 0.1733, 0.1733, 0])
    v3 = np.asarray([0.5, 0, 0, 0, 25, 15])
    v = np.stack([v3, v2, v1], axis=0)
    
    for item in decompose(v): print(item)