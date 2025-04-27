import numpy as np

from DOFsystem import DOFSystem
from parameters import EPSILON, EUC_SPACE_DIM
from screwVectors import screwVec
from utils import nullspace, fixFloat

class Flexure(object):
    def __init__(self, id):
        self.id = id
        self.conSpace = None

    def isAllowed(self, tarSpace):
        # tar space is a constraint space
        invalidConSpace = nullspace(tarSpace)
        dotProd = invalidConSpace @ self.conSpace.T
        
        return np.all(np.abs(dotProd) < EPSILON)
    
    def dirPosAllowed(self, consVecs):

        dir, pos = True, True
        for vec in self.conSpace:
            vec = vec / np.linalg.norm(vec)
            vec = vec.reshape((1, -1))
            dotProd = np.sum(vec * consVecs, axis=-1)
            projected = np.matmul(dotProd, consVecs)
            diff = vec - projected
            dirResidual = np.linalg.norm(diff[:EUC_SPACE_DIM])
            posResidual = np.linalg.norm(diff[EUC_SPACE_DIM:])
            if(dirResidual > EPSILON):
                dir = False
            if(posResidual > EPSILON):
                pos = False
        
        return dir, pos

class Rod(Flexure):
    type = "rod"
    def __init__(self, info):
        super().__init__(info["id"])
        
        self.start = np.asarray(info["start"])
        self.end = np.asarray(info["end"])
        self.vec = self.end - self.start
        self.len = np.linalg.norm(self.vec)
        self.vecUnit = self.vec / self.len

        self._modelSpaces()
    
    def _modelSpaces(self):

        screw = screwVec(self.vecUnit, self.start)
        
        self.conSpace = screw.reshape(1, -1)

    def json(self):

        info = {}
        info["type"] = Rod.type
        info["id"] = self.id
        info["start"] = fixFloat(self.start.tolist())
        info["end"] = fixFloat(self.end.tolist())

        return info

class Blade(Flexure):
    type = "blade"
    def __init__(self, info):
        super().__init__(info["id"])
        
        self.normal = fixFloat(np.asarray(info["normal"]))
        self.center = fixFloat(np.asarray(info["center"]))
        self.pts = info["pts"]

        self._modelSpaces()

    def _modelSpaces(self):
        
        dir1, dir2 = nullspace(self.normal.reshape(1, -1))

        t1 = screwVec(trans=self.normal)
        r1 = screwVec(dir1, self.center)
        r2 = screwVec(dir2, self.center)

        freedomSpace = np.stack([r1, r2, t1])
        self.conSpace = DOFSystem.freeToCon(freedomSpace)

    def json(self):

        info = {}
        info["type"] = Blade.type
        info["id"] = self.id
        info["normal"] = self.normal.tolist()
        info["center"] = self.center.tolist()
        info["pts"] = self.pts

        return info