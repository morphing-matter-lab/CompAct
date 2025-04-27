from graph import CMGraph
from bruteForce import NumAlgo
from parameters import MAX_HISTORY, BF_ITER_MAX
from model import CompliantJoint

class DesignManager(object):
    def __init__(self):

        self.history = []
        self.pointer = -1
        self.maxHistory = MAX_HISTORY
        self.solver = NumAlgo()

        self.isEmpty = len(self.history) == 0
    
    def reset(self):

        self.__init__()

    def getDesign(self):

        if(self.pointer is None): return None

        return self.history[self.pointer]

    def updateDesign(self, design):
        
        self._trunc()

        self.history += [design]
        self.pointer = -1

        self._trim()

        self.isEmpty = len(self.history) == 0

        return self.history[self.pointer]
    
    def replace(self, design):

        self.history[self.pointer] = design

    def expand(self, designs):
        
        self._trunc()

        self.history += designs
        self.pointer = -1

        self._trim()
        
        self.isEmpty = len(self.history) == 0

        return self.history[self.pointer]
    
    def undo(self):

        self.pointer -= 1
        self.pointer = max(self.pointer, -len(self.history))

        return self.getDesign()
    
    def redo(self):

        self.pointer += 1
        self.pointer = min(self.pointer, -1) # capping

        return self.getDesign()
    
    def solve(self, design, json):

        config = json["config"] if "config" in json else {}
        
        clone = config["clone"] if "clone" in config else True
        useCostVec = config["useCostVec"] if "useCostVec" in config else True
        returnLog = True
        maxIter = config["maxIter"] if "maxIter" in config else BF_ITER_MAX
        
        log, status = self.solver.solve(design, 
                                        clone=clone, 
                                        useCostVec=useCostVec, 
                                        returnLog=returnLog, 
                                        maxIter=maxIter)
        
        design = self.expand(log)

        sol = self.history[-1]
        json = sol.outputJSON()
        json["status"] = status

        return json
    
    def analyze(self, data):

        design = self.getDesign()
        sol, status = self.solver.analyze(design)
        if(isinstance(design, CMGraph)):
            self.replace(sol)
            
        sol = self.updateDesign(sol)

        json = sol.outputJSON()
        json["status"] = status

        return json

    def fix(self, fix):

        sol = self.getDesign()
        sol, status = self.solver.fix(sol, fix)
        sol = self.updateDesign(sol)

        json = sol.outputJSON()
        json["status"] = status

        return json

    def modelingUpdate(self, data):
        
        joint = CompliantJoint.fromJSON(data)
        update = joint.genInfo()

        return update

    def genModelingInfo(self):

        solution = self.getDesign()
        return solution.genModelingInstructions()

    def _trunc(self):

        self.history = self.history[:len(self.history) + self.pointer + 1] # truncate history
    
    def _trim(self):

        if(len(self.history) > self.maxHistory): self.history[len(self.history) - self.maxHistory:]