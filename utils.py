import numpy as np
import sympy as sp
from scipy.linalg import null_space
from parameters import EPSILON, NUM_PREC
from messages import MSG_UNIMPLEMENTED
from itertools import combinations
import math as m
import decimal
import json

decimal.getcontext().prec = NUM_PREC
spatialFrame = np.concatenate([np.zeros((1, 3)), np.identity(3)])

def writeCSV(file, path):
    txt = ""
    for line in file:
        for item in line:
            txt += str(item)
            txt += ','
        txt += '\n'
    
    writeFile(path, txt)

def readFile(path):
    
    with open(path, "rt") as f:
        file = f.read()
    
    return file

def writeFile(path, content):
    
    with open(path, "wt") as f:
        f.write(content)

def _fixFloat(inp):
    if(isinstance(inp, np.ndarray)):

        inp[np.isclose(inp, np.zeros(inp.shape), atol=EPSILON)] = 0
        inp[np.isclose(inp, np.ones(inp.shape), atol=EPSILON)] = 1
    else:
        if(abs(inp) <= EPSILON): inp = 0
        elif(abs(1 - abs(inp)) <= EPSILON): inp = -1 if inp < 0 else 1

    diff = inp - _fixFloat(inp) 
    if(isinstance(diff, np.ndarray)):
        if(np.any(diff > 0)):
            print(diff)
    elif(diff > 0):
        print(diff)
    return inp

def fixFloat(inp):
    # recursively fix an array
    if(isinstance(inp, np.ndarray) or\
       isinstance(inp, list) or isinstance(inp, tuple)):
        converted = []
        for item in inp:
            converted += [fixFloat(item)]
        
        # 
        if(isinstance(inp, np.ndarray)):
            return np.asarray(converted)
        elif(isinstance(inp, tuple)):
            return tuple(converted)
        else:
            return converted
    else:
        fixed = round(inp, NUM_PREC + 1)
            
        return fixed

def p2t(point):
    # convert a point coordinate into a tuple
        
    return tuple([num for num in point])

def solve(eqs, symbols):

    # try to solve symbolically
    solution = sp.solve(eqs)
    solved = not isinstance(solution, list)
    
    if(solved):
        symbVals = [solution[symb] for symb in symbols]
        symbVals = np.asarray(symbVals).astype(np.float64)
        return symbVals
    
    # try to solve numerically
    initVal = np.random.rand(len(symbols))
    for i in [9, 8, 7, 6, 5, 4]:
        try:
            solution = sp.nsolve(eqs, symbols, initVal, prec=i)
            solution = np.asarray(solution).reshape(-1).astype(np.float64)
        except:
            continue
    
    assert len(solution) == len(symbols), "multiple solutions found"
    symbVals = solution
    
    return symbVals

def str2num(inObj):

    if(isinstance(inObj, tuple) or isinstance(inObj, list)):
        container = []
        for obj in inObj:
            container += [str2num(obj)]
        
        if(isinstance(inObj, tuple)): # convert to same type as input
            container = tuple(container)
        return container
    else:
        if(isinstance(inObj, str)):
            try:
                return float(inObj)
            except:
                return inObj
        else:
            return inObj

def exhaustiveSubclasses(targClass):
    result = [targClass]

    subclasses = targClass.__subclasses__()
    if(len(subclasses) != 0):
        for subclass in subclasses:
            moreSubclassses = exhaustiveSubclasses(subclass)
            result += moreSubclassses

    return result

def VennSubspaces(dict, rev=False):
    
    combs = []
    for i in range(len(dict)):
        levelCombs = combinations(dict, i + 1)
        combs += [comb for comb in levelCombs]

    if(rev): combs = reversed(combs)
    
    return combs

def genSubspaceReport(names, validity, combs, space, delim=','):
    
    rows = []
    # header
    header = ["Degree", "Is valid"] + names + ["space"]
    rows += [delim.join(header)]

    #content
    for i in range(len(combs)):
        comb = combs[i]
        row = []
        # degree
        row += [str(len(comb))]
        # validity
        row += [str(validity[comb])]
        # flag modes
        for name in names:
            row += [str(name in comb)]
        # space
        row += [str(space[comb].tolist()).replace(", ", "__")]

        rows += [delim.join(row)]
    
    rows = '\n'.join(rows)

    return rows

def paretoFrontier(costs, return_mask=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    isEfficient = np.arange(costs.shape[0],dtype=np.int32)
    nPoints = costs.shape[0]
    nextPointIndex = 0  # Next index in the is_efficient array to search for
    while nextPointIndex < len(costs):
        nondominatedPointMask = np.any(costs < costs[nextPointIndex], axis=1)
        nondominatedPointMask[nextPointIndex] = True
        isEfficient = isEfficient[nondominatedPointMask]  # Remove dominated points
        costs = costs[nondominatedPointMask]
        nextPointIndex = np.sum(nondominatedPointMask[:nextPointIndex])+1
    if return_mask:
        isEfficientMask = np.zeros(nPoints, dtype = bool)
        isEfficientMask[isEfficient] = True
        return isEfficientMask
    else:
        return isEfficient

def loadJSON(input):

    # type conversion
    if(isinstance(input, str)):
        if(".json" in input): # input is a file path
            input = readFile(input)
        result = json.loads(input)
    elif(isinstance(input, dict)):
        result = input
    
    return result

def toJSON(input):

    return json.dumps(input)

def jsonTypeCheck(data, depth=0):

    if(isinstance(data, list) or isinstance(data, tuple)):
        stacks = [('\t' * depth) + str(type(data))]
        for item in data:
            t = jsonTypeCheck(item, depth+1)
            stacks += [t]
        return '\n'.join(stacks)
    
    if(isinstance(data, dict)):
        stacks = [('\t' * depth) + str(type(data))]
        for key in data:
            t = jsonTypeCheck(data[key], depth+1)
            stacks += [('\t' * depth) + key]
            stacks += [t]
        return '\n'.join(stacks)
    else: # base case, non-iterable object
        return ('\t' * depth) + str(type(data))

def writeJSON(input, path):

    with open(path, "w") as outfile:
        json.dump(input, outfile)

def nullspace(input):
    
    if(not isinstance(input, np.ndarray)):
        input = np.asarray(input, dtype=float)

    ns = null_space(input).T
    
    return ns

def nullspaceInt(input, simplify=False):
    
    if(not isinstance(input, sp.Matrix)):
        input = sp.Matrix(input)

    ns = input.nullspace(simplify=simplify)
    ns = sp.Matrix.hstack(*ns).T
    ns = np.asarray(ns).astype(float)

    return ns

def printBlock(input, size):

    for i in range(int(len(input) / size)):
        print(input[i * size:(i + 1) * size])

def isNdNumerical(input, ndim):

    arr = np.asarray(input)

    if(arr.ndim != ndim): return False
    
    isNum = np.issubdtype(arr.dtype, np.number)

    return isNum

def L0Filter(x, thres=EPSILON):

    arr = np.copy(x)

    arr[np.abs(arr) <= thres] = 0

    return arr

def L0Norm(x):
    return int(np.sum(np.abs(x) > EPSILON))

def invertMap(m, outType="list"):

    r = {}
    for key in m:
        if(hasattr(m[key], '__iter__')):
            for item in m[key]:
                r[item] = r.get(item, []) + [key]
        else:
            item = m[key]
            r[item] = r.get(item, []) + [key]
        
    if(outType == "list"): 
        return r
    elif(outType == "set"): 
        for key in r: r[key] = set(r[key])
    else: 
        raise Exception(MSG_UNIMPLEMENTED)

    return r

def copyToClipboard(content, name=""):

    if(not isinstance(content, str)):
        content = str(content)

    try:
        import pyperclip
        pyperclip.copy(content)
        if(len(name) > 0): print("%s copied to clipboard" % name)
        else: print("Copied to clipboard")
    except:
        print("attempted to copy %s, but pyperclip was not installed" % name)