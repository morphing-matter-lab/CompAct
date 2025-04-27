import cvxpy as cp
import numpy as np
from subspace import Subspace
from utils import L0Filter, L0Norm
from parameters import ENABLE_TIMMING, EPSILON
from messages import MSG_INVALID_INPUT

import time

def timmingFunc(func):

    def wrapped(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.2f} seconds to run.")

        return result
    
    return wrapped if ENABLE_TIMMING else func

def solveSpec(graph):

    A, B = graph.modelLinearSys(addStageLoop=True, printLoop=False)
    solSys = Subspace.solveLinearSys(A, B)
    sol = minSubspaceNormCVXPY(solSys.spanSpace, solSys.refPt)
    sol = L0Filter(sol)
    solSys.refPt = sol

    return solSys

def consLoss(A, B, x):
    return np.linalg.norm(A.dot(x) - B)

def minSubspaceNormCVXPY(A, B, minTar=1, cost=None):

    n = A.shape[0]
    # Define the variable to be optimized
    x = cp.Variable(n)

    # Define the objective function
    if(A.size == 0): func = B
    else: func = B + A.T @ x
    
    if(cost is not None):
        func = cp.multiply(func, cost)

    if(minTar == 1):
        objective = cp.Minimize(cp.norm1(func))
    elif(minTar == 2):
        objective = cp.Minimize(cp.norm2(func))

    # Define the optimization problem
    problem = cp.Problem(objective)

    # Solve the problem
    problem.solve(solver=cp.ECOS)

    # Retrieve the optimal solution
    optimal_x = x.value
    
    # Print the optimal solution
    if(A.size == 0): vec = B
    else: vec = B + A.T @ optimal_x

    return vec

@timmingFunc
def solveSkLearn(A, B, guess=None, maxIter=1e4, tol=1e-10, printInfo=False, eval=False, method="OMP"):

    from sklearn.linear_model import Lasso, LassoLars, Lars, OrthogonalMatchingPursuit

    if(method == "OMP"):
        lasso = OrthogonalMatchingPursuit(fit_intercept=False, n_nonzero_coefs=1, tol=tol)
    elif(method == "lasso"):
        alpha = (1 / (2 * A.shape[-1])) * 10 ** -1
        maxIter = int(maxIter)
        lasso = Lasso(alpha=alpha, fit_intercept=False,max_iter=maxIter, tol=tol)
    else:
        raise Exception(MSG_INVALID_INPUT)

    lasso.fit(A, B)
    ssol = lasso.coef_

    if(not eval): return ssol

    lossPre = consLoss(A, B, ssol)
    ssol = L0Filter(ssol)
    lossPost = consLoss(A, B, ssol)
    sparsity = L0Norm(ssol)

    if(printInfo):
        print("Constraint loss: %f, filtered loss: %f, sparsity: %d" %\
            (lossPre, lossPost, sparsity))

    return ssol

@timmingFunc
def solveSciPy(A, B, guess=None, maxIter=2e3, tol=1e-10, printInfo=False, eval=False):
    
    from scipy.optimize import minimize

    # Define the objective function for L0 regularization
    def objL1(x):
        return np.sum(np.abs(x))

    def objL0(x):
        return np.sum(np.abs(x) > EPSILON)

    # Define the constraint function for Ax = B
    def consFunc(x):
        return np.dot(A, x) - B
    
    # Initial guess for the solution vector x
    if(guess is None):
        guess = np.zeros(A.shape[1])
    elif(isinstance(guess, np.ndarray)):
        guess = guess
    else:
        raise Exception(MSG_INVALID_INPUT)
    
    # Solve the optimization problem
    result = minimize(objL1, guess, constraints={'type': 'eq', 'fun': consFunc},
                      tol=tol, options={'disp': False, "maxiter":maxIter})
    ssol = result.x

    if(not eval): return ssol
    
    lossPre = consLoss(A, B, ssol)
    ssol = L0Filter(ssol)
    sparsity = L0Norm(ssol)
    lossPost = consLoss(A, B, ssol)

    if(printInfo):
        print("Constraint loss: %f, filtered loss: %f, sparsity: %d" %\
            (lossPre, lossPost, sparsity))
    
    return ssol

@timmingFunc
def solveCVXPY(A, B, cost=None, normTarg=1, cons=None):
    
    useMulti = isinstance(A, list)
    sDim = len(A) if useMulti else 1
    n = A[0].shape[-1] if useMulti else A[0].shape[-1]
    x = cp.Variable((n, sDim))
    if(not useMulti): A, B = [A], [B]

    if(cost is None): cost = np.ones(n)

    # Define the objective function (minimize L1 norm)
    maxVel = cp.max(cp.multiply(cp.abs(x), cost), axis=1)
    norm = cp.norm(maxVel, normTarg)
    objective = cp.Minimize(norm)
    #objective = cp.Minimize(cp.norm(cp.multiply(x, cost), normTarg))

    # Define the constraints (Ax = B)
    constraints = [A[i] @ x[:,i] == B[i] for i in range(sDim)]
    if(cons is not None):
        for con in cons:
            constraints += [con.modelCons(A, B, x)]

    # Formulate the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # Get the sparse solution
    ssol = x.value

    return ssol

def solveCVXPY_dual(A, B, fullJFTsq, avoids, cost=None, normTarg=1):
    
    n = A.shape[-1]
    x = cp.Variable(n)

    if(cost is None): cost = np.ones(n)

    # Define the objective function (minimize L1 norm)
    objective = cp.Minimize(cp.norm(cp.multiply(x, cost), normTarg))

    # Define the constraints (Ax = B)
    constraints = [A @ x == B]
    
    if(len(avoids) > 0):

        normals = np.stack(avoids, axis=0)
        solSpeed = fullJFTsq @ x
        orthoTarg = np.zeros(len(avoids))
        constraints += [normals @ solSpeed == orthoTarg]

    # Formulate the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="ECOS_BB")

    # Get the sparse solution
    ssol = x.value

    return ssol