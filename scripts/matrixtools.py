import numpy as np

def explosion(M, x, y, index = 0): # M[i] = x*M[index] + y*M[i] except i = index
    N = M.copy()
    for i in range(len(N)):
        if i != index:
            N[i] = x*N[index] + y*N[i]
    L = np.delete(N, index, axis=0)
    return L

# ---

def compose(f, g):
    def composed_function(x):
        return f(g(x))
    composed_function.__name__ = f"{f.__name__}*{g.__name__}"
    return composed_function

def _id(M):
    return M

def _unify(M):
    return M / np.linalg.norm(M, axis=1)[:, np.newaxis]

def unify(MatFunc):
    def wrapper(M):
        return MatFunc(_unify(M))
    wrapper.__name__ = f"{MatFunc.__name__}*unify"
    return wrapper

def countVec(v, error=1e-3, numberOfPoints = None):
    set = []
    if numberOfPoints is None:
        numberOfPoints = len(v)
    for i in range(len(v)):
        alreadyin = False
        for j in set:
            if abs(j[0] - v[i]) < error:
                j[1] += 1
                alreadyin = True
                break
        if not alreadyin:
            set.append([v[i], 1])
    set.sort()
    for s in set:
        s.append(s[1] / numberOfPoints)
    return set

def _countMat(M, error=1e-3):
    v = M.flatten()
    return countVec(v, error, len(M))

def _rowMinimumSet(M, error = 1e-3):
    v = np.min(M, axis=1)
    return countVec(v, error)

def _rowMultiple(M, value, error=1e-3):
    # returns a vector, where n-th element is "how many of M[n] are equal to value, with error of error"
    return countVec(np.count_nonzero(np.abs(M - value) < error, axis=1))

def _beautifulEig(M):
    N = np.linalg.eig(M)
    evals = N[0]
    evecs = N[1]
    eigList = []
    for i in range(len(evals)):
        evec = evecs[i]
        evec /= np.linalg.norm(evec)
        eigList.append(np.concatenate([np.array([evals[i], np.nan]),  evec]))
    return np.array(eigList) 

@unify
def _iprod(M):
    return M @ M.T

def iMat(MatFunc):
    def wrapper(M):
        return MatFunc(_iprod(M))
    wrapper.__name__ = f"{MatFunc.__name__}*iprod*unify"
    return wrapper

@unify
def _jprod(M):
    return M.T @ M

def jMat(MatFunc):
    def wrapper(M):
        return MatFunc(_jprod(M))
    wrapper.__name__ = f"{MatFunc.__name__}*jprod*unify"
    return wrapper

# ---
# @jMat
# def _charPoly(V):
#     M = V
#     div = np.linalg.trace(M) / len(M)
#     M = M / div
#     return np.flip(np.poly(M))

@jMat
def _jeig(V):
    return _beautifulEig(V)


@jMat
def _jdet(V):
    return np.linalg.det(V)

@iMat
def _allDist(V):
    return _countMat(V)

@iMat
def _furthestPointDist(V):
    return _rowMinimumSet(V)

@iMat
def _closestPointNumber(V):
    return _rowMultiple(V, 0.5, error=1e-2)


# --- unfinished



def _rank(M):
    return np.linalg.matrix_rank(M)

def _trace(M):
    return np.trace(M)
def _det(M):
    return np.linalg.det(M)

def _inv(M):
    return np.linalg.inv(M)

def _svd(M):
    return np.linalg.svd(M)

def _eig(M):
    return np.linalg.eig(M)

def _qr(M):
    return np.linalg.qr(M)

def _chol(M):
    return np.linalg.cholesky(M)

def _svdvals(M):
    return np.linalg.svdvals(M)

def _cond(M):
    return np.linalg.cond(M)