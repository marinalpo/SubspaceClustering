import numpy as np
from scipy.special import comb
#import torch
import matplotlib.pyplot as plt


class RwHoptCond:
    # Conditions for reweighted heuristic
    def __init__(self, maxIter, eigThres, corner):
        self.maxIter = maxIter  # number of max iterations
        self.eigThres = eigThres  # threshold on the eigenvalue fraction for stopping procedure
        self.corner = corner  # rank-1 on corner (1) or on full matrix (0)


def sortEigens(val, vec):
    """ Sort Eigenvalues in a descending order and sort its corresponding Eigenvectors
    :param val: Unsorted eigenvalues
    :param vec: Unsorted eigenvectors
    :return: sortedval: Sorted eigenvalues
    :return: sortedvec: Sorted eigenvectors
    """
    idx = val.argsort()[::-1]
    sortedval = val[idx]
    sortedvec = vec[:, idx]
    return sortedval, sortedvec


def plotNormalsAndPoints(normals, Xp, ss_ind, t, last):
    """ WARNING: Only for D = 2 and up to 7 Ns
    :param normals: DxNs matrix with normal vectors
    :param Xp: DxN matrix with the N points
    :param ss_ind: N array with indeces of the subspace each point belongs to
    :param t: String with title
    :param last: Boolean indicating if it is the last plot to show
    """
    c = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
    [a, Ns] = normals.shape
    for i in range(Ns):
        idx = np.where(ss_ind == i)
        Xs = Xp[:, idx[0]]
        x = np.arange(-10, 10)
        y = -(normals[0, i] / normals[1, i]) * (x)
        plt.plot(x, y, color=c[i])
        plt.scatter(Xs[0, :], Xs[1, :], c=c[i], edgecolors='k')
    plt.axis((-5, 5, -5, 5))
    plt.title(t)
    if last:
        plt.show()


def createNormalVectors(D, Ns):
    """ Generates a matrix of Normal Vectors from a random distribution
    :param D: dimension
    :param Ns: number of Subspaces
    :return: normals: DxNs matrix with normal vectors
    """
    normals = np.random.randn(D, Ns)
    normals = (np.matmul(normals, np.diag(np.sign(normals[0, :])))) / (
        np.sqrt(np.diag(np.matmul(np.transpose(normals), normals))))
    b = np.argsort(-normals[0, :])
    normals = normals[:, b]
    return normals


def generatePoints(normals, num_points, noise_b, sq_side):
    """ Generates points from a set of subspaces, by sampling from a square
        of D-1 dimensions and rotating about the normal.
    :param normals: DxS matrix with the S normals in D dimensions
    :param num_points: Sx1 vector with the number of points to be sampled in each subspace
    :param noise_b: noise bound
    :param sq_side: size of the square from which we sample
    :return: X: DxN matrix with the N points
    :return: ss_ind: N array with indeces of the subspace each point belongs to
    """
    N = np.sum(num_points)
    [D, S] = normals.shape
    X = np.zeros([D, N])
    ss_ind = np.zeros([N, 1])
    k = 0
    for ss in range(S):
        X_tmp = np.vstack((2 * (np.random.uniform(0, 1, [1, num_points[ss]]) - 0.5) * noise_b,
                           2 * (np.random.uniform(0, 1, [(D - 1), num_points[ss]]) - 0.5) * sq_side))
        SVD = np.linalg.svd(
            np.eye(D) - (1 / np.sqrt(np.dot(normals[:, ss], normals[:, ss]))) * np.outer(normals[:, ss], normals[:, ss]))
        U = np.fliplr(SVD[0])
        X_tmp = np.matmul(U, X_tmp)
        X[:, k:(k + num_points[ss])] = X_tmp
        ss_ind[k:(k + num_points[ss])] = ss * np.ones((num_points[ss], 1))
        k = k + num_points[ss]
    return X, ss_ind

def varietySample(V, x, count_max, R2, eps_bound):
    """ Samples points on a variety using random intersection lines
        Variety may be noise-corrupted
    :param V: Polynomial (variety) to be sampled. 
    :param x: Data variables
    :param count_max: Minimum number of points to return
    :param R2: Radius squared encompassing region of interest, especially when compact
    :param eps_bound: noise range in epsilon
    """
    
    #x = np.array(V.gens);
    N = len(x)
    t = sp.symbols("t")         #Parameter of line
    #count_max = 200
    count = 0
    #N = length(x)
    P = np.empty([N,0])
    while count < count_max:
        # Corrupt the variety with bounded noise    
        epsilon = np.random.uniform(-eps_bound, eps_bound)
        Ve = V + epsilon    
    
        # Get a line u + v t in space    
        U = sp.Matrix(np.random.randn(2, N+1));
        Ur = np.array(U.rref()[0].transpose().tolist(), dtype=float)
        u = Ur[1:, 0]
        v = Ur[1:, 1]
        
        L = u + v * t    
        
        #substitute in the line and find real roots
        VL = Ve.subs([i for i in zip(x, L)])
        cVL = sp.Poly(VL).coeffs()
        rVL = np.roots(cVL)
        r_real = np.real(rVL[np.isreal(rVL)])
        
        
        
        #recover points of intersection and append to array
        p = u[:, np.newaxis]  + np.outer(v, r_real)   
        pnorm = np.sum(p**2, 0)
        
        pcand= p[:, pnorm <= R2]
        
        P = np.concatenate([P, pcand], 1)
        
        #start new sampling iteration
        count = count + np.size(pcand, 1)
        
    return P