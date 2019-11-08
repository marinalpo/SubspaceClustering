import numpy as np
import torch
from scipy.special import comb
import matplotlib.pyplot as plt

class RwHoptCond:
    # conditions for reweighted heuristic
    def __init__(self, maxIter, eigThres, corner):
        self.maxIter = maxIter  # number of max iterations
        self.eigThres = eigThres  # threshold on the eigenvalue fraction for stopping procedure
        self.corner = corner  # rank-1 on corner (1) or on full matrix (0)


def sortEigens(val, vec):
    idx = val.argsort()[::-1]
    sortedval = val[idx]
    sortedvec = vec[:, idx]
    return sortedval, sortedvec


def plotNormalsAndPoints(normals, Xp, ss_ind, t, last):
    # ONLY FOR 2 DIMENSIONS AND UP TO 7 SUBSPACES
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
    # Generates a matrix of Normal Vectors from a random distribution
    # Inputs: - D: dimension
    #         - Ns: number of Subspaces
    # Outputs: - Normals: DxNs matrix with normal vectors
    normals = np.random.randn(D, Ns)
    normals = (np.matmul(normals, np.diag(np.sign(normals[0, :])))) / (
        np.sqrt(np.diag(np.matmul(np.transpose(normals), normals))))
    b = np.argsort(-normals[0, :])
    normals = normals[:, b]
    return normals


def generatePoints(normals, num_points, noise_b, sq_side):
    # Generates points from a set of subspaces, by sampling from a square of D-1 dimensions and rotating about the normal.
    # Inputs: - Normals: DxS matrix with the S normals in D dimensions
    #         - num_points: Sx1 vector with the number of points to be sampled in each subspace
    #         - noise_b: noise bound
    #         - sq_side: size of the square from which we sample
    # Outputs: - X: DxN matrix with the N points
    #          - ss_ind: index of the subspace
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


def exponent_nk(n, K):
        id = np.diag(np.ones(K))
        exp = id
        for i in range(1, n):
            rene = np.asarray([])
            for j in range(0, K):
                for k in range(exp.shape[0] - int(comb(i + K - j - 1, i)), exp.shape[0]):
                    if rene.shape[0] == 0:
                        rene = id[j, :] + exp[k, :]
                        rene = np.expand_dims(rene, axis=0)
                    else:
                        rene = np.concatenate([rene, np.expand_dims(id[j, :] + exp[k, :], axis=0)], axis=0)
            exp = rene.copy()
        return exp


def veronese_nk(x, n, if_cuda=False, if_coffiecnt=False):
    '''
     Computes the Veronese map of degree n, that is all
     the monomials of a certain degree.
     x is a K by N matrix, where K is dimension and N number of points
     y is a K by Mn matrix, where Mn = nchoosek(n+K-1,n)
     powers is a K by Mn matrix with the exponent of each monomial

     Example veronese([x1;x2],2) gives
     y = [x1^2;x1*x2;x2^2]
     powers = [2 0; 1 1; 0 2]

     Copyright @ Rene Vidal, 2003
    '''
    if if_coffiecnt:
        assert n == 2
    K, N = x.shape[0], x.shape[1]
    powers = exponent_nk(n, K)
    if if_cuda:
        powers = torch.tensor(powers, dtype=torch.float).cuda()
    else:
        powers = torch.tensor(powers, dtype=torch.float)
    if n == 0:
        y = 1
    elif n == 1:
        y = x
    else:
        s = []
        for i in range(0, powers.shape[0]):
            tmp = x.t().pow(powers[i, :])
            ttmp = tmp[:, 0]
            for j in range(1, tmp.shape[1]):
                ttmp = torch.mul(ttmp, tmp[:, j])
            if if_coffiecnt:
                if powers[i, :].max() == 1:
                    ttmp = ttmp * 1.4142135623730951
            s.append(ttmp.unsqueeze(dim=0))
        y = torch.cat(s, dim=0)
    return y, powers


def generate_veronese(x, n):
    """Concatenates the results of veronese_nk function to generate the complete veronese map of x
        @param x: Matrix (dim, npoints)
        @param n: the veronese map will be up to degree n

        Output: the complete veronese map of x (veronese_dim, BS)
        Example:
        if x is a two dimensional vector x = [x1 x2]
        generate_veronese(x, 2) ==> [1 x1 x2 x1^2 x1*x2 x2^2]
        generate_veronese(x, 3) ==> [1 x1 x2 x1^2 x1*x2 x2^2 x1^3 x1^2*x2 x1*x2^2 x2^3]
    """
    v_x = x
    p_x = None
    for i in range(0, n - 1):
        v_x_n, p_n = veronese_nk(x, i + 2, if_cuda=False, if_coffiecnt=False )
        v_x = torch.cat([v_x, v_x_n], dim=0)
    v_x = torch.cat([torch.ones(1, v_x.size()[1]), v_x])
    return v_x, p_x