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


def exponent_nk(n, K):
    """ Computes exponents of the veronese map of degree n
    :param n: Degree of the Veronese map
    :param K: Sx1 vector with the number of points to be sampled in each subspace
    :return: exp: DxN matrix with the N points
    """
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
    """ Computes the Veronese map of degree n, (all the monomials of a certain degree)
     Example: veronese([x1;x2],2) gives y = [x1^2;x1*x2;x2^2] and powers = [2 0; 1 1; 0 2]
     Copyright @ Rene Vidal, 2003
    :param x: K by N matrix, where K is dimension and N number of points
    :param n: degree of the veronese map
    :param if_cuda: boolean
    :param if_coffiecnt: boolean
    :return: y: K by Mn matrix, where Mn = nchoosek(n+K-1,n)
    :return: powers: K by Mn matrix with the exponent of each monomial
    """
    if if_coffiecnt:
        assert n == 2
    K, N = x.shape[0], x.shape[1]
    powers = exponent_nk(n, K)
    if if_cuda:
        powers = torch.tensor(powers, dtype=torch.float)
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
    Example: If x is a two dimensional vector x = [x1 x2]
        generate_veronese(x, 2) ==> [1 x1 x2 x1^2 x1*x2 x2^2]
        generate_veronese(x, 3) ==> [1 x1 x2 x1^2 x1*x2 x2^2 x1^3 x1^2*x2 x1*x2^2 x2^3]
    :param x: (dim)x(npoints) matrix
    :param n: degree of the veronesse map
    :return: v_x: the complete veronese map of x (veronese_dim, BS)
    :return: p_y: exponentials of the veronese map
    """
    v_x = x
    p_x = None
    for i in range(0, n - 1):
        v_x_n, p_n = veronese_nk(x, i + 2, if_cuda=False, if_coffiecnt=False, )
        v_x = torch.cat([v_x, v_x_n], dim=0)
    v_x = torch.cat([torch.ones(1, v_x.size()[1]).float(), v_x.float()])
    return v_x, p_x


def findXRange(coef, eps):
    """ Finds X range that corresponds to polynomial at level set = eps
    :param coef: 1x6 matrix containing coefficients of the polynomial
    :param eps: Level set
    :return: xrange: 1x2 array containing range's beginning and ending
    :return: a: Quadratic coefficient
    :return: b: Linear coefficient
    :return: c: Independent coefficient
    """
    xrange = np.zeros(2)
    a = np.power(coef[4], 2) - 4 * coef[5] * coef[3] + 0.000001
    b = 2 * coef[2] * coef[4] - 4 * coef[5] * coef[1]
    c = np.power(coef[2], 2) - 4 * coef[5] * coef[0] + 4 * coef[5] * eps
    dis = np.power(b, 2) - 4 * np.multiply(a, c)
    sol1 = (-b - np.sqrt(dis)) / (2 * a)
    sol2 = (-b + np.sqrt(dis)) / (2 * a)
    xrange[0] = np.min([sol1, sol2])
    xrange[1] = np.max([sol1, sol2])
    return xrange, a, b, c


def findXPointsRange(coef, eps, Npoly):
    """ Finds X range from where to sample random points and classifies polynomials in 4 groups
    :param coef: (Npoly_ext)x(D) Matrix with coefficients from polynomials
    :param eps: Level set (noise bound)
    :param Npoly: Number of polynomials to be considered
    :return: xPointsRange:  X range from where to to sample random points
    :return: typeRange: Type of range value
    :return: coef: (Npoly)x(D) Matrix with coefficients from considered polynomials
    """
    delPol = []
    (Npoly_ext, d) = coef.shape
    xPointsRange = np.zeros((Npoly_ext, 2))
    typeRange = np.zeros(Npoly_ext)
    for p in range(Npoly_ext):
        xrange, a, b, c = findXRange(coef[p, :], 0)
        if np.isnan(xrange[0]):  # range has no real roots
            typeRange[p] = 2
            xPointsRange[p, :] = xrange
            if c < 0:  # range is CONCAVE
                # Polynomial is concave without real roots. It will not be considered
                delPol.append(p)
        else:
            xmid = 0.5*(xrange[0] + xrange[1])
            y_xmid = c + b*xmid + a*np.power(xmid, 2)
            if y_xmid > 0:
                typeRange[p] = 1  # range is CONCAVE
                xrange_meps, a, b, c = findXRange(coef[p, :], -eps)
                if np.isnan(xrange_meps[0]):
                    xrange_meps = 1e-5 * np.ones(2)
                xrange_eps, a, b, c = findXRange(coef[p, :], eps)
                if np.isnan(xrange_eps[0]):
                    xrange_eps = 1e-5 * np.ones(2)
                if xrange_eps[1] > xrange_meps[1]:  # Polynomial is a "bowl"
                    xPointsRange[p, :] = xrange_eps
                else:  # Polynomial is a "hat"
                    xPointsRange[p, :] = xrange_meps
            else:
                typeRange[p] = 0  # range is CONVEX
                xrange_eps, a, b, c = findXRange(coef[p, :], eps)
                if np.isnan(xrange_eps[0]):
                    xrange_eps = 1e-5 * np.ones(2)
                    typeRange[p] = 2
                xrange_meps, a, b, c = findXRange(coef[p, :], -eps)
                if np.isnan(xrange_meps[0]):
                    xrange_meps = 1e-5 * np.ones(2)
                    typeRange[p] = 2
                if xrange_eps[1] > xrange_meps[1]:
                    xPointsRange[p, :] = xrange_meps
                else:
                    xPointsRange[p, :] = xrange_eps
    # Delete concave polynomials with no real roots
    xPointsRange = np.delete(xPointsRange, delPol, 0)
    xPointsRange = xPointsRange[0:Npoly, :]
    coef = np.delete(coef, delPol, 0)
    coef = coef[0:Npoly, :]
    typeRange = np.delete(typeRange, delPol)
    typeRange = typeRange[0:Npoly]
    return xPointsRange, typeRange, coef


def generateXpoints(xrange, typeRange, Np, sq_size):
    """ Generates random x points considering the polynomial range type
    :param xrange: X range from where to to sample random points
    :param typeRange: Type of range value
    :param Np: Desired number of random points
    :param sq_size: Sampling window size
    :return: xp: (Npoly)x(Np) Random x points sampled in the range of each polynomial
    """
    (Npoly, d) = xrange.shape
    Np = Np * 10   # Create extra points to avoid NaN s
    xp = np.zeros((Npoly, Np))
    for p in range(Npoly):
        if typeRange[p] == 0:  # CONVEX and roots are real
            xp1 = np.random.uniform(-sq_size, xrange[p,0], int(Np/2))
            xp2 = np.random.uniform(xrange[p,1], sq_size, int(Np/2))
            xp_stack = np.hstack((xp1[:int(Np/2)], xp2[:int(Np/2)]))
            np.random.shuffle(xp_stack)
            xp[p,:] = xp_stack
        elif typeRange[p] == 1:  # CONCAVE and roots are real
            xp[p, :] = np.random.uniform(xrange[p, 0], xrange[p, 1], Np)
        elif typeRange[p] == 2:  # CONVEX but roots are not real
            xp[p, :] = np.random.uniform(-sq_size, sq_size, Np)
    return xp


def generateYp(xp, coef, Np, eps, GT, sol_num):
    """ Finds the y values of the sampled x points at a random level set
    :param xp: (Npoly)x(Np_extra) matrix containing sampled points
    :param coef: (Npoly)x(D) Matrix with coefficients from considered polynomials
    :param Np: Number of points
    :param eps: Level set (noise bound)
    :param GT: Boolean. Ground Truth
    :param sol_num: scalar indicating if the upper(0) or lower(1) solution is desired
    :return: xp_out: (Npoly)x(Np) Matrix with the x points for each polynomial
    :return: yp_out: (Npoly)x(Np) Matrix with the y points for each polynomial
    :return: labels: (Npoly)x(Np) Matrix with the labels of each points for each polynomial
    """
    (Npoly, Np_extra) = xp.shape
    xp_out = np.zeros((Npoly, Np))
    yp_out = np.zeros((Npoly, Np))
    sol = np.zeros((2, Np_extra))
    labels = np.zeros((Npoly, Np))
    for p in range(Npoly):
        a = coef[p,5] * np.ones(Np_extra) + 0.000001
        b = coef[p,2] + coef[p,4]*xp[p,:]
        if GT:
            c = coef[p, 0] + coef[p, 1] * xp[p, :] + coef[p, 3] * np.power(xp[p, :], 2) + eps
        else:
            noise = np.random.uniform(-eps, eps, size=(1, Np_extra))
            c = coef[p, 0] + coef[p, 1] * xp[p, :] + coef[p, 3] * np.power(xp[p, :], 2) - noise
        dis = np.power(b,2) - 4 * np.multiply(a, c)
        sol[0,:] = (-b - np.sqrt(dis)) / (2 * a)
        sol[1,:] = (-b + np.sqrt(dis)) / (2 * a)
        if GT:
            yp_out[p,:] = sol[sol_num,:]
            xp_out = xp
        else:
            sol_idx = np.random.randint(2, size =(1, Np_extra))
            yp = sol[sol_idx, np.arange(Np_extra)]
            # Delete points with NaN
            idx_nan = np.where(np.isnan(yp))
            yp = np.delete(yp, idx_nan)
            xp_poly = np.delete(xp[p, :], idx_nan, 0)
            try:
                yp_out[p,:] = yp[0:Np]
                xp_out[p, :] = xp_poly[0:Np]
            except ValueError:
                print('Polynomial', p, 'Dismissed')
        labels[p, :] = p * np.ones((1, Np))
    return xp_out, yp_out, labels


def plotInput(points, sq_size, Np, Npoly):
    """ Plots data information that is inputted to the algorithm
    :param points: 2x(NpxNpoly) matrix with the points inputted to the algorithm
    :param sq_size: Size of the sampling window
    :param Np: Number of points for each polynomial
    :param Npoly: Number of polynomials
    """
    plt.scatter(points[0,:], points[1,:], color = 'k')
    plt.title('Input to Algorithm\nVeronesse (deg=2) of the points')
    textstr = 'Npoly: ' + np.str(Npoly) + '\nNpoints: ' + np.str(Np) + ' (x'+ np.str(Npoly)+ ')'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    sq_size_y = 3 * sq_size
    plt.text(- sq_size + 0.5, sq_size_y - 3, textstr, fontsize=10,
             verticalalignment='top', bbox=props)
    plt.axis((- sq_size, sq_size, - sq_size_y, sq_size_y))
    #  If axis are limited, there might be points that are created but not shown in the plot


def plotData(coef, labels, xp, yp, sq_size, eps, Np, Npoly, tit):
    """ For each polynomial, the level set evaluation at 0, eps and -eps is plotted along with its
        randomly sampled points.
    :param coef: (Npoly)x6 Matrix with the coefficients for each polynomial
    :param labels: NpolyxNp Array with the labels of each point
    :param xp: NpolyxNp Array with the x values of each point
    :param yp: NpolyxNp Array with the y values of each point
    :param sq_size: Sampling window size
    :param eps: Noise bound
    :param Np: Number of points per polynomial
    :param Npoly: Number of polynomials
    :param tit: String containing Title
    """
    Np_GT = int(1e5)  # Number of points per polynomial to plot GT
    noise = [0, eps, -eps]
    lstyle = ['-', ':', ':']
    cul = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
    xp_GT = np.tile(np.arange(-sq_size, sq_size, (2 * sq_size) / Np_GT), (Npoly, 1))
    for i in range(3):
        xp_GT_1, yp_GT_1, labels_GT = generateYp(xp_GT, coef, Np_GT, noise[i], True, 0)
        xp_GT_2, yp_GT_2, labels_GT = generateYp(xp_GT, coef, Np_GT, noise[i], True, 1)
        for p in range(Npoly):
            idx = np.where(labels == p)
            plt.plot(xp_GT[p, :], yp_GT_1[p, :], c=cul[p], linestyle=lstyle[i])
            plt.plot(xp_GT[p, :], yp_GT_2[p, :], c=cul[p], linestyle=lstyle[i])
            plt.scatter(xp[idx[0]], yp[idx[0]], color=cul[p], edgecolors='k')
    plt.title(tit)
    textstr = 'Npoly: ' + np.str(Npoly) + '' \
                                          '\nNpoints: ' + np.str(Np) + ' (x' + np.str(Npoly) + ')'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    sq_size_y = 3 * sq_size
    plt.text(- sq_size + 0.5, sq_size_y - 3, textstr, fontsize=10,
             verticalalignment='top', bbox=props)
    plt.axis((- sq_size, sq_size, - sq_size_y, sq_size_y))
    #  If axis are limited, there might be points that are created but not shown in the plot