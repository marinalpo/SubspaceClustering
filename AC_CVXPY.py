import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2020)

def dataLift1D(Xp, n):
    [D, Np_total] = Xp.shape
    V = np.zeros((n+2, Np_total))
    x = Xp[0,:]
    for i in range(n + 1):
        V[i,:] = np.power(x,i)
    V[-1,:] = Xp[1,:]
    return V


def dataUnlift1D(R, eps, degmax):
    [Npoly, Np_total] = R.shape
    C = np.zeros((Npoly, degmax + 1))
    for p in range(Npoly):
        c = np.zeros((1, degmax + 1))
        c = R[p, :-1]
        c[0] = c[0] - eps
        C[p,:] = -np.multiply((1/R[p,-1]), c)
    return C


def createPolynomials(D, Npoly, degmax, cmax):
    # Generates Npoly polynomials defined by its coefficients and powers
    # Inputs: - D: Dimension (fixed to 2 at this moment)
    #         - Npoly: Number of polynomials to be created
    #         - alpha: Powers of all polynomials
    #         - cmax: Absolute maximum value of the polynomials coefficients
    # Outputs: - C: Array containing the coefficients of each polynomial
    #            Ex// c1 + c2*x + c3*y + c4*x^2 + c5*xy + c6*y^2
    C = []
    for p in range(Npoly):  # for each polynomial
        c = np.random.randint(-cmax, cmax, size=len(alpha))
        print('Polynomial ', p)
        print(' - alpha:', alpha)
        print(' - c:', c)
        print('\n')
        C.append(c)
    return C

def evaluatePoly(x, c, alpha):
    # Evaluates polynomial defined by c and alpha in all points of vector x
    # Inputs: - x: Array of points
    #         - c: Polynomial coefficients
    #         - alpha: Polynomial powers
    # Outputs: - y: Array containing polynomial evaluations of x
    y = np.zeros(len(x))
    for i in range(len(c)):
        y = y + c[i]*np.power(x, alpha[i])
    return y

def generatePointsPoly(C, alpha, num_points, noise_b, sq_size):
    # Generates noisy points corresponding to different polynomials
    # Inputs: - C: Array containing the coefficients of each polynomial
    #         - alpha: Array containing the powers of the polynomials
    #         - num_points: Array containing number of points per polynomial
    #         - noise_b: Noise bound
    #         - sq_size: Horizontal sampling window
    # Outputs: - Xp: 2xN matrix with the N points
    #          - ss_ind: N array indicating the polynomial label
    Npoly = len(C)
    N = np.sum(num_points)
    ss_ind = np.zeros([N, 1])
    Xp = np.zeros([2, N])  # For multidimension change 2 for D
    for p in range(Npoly):  # For each poly
        x = np.random.uniform(-sq_size,sq_size,num_points[0])
        y = evaluatePoly(x, C[p], alpha) + noise_b * np.random.uniform(-1,1,num_points[0])
        ini = np.sum(num_points[0:p])
        fin = np.sum(num_points[0:(p+1)])
        Xp[0,ini:fin] = x
        Xp[1,ini:fin] = y
        ss_ind[ini:fin] = p
    return Xp, ss_ind

def generatePointsLines(C, alpha, sq_size):
    # Generates dense points corresponding to different polynomials
    # Inputs: - C: Array containing the coefficients of each polynomial
    #         - A: Array containing the powers of each polynomial
    #         - sq_size: Horizontal sampling window
    # Outputs: - Xline: 2x(Npoly*pointsxpoly) matrix with the points of the polynomials
    step = 0.1
    Npoly = len(C)
    pointsxpoly = int((2*sq_size)*(1/step))
    Xp_line = np.zeros([2, Npoly * pointsxpoly])  # For multidimension change 2 for D
    for p in range(Npoly):
        x = np.arange(-sq_size,sq_size, step)
        y = evaluatePoly(x, C[p], alpha)
        ini = pointsxpoly*p
        fin = pointsxpoly*(p+1)
        Xp_line[0, ini:fin] = x
        Xp_line[1, ini:fin] = y
    return Xp_line

def plotLinesAndPoints(Xline, Xp, Npoly, num_points, t):
    # Plots lines corresponding to the polynomials and scatters the noisy points
    # Inputs: - Xline: 2x(Npoly*pointsxpoly) matrix with the points of the polynomials
    #         - Xp: 2xN matrix with the N points
    #         - Npoly: Number of polynomials
    #         - num_points: Array containing number of points per polynomial
    c = ['r','b','g','m','c','k','y']
    s = Xline.shape
    pointsxpoly = int(s[1]/Npoly)
    for p in range(Npoly):
        ini = pointsxpoly*p
        fin = pointsxpoly*(p+1)
        xline = Xline[0, ini:fin]
        yline = Xline[1, ini:fin]
        ini = np.sum(num_points[0:p])
        fin = np.sum(num_points[0:(p+1)])
        x = Xp[0, ini:fin]
        y = Xp[1, ini:fin]
        plt.plot(xline, yline, color=c[p])
        plt.scatter(x[:], y[:], c=c[p], edgecolors='k')
    plt.title(t)

def plotLinesAndPointsPredicted(Xline, Xline_pluseps, Xline_minuseps, Xp,  Npoly, num_points, t):
    # Plots lines corresponding to the polynomials and scatters the noisy points
    # Inputs: - Xline: 2x(Npoly*pointsxpoly) matrix with the points of the polynomials
    #         - Xp: 2xN matrix with the N points
    #         - Npoly: Number of polynomials
    #         - num_points: Array containing number of points per polynomial
    c = ['r','b','g','m','c','k','y']
    s = Xline.shape
    pointsxpoly = int(s[1]/Npoly)
    for p in range(Npoly):
        ini = pointsxpoly*p
        fin = pointsxpoly*(p+1)
        xline = Xline[0, ini:fin]
        yline = Xline[1, ini:fin]
        xline_pluseps = Xline_pluseps[0, ini:fin]
        yline_pluseps = Xline_pluseps[1, ini:fin]
        xline_minuseps = Xline_minuseps[0, ini:fin]
        yline_minuseps = Xline_minuseps[1, ini:fin]
        ini = np.sum(num_points[0:p])
        fin = np.sum(num_points[0:(p+1)])
        x = Xp[0, ini:fin]
        y = Xp[1, ini:fin]
        plt.plot(xline, yline, color=c[p])
        plt.plot(xline_pluseps, yline_pluseps, color=c[p], linestyle = ':')
        plt.plot(xline_minuseps, yline_minuseps, color='k', linestyle = ':' )
        plt.scatter(x[:], y[:], c=c[p], edgecolors='k')
    plt.title(t)

# D = 2  # Number of dimensions
# Npoly = 2 # Number of polynomials
# cmax = 10  # Absolute maximum value of the polynomials coefficients
# degmax = 3  # Maximum value of the polynomials degree
# noise_b = 500  # Noise bound
# Np = 15  # Constant Np points per polynomial
# num_points = (Np * np.ones(Npoly)).astype(int)  # Number of points per polynomial
# sq_size = 10  # Horizontal sampling window
#
# (C, A) = createPolynomials(D, Npoly, degmax, cmax)
# (Xp, ss_ind) = generatePointsPoly(C, A, num_points, noise_b, sq_size)
# Xline = generatePointsLines(C, A, sq_size)
# plotLinesAndPoints(Xline, Xp, Npoly, num_points)



def findXRange(coef, eps):
    # xrange[p, 0] - lower root
    # xrange[p, 1] - higher root
    # xrange[p, 2] - 1 if polynomial has no roots
    # xrange[p, 3] - 0 if polynomial is CONVEX, 1 if it is CONCAVE
    (Npoly, D_lift) = coef.shape
    xrange = np.zeros((Npoly, 4))
    noRoots = False
    concave = True
    x = np.arange(-10,10,0.1)
    for p in range(Npoly):
        a = np.power(coef[p, 4], 2) - 4*coef[p, 5]*coef[p, 3]
        b = - 4*coef[p, 5]*coef[p, 1]
        c = np.power(coef[p, 2], 2) + 2*coef[p, 2]*coef[p, 4] - 4*coef[p, 5]*coef[p, 0] + 4*coef[p, 5]*eps
        dis = np.power(b,2) - 4 * np.multiply(a, c)
        sol1 = (-b - np.sqrt(dis)) / (2 * a)
        sol2 = (-b + np.sqrt(dis)) / (2 * a)
        xrange[p, 0] = np.min([sol1, sol2])
        xrange[p, 1] = np.max([sol1, sol2])
        if np.isnan(xrange[p,0]):
            xrange[p,2] = 1
            if c < 0:
                xrange[p, 3] = 1
        else:
            xmid = 0.5*(xrange[p,0] + xrange[p,1])
            y_xmid = c + b*xmid + a*np.power(xmid,2)
            if y_xmid > 0:
                xrange[p, 3] = 1
    # print('xrange:\n', np.around(xrange, decimals = 1))
    return xrange

def generateXpoints(xrange, Np, sq_size, random):
    (Npoly, d) = xrange.shape
    xp = np.zeros((Npoly, Np))
    delPol = []
    for p in range(Npoly):
        if xrange[p, 2] == 0:  # Roots are real
            if xrange[p, 3] == 0:  # Polynomial is convex
                if random:
                    xp1 = np.random.uniform(-sq_size, xrange[p,0], int(Np/2))
                    xp2 = np.random.uniform(xrange[p,1], sq_size, int(Np/2))
                    xp[p, :] = np.hstack((xp1[:int(Np/2)], xp2[:int(Np/2)]))
                else:
                    xp1 = np.arange(-sq_size, xrange[p,0], (xrange[p,0]+sq_size)/(int(Np/2)))
                    xp2 = np.arange(xrange[p,1], sq_size, (sq_size-xrange[p,1])/(int(Np/2)))
                    xp[p, :] = np.hstack((xp1[:int(Np/2)], xp2[:int(Np/2)]))
            else:  # Polynomial is concave
                if random:
                    xp[p, :] = np.random.uniform(xrange[p,0], xrange[p,1], Np)
                else:
                    xp[p, :] = np.arange(xrange[p,0], xrange[p,1], (xrange[p,1]-xrange[p,0])/(Np-0.5))
        else:  # No real roots
            if xrange[p, 3] == 0:  # Polynomial is convex
                if random:
                    xp[p, :] = np.random.uniform(-sq_size, sq_size, Np)
                else:
                    xp[p, :] = np.arange(-sq_size, sq_size, (2*sq_size) / Np)
            else:
                print('Polynomial is concave without real roots. It will not be considered')
                delPol.append(p)
    # print('xp:\n', np.around(xp, decimals=1))
    return xp, delPol


def generateYp(xp, coef, sq_size, eps, GT, sol_num):
    (Npoly, Np) = xp.shape
    yp = np.zeros((Npoly,Np))
    sol = np.zeros((2,Np))
    cul = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
    labels = np.zeros((Npoly, Np))
    for p in range(Npoly):
        a = coef[p,5] * np.ones(Np)
        b = coef[p,2] + coef[p,4]*xp[p,:]
        if GT:
            c = coef[p, 0] + coef[p, 1] * xp[p, :] + coef[p, 3] * np.power(xp[p, :], 2) + eps
        else:
            c = coef[p, 0] + coef[p, 1] * xp[p, :] + coef[p, 3] * np.power(xp[p, :], 2) - np.random.uniform(-eps, eps,
                                                                                                            size=(
                                                                                                            1, Np))
        dis = np.power(b,2) - 4 * np.multiply(a, c)
        sol[0,:] = (-b - np.sqrt(dis)) / (2 * a)
        sol[1,:] = (-b + np.sqrt(dis)) / (2 * a)
        if GT:
            yp[p, :] = sol[sol_num,:]
        else:
            sol_idx = np.random.randint(2, size =(1,Np))
            yp[p, :] = sol[sol_idx, np.arange(Np)]
        labels[p,:] = p * np.ones((1, Np))
        # plt.scatter(xp[p, :], yp[p, :], c=cul[p])
    # plt.axis((- sq_size, sq_size, - sq_size, sq_size))
    # plt.show()
    return yp, labels


def plotGT(coef, xp, yp, sq_size, eps):
    (Npoly, Np) = xp.shape
    Np_GT = int(1e5)  # Number of points per polynomial to plot GT
    noise = [0, eps, -eps]
    l = ['-', ':', ':']
    cul = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
    xp_GT = np.arange(-sq_size, sq_size, (2 * sq_size) / Np_GT)
    xp_GT = np.tile(xp_GT, (Npoly, 1))
    for i in range(3):
        yp_GT_1, labels_GT = generateYp(xp_GT, coef, sq_size, noise[i], True, 0)
        yp_GT_2, labels_GT = generateYp(xp_GT, coef, sq_size, noise[i], True, 1)
        for p in range(Npoly):
            plt.plot(xp_GT[p, :], yp_GT_1[p, :], c=cul[p], linestyle = l[i])
            plt.plot(xp_GT[p, :], yp_GT_2[p, :], c=cul[p], linestyle = l[i])
            plt.scatter(xp[p, :], yp[p, :], color=cul[p], edgecolors='k')
    plt.title('Ground Truth')
    plt.axis((- sq_size, sq_size, - sq_size, sq_size))
    #  If axis are limited, there might be points that are created but not shown in the plot
