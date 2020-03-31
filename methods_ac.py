# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:11:54 2020

@author: Jared
"""

import numpy as np
import cvxpy as cp
import sympy as sp
from time import time
from utils_reduced import *

class AC_manager:
    """ Algebaric Clustering for multi-class model validation    
    
    Classify data into a set of polynomial models under bounded noise
    class contains all functionality needed to perform AC
    Each class is a model of the form f(x; theta) = 0
    Future: extend to semialgebraic clustering: f(x; theta) >= 0
    
    Parameters
    ----------
    x: sympy array
        Data (space) sympy variables
    th: sympy array
        parameter sympy variables, want to solve
    
    model: list of models (model class, semialgebraic sets)
        Polynomial models f(x; theta) that generate data
    mult: list of integers
        multiplicity of each model in data (e.g. 5 lines and 2 circles w/ radius 1)
    """
    
    def __init__(self, x, th):
        """Constructor for algebraic clustering"""
        self.x = x;
        self.th = th;


        #polynomial models under consideration
        self.model = []
        self.mult = []
        self.Nclass = 0

        
    def add_model(self, model_new, mult):
        """Add new polynomial model to system
        
        Parameters
        ----------
        model_new: model class
            New polynomial models f(x; theta) that generate data
            generalized to semialgebraic sets
        mult: integer
            multiplicity of each model in data (e.g. 5 lines and 2 circles w/ radius 1)      
        
        """        
        self.model += [model_new]
        self.mult  += [mult]
        self.Nclass += mult
                

    def moment_SDP(self):
        #roughly equivalent to AC_CVXPY, generate the SDP to classify data in Xp
        #or maybe generate model without data Xp, and then insert classification constraints. That sounds better.

        """
        Constraints in problem

        con_one:        [0,0...0] entry of moment matrix is 1
        con_mom:        Parameter (theta) entries obey moment structure
        con_geom:       Models obey geometric constraints g(theta) >= 0, h(theta) == 0

        con_bin:        binary classification: s^2 = s
        con_assign:     all datapoints are assigned to one and only one model class

        con_classify:   Models classify in-class data within epsilon

        C = con_one + con_assign + con_mom + con_bin + con_classify + con_geom

        """
        Mth = []
        C = []
        count = 0

        #Iterate through all classes
        for i in range(len(self.model)):
            cvx_out = self.model[i].generate_moment(self.mult[i])
            Mth += [cvx_out["Mth"]]
            C += cvx_out["C"]

        cvx_moment = {"Mth": Mth, "C": C}
        return cvx_moment

    def classify_SDP(self, Xp, eps, cvx_moment):
        """Constraints for model classification of data in Xp, as corrupted by bounded noise (eps)

        Parameters
        ----------
        Xp: numpy ndarray
            Data to be classified
        eps: double
            Noise bound in classification (assume scalar for now)
        cvx_moment:
            dictionary containing moment information for problem
        """
        
        return 0
        

    def generate_SDP(self, Xp, eps):
        cvx_moment = self.moment_SDP()
        cvx_classify = self.classify_SDP(Xp, eps, cvx_moment)


        return cvx_classify

    def solve_SDP(self, RwHopt):
        #solve the SDP through reweighted heuristic, or some other SDP method. r*-norm on nuclear norm?
        # Anything that sticks

        pass

class Model:
    """A model that might plausibly generate observed data (example circle, subspace, curve)"""
    def __init__(self, x, th):
        self.x = x
        self.th = th

        #depends on data and parameters
        self.eq = []    #f(x, theta) == 0
        self.ineq = []  #f(x, theta) >= 0

        #depends only on parameters (geometric constraints)
        self.eq_geom = []   #h(theta) == 0
        self.ineq_geom = [] #g(theta) >= 0

        self.moment = None

    #add functions to the semialgebraic set description
    def add_eq(self, eq_new):
        if type(eq_new) == list:
            self.eq += eq_new
        else:
            self.eq += [eq_new]

    def add_ineq(self, ineq_new):
        if type(ineq_new) == list:
            self.ineq += ineq_new
        else:
            self.ineq += [ineq_new]

    def add_eq_geom(self, eq_new):
        if type(eq_new) == list:
            self.eq_geom += eq_new
        else:
            self.eq_geom += [eq_new]

    def add_ineq_geom(self, ineq_new):
        if type(ineq_new) == list:
            self.ineq_geom += ineq_new
        else:
            self.ineq_geom += [ineq_new]

    def moment_constraint(self, eq, ineq, eq_geom, ineq_geom):
        """Form moment constraints given semialgebraic set description"""
        all_poly = ineq + eq + eq_geom + ineq_geom

        moment = ACmomentConstraint(all_poly, [self.x, self.th])
        return moment

    def generate_moment(self, mult):

        self.moment = self.moment_constraint(self.eq, self.ineq, self.eq_geom, self.ineq_geom)

        sizes = [len(self.eq), len(self.ineq), len(self.eq_geom), len(self.ineq_geom)]

        cvx_out = self.moment_SDP(sizes, self.moment, mult)

        return cvx_out


    def moment_SDP(self, sizes, mom, mult):
        """
        CVXPY constraints for moments, given semialgebraic set in model

        :param: sizes - [#eq, #ineq, #eq_geom, #ineq_geom]
        :param: moment - dictionary containing moment information for the current model
        :param: mult - multiplicity of current model in classification task

        :return: cvx_out - dictionary that contains the moment variable 'y' as well as constraints on moments 'C'
        """

        #moment matrix for each corner
        #turn this into vectorized moments later
        M_size = len(mom["supp"])
        Mth = [cp.Variable(M_size, M_size, symmetric=True) for i in range(mult)]


        #set up constraints
        con_mom = []
        con_geom = []

        #[0,0,...0] entry of moment matrix is 1
        Nth = len(self.th)
        ind_1 = mom["monom_all"].index(tuple([0]*Nth))
        con_one = [(Mth[i][ind_1, ind_1] == 1) for i in range(mult)]

        #matrix obeys moment structure
        con_mom = []
        for monom, ind_agree in moment["cons"].items():
            for iagree in range(len(ind_agree) - 1):
                iprev = ind_agree[iagree]
                inext = ind_agree[iagree + 1]
                con_mom+= [(Mth[i][iprev] == Mth[i][inext]) for i in range(mult)]

        #geometric constraints
        con_geom_eq = []
        con_geom_ineq = []
        sizesc = np.cumsum(sizes)

        #geometric equality constraints
        monom_eq = mom["monom_all"][sizesc[1]:sizesc[2]]
        monom_ind = [mom["lookup"][m][0] for m in monom_eq]

        #fb_eq = mom["fb"][sizesc[1]:sizesc[2]]
        coeff_eq = [mom["fb"][ieq](*([0] * Nth)) for ieq in range(sizesc[1], sizesc[2])]

        #eval_eq = [(monom_curr[count] @ coeff_curr == 0)for count in range(mult)]
        eval_eq = [(Mth[count].__getitem__(monom_ind[mi]) @ coeff_eq[mi] == 0) for count in range(mult) for mi in range(sizes[2])]

        con_geom_eq = eval_eq

        #inequality constraints
        monom_ineq = mom["monom_all"][sizesc[2]:sizesc[3]]
        monom_ind_ineq = [mom["lookup"][m][0] for m in monom_ineq]

        #fb_eq = mom["fb"][sizesc[1]:sizesc[2]]
        coeff_ineq = [mom["fb"][ieq](*([0] * Nth)) for ieq in range(sizesc[2], sizesc[3])]

        #eval_eq = [(monom_curr[count] @ coeff_curr == 0)for count in range(mult)]
        eval_ineq = [(Mth[count].__getitem__(monom_ind_ineq[mi]) @ coeff_ineq[mi] >= 0) for count in range(mult) for mi in range(sizes[3])]

        con_geom_ineq = eval_ineq

        #break symmetry between model classes in program with arbitrary constraint
        con_symmetry = [(Mth[count][0,0] >= Mth[count+1][0,0]) for count in range(mult-1)]

        C = con_one + con_mom + con_geom_eq + con_geom_ineq + con_symmetry

        cvx_out = {"Mth": Mth, "C", C}

        return cvx_out

    def classify_SDP(self, sizes, Mth):
        """Classify data in Xp, eps given current model Mth (no multiplicity here)"""

        #TODO Implement the classificatiton for a given model instance. This will require returning a set of matrices M
        # as well as a set of new constraints C linking them together. Call classify_SDP from AC_manager, and combine
        # together into a new program

        pass




def AC_CVXPY(Xp, eps, var, P, mult, RwHopt): #, , delta
    """ Subspace Clustering using Cheng Implementation
    :param Xp: DxNp matrix, each column is a point
    :param eps: Noise bound: allowed distance from the subspace
    :param var: sympy variables (x, th): x-data, th-parameters
    :param P: Polynomial classes in problem
    :param mult: multiplicity of each class in P
    :param RwHopt: conditions for reweighted heuristic contained on an object that includes:
                   - maxIter: number of max iterations
                   - eigThres: threshold on the eigenvalue fraction for stopping procedure
                   - corner: rank-1 on corner (1) or on full matrix (0)
    :param: delta: Noise factor on the identity at first iteration

    Need to revise the return section of documentation
    :return: R: tensor of length Ns, where each item is a (1+D)x(1+D) matrix with the subspace coordinates
    :return: S: NsxNp matrix, with labels for each point and subspace
    :return: runtime: runtime of the algorithm (excluding solution extraction)
    :return: rank1ness: rankness of every iteration
    """
    
    
    #This system will have cliques [1 parameter-monomials all data]
    #Will split up data later
    
    # Define dimension and number of points
    #x = var[0]
    #th = var[1]
    Mom_con = [ACmomentConstraint(p, var) for p in P]
    
    
    [D, Np] = Xp.shape
    Nx = len(var[0])
    Nth = len(var[1])

    Nclass = sum(mult)

    #number of monomials    
    Nm = [len(m["monom_all"]) for m in Mom_con]
    Ns = [len(m["supp"])  for m in Mom_con]
    
    #Nm = 1 + Ns * (Np + D)  # Size of the overall matrix


    M = [] #moment matrix
    Mth = [] #moment matrix restricted to parameter entries (theta)
    C = [] #constraints
    
    con_assign_pre = np.zeros([Np, 1])
    con_classify = []
    con_one = [] 
    con_mom = []
    con_bin = []
    con_assign = []
    count = 0
    
    
    #R = [None]*Nclass
    S  = [None]*Nclass
    TH_monom = [None]*Nclass
    TH = [None]*Nclass
    #S = cp.Variable([Np, Nclass])
    
    class_sum = []
    for ic in range(len(P)):
        
        #Moment Constraints
        mom = Mom_con[ic]
                
        TH_ind = [mom["cons"][tuple(i)][0] for i in np.eye(Nth).tolist()]
        
        for im in range(mult[ic]):
            m = cp.Variable([Ns[ic] + Np, Ns[ic]+Np], PSD=True)
            M.append(m)
            Mth.append(m[:Ns[ic], :Ns[ic]])
            
            #"corner" of moment matrix is 1
            #ind_1 = len(mom["monom_all"])-1
            ind_1 = mom["monom_all"].index(tuple([0]*Nth))
            con_one.append(M[count][ind_1, ind_1] == 1)
            #r_curr = M[count][ind_1, :Nm[ic]]
            #R[count] = r_curr    
            
            TH_monom[count] = Mth[count][ind_1, :]
            TH[count] = [Mth[count][thi] for thi in TH_ind]
            
            #entries of moment matrix agree with each other
            #like (xy)(xy) == (x^2) (y^2)
            for monom, ind_agree in mom["cons"].items():                
                for iagree in range(len(ind_agree)-1):
                    iprev = ind_agree[iagree]
                    inext = ind_agree[iagree+1]
                    con_mom.append(M[count][iprev] == M[count][inext])
            
            
            #each point is properly classified (function evaluation)
            for ip in range(Np):                
                i = ip + Ns[ic]
                
                coeff_curr = mom["fb"][0](*Xp[:, ip])
                
                #f_curr = 0
                rs_curr = M[count][i, :Nm[ic]]
                s_curr  = M[count][i, ind_1]
                
                #add in the new variable
                #con_S += [S[ip, count]  == s_curr]
                
                
                f_curr = coeff_curr @ rs_curr
                
                con_classify.append(-eps*s_curr <= f_curr)
                con_classify.append(f_curr <= eps*s_curr)
                
                
                if ip == 0:
                    S[count] = [s_curr]
                else:
                    S[count] += [s_curr]    
                
                if count == 0:
                    class_sum += [s_curr]
                else:
                    class_sum[ip] += s_curr
                
                #all classifications are binary
                con_bin += [M[count][i, ind_1] == M[count][i,i]]
            
            #prepare for new moment matrix
            count += 1

    #unique classification of points


    for ip in range(Np):
        con_assign += [cp.sum(class_sum[ip]) == 1]

    #con_assign = [class_sum == 1]
    
    
    #C = con_one + con_assign + con_mom + con_bin
    
    #still need to add in extra constraints, instituting relations among/between classes
    #like if a set of planes are perpendicular, or to normalize a projective curve
    C = con_one + con_assign + con_mom + con_bin + con_classify

    

    #develop the objective
    W = [np.eye(mth.shape[0]) + 1e-1*np.random.randn(mth.shape[0]) for mth in Mth]
    objective = 0
    #for i = 1:Nclass
    #    objective = cth[i] @ Mth[i]
    
    #add in tau[i]
    #objective = sum([cp.trace(cth[i] @ Mth[i]) for i in range(Nclass)])
    #objective = sum([cp.trace(mi) for mi in M])
    #objective = 0
    
    #prob = cp.Problem(cp.Minimize(objective), C)
    #sol = prob.solve(solver=cp.MOSEK, verbose = 1)


    #Start reweighted heuristic
    rank1ness = np.zeros([RwHopt.maxIter, 1])
    rank1curr = 0
    for iter in range(0, RwHopt.maxIter):
        print('   - R.H. iteration: ', iter)
        #objective = cp.Minimize(cp.trace(W.T * R))
        cost = sum([cp.trace(W[i] @ Mth[i]) for i in range(Nclass)])
        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, C)
        sol = prob.solve(solver=cp.MOSEK)
        
        for i in range(Nclass):
            val, vec = np.linalg.eig(Mth[i].value)
            [sortedval, sortedvec] = sortEigens(val, vec)
            rank1ness[iter] = sortedval[0] / np.sum(sortedval)
            W[i] = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)

        if rank1ness[iter] > RwHopt.eigThres:
            iter = iter + 1  # To fill rank1ness vector
            break


    TH_out = [[t.value for t in T] for T in TH]
    S_out  = [[t.value for t in T] for T in S]
    
    return {"cost":cost, "W": W, "M":[m.value for m in M], "mom":Mom_con, "S":S_out, "TH": TH_out, "rank1ness": rank1ness}

    # # Define index entries
    # ind_r = lambda i, d: (1 + i * D + d)
    # ind_s = lambda i, j: (1 + Ns * D + j * Ns + i)

