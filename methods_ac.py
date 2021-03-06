# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:11:54 2020

@author: Jared
"""
from typing import Dict, List, Any, Union

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
    Future: extend to semialgebraic clustering: f(x; theta) <= 0
    
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

        # polynomial models under consideration
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
        self.mult += [mult]
        self.Nclass += mult

    def moment_SDP(self):
        # roughly equivalent to AC_CVXPY, generate the SDP to classify data in Xp
        # or maybe generate model without data Xp, and then insert classification constraints. That sounds better.

        """
        Constraints in problem

        con_one:        [0,0...0] entry of moment matrix is 1
        con_mom:        Parameter (theta) entries obey moment structure
        con_geom:       Models obey geometric constraints g(theta) <= 0, h(theta) == 0

        con_bin:        binary classification: s^2 = s
        con_assign:     all datapoints are assigned to one and only one model class

        con_classify:   Models classify in-class data within epsilon

        C = con_one + con_assign + con_mom + con_bin + con_classify + con_geom

        """
        Mth = []
        C = []
        TH = []
        count = 0

        # Iterate through all classes
        for i in range(len(self.model)):
            cvx_out = self.model[i].generate_moment(self.mult[i])
            Mth += [cvx_out["Mth"]]
            TH += cvx_out["TH"]
            C += cvx_out["C"]

        cvx_moment = {"Mth": Mth, "C": C, "TH": TH}
        return cvx_moment

    def classify_SDP(self, Xp, eps, cvx_moment):
        """Constraints for model classification of data in Xp, as corrupted by bounded noise (eps)

        Parameters
        ----------
        Xp: numpy ndarray
            Data to be classified
        eps: double
            Noise bound in classification (assume scalar for now)
            If eps < 0, this is a validation problem.
        cvx_moment:
            dictionary containing moment information for problem
        """
        Mth = cvx_moment["Mth"]
        C_moment = cvx_moment["C"]
        C_classify = []
        M = []
        S = []

        Invalidation = (eps < 0)
        if Invalidation:
            eps = cp.Variable(1)
            C_classify += [eps >= 0]


        for i in range(len(self.model)):
            cvx_out = self.model[i].generate_classify(Xp, Invalidation, eps, Mth[i])
            M += cvx_out["M"]
            S += cvx_out["S"]
            C_classify += cvx_out["C"]



        # missing con_assign
        # each datapoint is uniquely assigned to a model
        assignment = np.sum(np.array(S), axis=0)
        con_assign = [si == 1 for si in assignment]

        C = C_moment + C_classify + con_assign

        S = np.array(S)

        cvx_classify = {"C": C, "M": M, "Mth": Mth, "S": S, "TH": cvx_moment["TH"], "epsilon": eps, "Invalidation": Invalidation}
        return cvx_classify

    def generate_SDP(self, Xp, eps):
        cvx_moment = self.moment_SDP()
        cvx_classify = self.classify_SDP(Xp, eps, cvx_moment)

        return cvx_classify

    def solve_SDP(self, cvx_classify, RwHopt):
        """Solve semialgebraic clustering through reweighted heuristic """

        s_penalize = RwHopt.s_penalize
        combine_theta = RwHopt.combine_theta
        Mth_structured = cvx_classify["Mth"]
        M = cvx_classify["M"]
        C = cvx_classify["C"]
        S = cvx_classify["S"]
        TH = cvx_classify["TH"]

        #start by labeling a single point (get rid of this when outliers enter)
        C += [M[0][0][-1,-1] == 1]

        Mth = [item for sublist in Mth_structured for item in sublist]
        Ms = [item for sublist in M for item in sublist]
        # Reweighted heuristic for rank-1 magic
        Nclass = len(Mth)

        # https: // www.cvxpy.org / tutorial / advanced / index.html  # disciplined-parametrized-programming
        blocksize = [mth.shape[0] for mth in Mth]
        Ws_val = np.ones(S.shape)/np.sqrt(Nclass)


        if combine_theta:
            combined_size = sum([b-1 for b in blocksize]) + 1

            #form the combined moment matrix and sew together constraints
            Mall = cp.Variable((combined_size, combined_size), PSD=True)

            con_combine = [Mall[-1, -1] == 1]

            lo = 0
            hi = 0
            for mth in Mth:
                curr_blocksize = mth.shape[0]
                hi = lo + curr_blocksize-1

                #equality constraints
                curr_cons = []
                curr_cons += [mth[:-1, :-1] == Mall[lo:hi, lo:hi]]
                curr_cons += [mth[:-1, -1] == Mall[lo:hi, -1]]

                lo = lo + curr_blocksize - 1

                con_combine += curr_cons

            C += con_combine
            #penalize the rank of the combined moment matrix
            W = cp.Parameter((combined_size, combined_size), symmetric=True)
            cost_th = cp.trace(W @ Mall)

            W_val = np.eye(combined_size) + RwHopt.delta * np.random.randn(combined_size)
            W_val = (W_val + W_val.T)*0.5

            rank1ness = np.zeros([RwHopt.maxIter, 1])
        else:
            W = [cp.Parameter((bi, bi), symmetric=True) for bi in blocksize]
            cost_th = sum([cp.trace(W[i] @ Mth[i]) for i in range(Nclass)])

            W_val = [np.eye(bi) + RwHopt.delta * np.random.randn(bi) for bi in blocksize]
            W_val = [(Wv + Wv.T) * 0.5 for Wv in W_val]

            rank1ness = np.zeros([RwHopt.maxIter, sum(self.mult)])




        #Reweighted L1-norm on all s sets, encourage points to be uniquely classified

        Ws = cp.Parameter(tuple(S.shape))
        cost_s = sum([sum([Ws[i, j]*S[i, j] for i in range(Ws.shape[0])]) for j in range(Ws.shape[1])])

        if s_penalize:
            blocksizeM = [m.shape[0] for m in Ms]
            WM_val = [np.eye(bi) + RwHopt.delta * np.random.randn(bi) for bi in blocksizeM]
            WM_val = [(Wv + Wv.T) for Wv in WM_val]
            WM = [cp.Parameter((bi, bi), symmetric=True) for bi in blocksizeM]


        #additional encouragement by reweighting s-augmented moment matrices
        if s_penalize:
            cost_s_matrix = sum([cp.trace(WM[i] @ Ms[i]) for i in range(len(Ms))])
            cost = cost_th + RwHopt.s_rankweight *( cost_s +cost_s_matrix)
        else:
            cost = cost_th + RwHopt.s_rankweight * cost_s

        objective = cp.Minimize(cost)

        iter = 0
        infeasible = False
        # for iter in range(0, RwHopt.maxIter):
        while iter < RwHopt.maxIter:

            if combine_theta:
                W.value = W_val
            else:
                for i in range(Nclass):
                    W[i].value = W_val[i]

            Ws.value = Ws_val

            if s_penalize:
                for i in range(len(Ms)):
                    WM[i].value = WM_val[i]
            prob = cp.Problem(objective, C)
            sol = prob.solve(solver=cp.MOSEK, verbose=RwHopt.verbose)

            if prob.status == 'infeasible':
                infeasible = True
                break


            #Reweight the parameters in moment matrices
            if combine_theta:
                val, vec = np.linalg.eig(Mall.value)
                [sortedval, sortedvec] = sortEigens(val, vec)
                rank1ness[iter] = sortedval[0] / np.sum(sortedval)
                W_val = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)
                W_val = W_val / np.linalg.norm(W_val, 'fro')

            else:
                for i in range(Nclass):
                    val, vec = np.linalg.eig(Mth[i].value)
                    [sortedval, sortedvec] = sortEigens(val, vec)
                    rank1ness[iter, i] = sortedval[0] / np.sum(sortedval)
                    W_val[i] = np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))), sortedvec.T)
                    W_val[i] = W_val[i] / np.linalg.norm(W_val[i], 'fro')

            #Reweight the indicator variables
            #indicator variables are always positive, so don't need to worry about absolute values
            S_out = np.array([[t.value for t in T] for T in S])
            Ws_val = 1/(S_out + 1e-4)
            Ws_val = Ws_val/np.sqrt(np.sum(Ws_val**2, axis=0))

            #Reweight the labeled moment matrices
            if s_penalize:
                for i in range(len(Ms)):
                    val, vec = np.linalg.eig(Ms[i].value)
                    [sortedval, sortedvec] = sortEigens(val, vec)
                    # rank1ness[iter, i] = sortedval[0] / np.sum(sortedval)
                    WM_val[i] = 0.1 * np.matmul(np.matmul(sortedvec, np.diag(1 / (sortedval + np.exp(-5)))),
                                                sortedvec.T)

            minrank = min(rank1ness[iter, :])
            # print('   - R.H. iteration: ', iter)
            print(f'\t - R.H. iteration: {iter} \t rank1ness: {minrank:5.3}')
            if minrank > RwHopt.eigThres:
                # iter = RwHopt.maxIter
                break

            iter = iter + 1  # To fill rank1ness vector

        TH_out = [[t.value for t in T] for T in TH]

        M_out = [[m.value for m in mc] for mc in M]
        Mth_out = [m.value for m in Mth]

        out_dict = {"cost": cost, "W": W, "S": S_out, "TH": TH_out, "infeasible": infeasible,
                    "rank1ness": rank1ness, "M": M_out, "Mth": Mth_out, "rank1ness_min": minrank}

        if s_penalize:
            out_dict["WM"] = WM

        if combine_theta:
            out_dict["Mall"] = Mall.value

        return out_dict

    def solve_invalidation(self, cvx_classify, verbose):
        """Invalidation SDP: return "eps_min" as minimum epsilon that may have generated data"""
        Mth_structured = cvx_classify["Mth"]
        M = cvx_classify["M"]
        C = cvx_classify["C"]
        S = cvx_classify["S"]
        TH = cvx_classify["TH"]

        eps = cvx_classify["epsilon"]

        # start by labeling a single point (get rid of this when outliers enter)
        C += [M[0][0][-1, -1] == 1]

        Mth = [item for sublist in Mth_structured for item in sublist]

        objective = cp.Minimize(eps)
        prob = cp.Problem(objective, C)
        sol = prob.solve(solver=cp.MOSEK, verbose=verbose)

        if prob.status == 'infeasible':
            infeasible = True
        else:
            infeasible = False
            eps_out = eps.value

        TH_out = [[t.value for t in T] for T in TH]

        M_out = [[m.value for m in mc] for mc in M]
        Mth_out = [m.value for m in Mth]

        out_dict = { "TH": TH_out, "infeasible": infeasible, "M": M_out, "Mth": Mth_out, "eps_min": eps_out}

        return out_dict

    def solve_SDP_rstar(self, cvx_classify, verbose):
        # solve the SDP through the r* norm method

        # take variables that have been formulated from the SDP
        Mth_structured = cvx_classify["Mth"]
        M = cvx_classify["M"]
        C = cvx_classify["C"]
        S = cvx_classify["S"]
        TH = cvx_classify["TH"]

        Mth = [item for sublist in Mth_structured for item in sublist]
        # Reweighted heuristic for rank-1 magic
        Nclass = len(Mth)

        # https://arxiv.org/pdf/1612.03186.pdf page 13
        #
        r = 0.1
        blocksize = [mth.shape[0] for mth in Mth]
        W1 = [cp.Variable((bi, bi), PSD=True) for bi in blocksize]
        W2 = [cp.Variable((bi, bi), symmetric=True) for bi in blocksize]
        k = cp.Variable(Nclass)

        # parameters in the r* model
        cost = cp.sum([cp.trace(W2[i]) + k[i] for i in range(Nclass)])
        con_r_star = []
        for i in range(len(Mth)):
            bl = blocksize[i]
            I = np.eye(bl)
            con_trace = (cp.trace(W1[i]) == (bl - r) * k[i])
            con_psd = (W1[i] >> 0)
            B = cp.bmat([[k[i] * I - W1[i], Mth[i]], [Mth[i].T, W2[i]]])
            con_lmi = (B >> 0)
            con_r_star += [con_trace, con_psd, con_lmi]

        C += con_r_star

        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, C)
        sol = prob.solve(solver=cp.MOSEK, verbose=verbose)

        rank1ness = [0] * Nclass

        for i in range(Nclass):
            val, vec = np.linalg.eig(Mth[i].value)
            [sortedval, sortedvec] = sortEigens(val, vec)
            rank1ness[i] = sortedval[0] / np.sum(sortedval)

        TH_out = [[t.value for t in T] for T in TH]
        S_out = np.array([[t.value for t in T] for T in S])
        M_out = [[m.value for m in mc] for mc in M]
        Mth_out = [m.value for m in Mth]

        return {"cost": cost, "S": S_out, "TH": TH_out,
                "rank1ness": rank1ness, "M": M_out, "Mth": Mth_out}

    def run_SDP(self, Xp, eps, RwHopt):
        """Main routine, take everything together. All model synthesis routines + running the solver"""
        cvx_classify = self.generate_SDP(Xp, eps)
        if cvx_classify["Invalidation"]:
            cvx_result = self.solve_invalidation(cvx_classify, RwHopt.verbose)
        else:
            cvx_result = self.solve_SDP(cvx_classify, RwHopt)

        return cvx_result


class Model:
    """A model that might plausibly generate observed data (example circle, subspace, curve)"""

    def __init__(self, x, th):
        self.x = x
        self.th = th

        # depends on data and parameters
        self.eq = []  # f(x, theta) == 0
        self.ineq = []  # f(x, theta) <= 0

        self.moment = None

    # add functions to the semialgebraic set description
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

    def moment_constraint(self, eq, ineq):
        """Form moment constraints given semialgebraic set description"""
        all_poly = eq + ineq

        moment = ACmomentConstraint(all_poly, [self.x, self.th])
        return moment

    def generate_moment(self, mult):

        self.moment = self.moment_constraint(self.eq, self.ineq)

        sizes = [len(self.eq), len(self.ineq)]

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

        # moment matrix for each corner
        # turn this into vectorized moments later
        M_size = len(mom["supp"])
        Mth = [cp.Variable((M_size, M_size), symmetric=True) for i in range(mult)]

        # [0,0,...0] entry of moment matrix is 1
        Nth = len(self.th)
        # ind_1 = mom["monom_classify"].index([0] * Nth)
        ind_1 = -1
        con_one = [(Mth[i][ind_1, ind_1] == 1) for i in range(mult)]

        # matrix obeys moment structure
        con_mom = []
        for monom, ind_agree in mom["cons"].items():
            for iagree in range(len(ind_agree) - 1):
                iprev = ind_agree[iagree]
                inext = ind_agree[iagree + 1]
                con_mom += [(Mth[i][iprev] == Mth[i][inext]) for i in range(mult)]

        # break symmetry between model classes in program with arbitrary constraint
        con_symmetry = [(Mth[count][0, 0] >= Mth[count + 1][0, 0]) for count in range(mult - 1)]

        C = con_one + con_mom + con_symmetry

        # extract parameters theta
        TH_ind = [mom["cons"][tuple(i)][0] for i in np.eye(Nth).tolist()]
        TH = [[Mth[count][thi] for thi in TH_ind] for count in range(mult)]

        cvx_out = {"Mth": Mth, "C": C, "TH": TH}

        return cvx_out

    def generate_classify(self, Xp, Invalidation, eps, Mth):
        """Run the classify generation routine"""

        sizes = [len(self.eq), len(self.ineq)]
        cvx_out = self.classify_SDP(sizes, self.moment, Xp, Invalidation, eps, Mth)

        return cvx_out

    # def geom_SDP(self, sizes, mom, Mth):
    #     """Generate geometric constraints among the model parameters"""
    #     pass

    def classify_SDP(self, sizes, mom, Xp, Invalidation, eps, Mth):
        """Classify data in Xp, eps given current model Mth (with multiplicity)"""

        [D, Np] = Xp.shape
        mult = len(Mth)
        # len(mom["supp"])
        # simple implementation first
        # then trim out unnecessary variables

        con_invalidation = []

        con_classify_eq = []
        con_classify_ineq = []

        con_geom_eq = []
        con_geom_ineq = []

        con_bin = []
        con_mom_agree = []
        Nth = len(mom["supp"][0])

        Ns = len(mom["supp"])
        mult = len(Mth)
        S = [None] * mult  # output labels
        M = []
        for im in range(mult):
            Mi = []
            monom_poly = mom["monom_poly"]
            ind_1 = mom["supp"].index([0] * Nth)

            # TODO: refine this so only monomials in the active polynomial are active, and the other monomials/lifts
            #       are not present in the classifier psd blocks
            # monom_idx = [[mom["supp"].index(list(m)) for m in mp] for mp in monom_poly]

            for k in range(sizes[0] + sizes[1]):
                if mom["geom"][k]:
                    #geometric constraint (data-independent)

                    #coefficients are actually invariant under datapoint, but hack it here
                    coeff_geom = np.array(mom["fb"][k](*[0]*D))

                    #what moments are involved in the current polynomial?
                    #get indices from lookup table
                    moment_ind = [mom["cons"][mp][0] for mp in monom_poly[k]]
                    #and read those values from the moment matrix
                    moment_value = np.array([Mth[im].__getitem__(mic) for mic in moment_ind])

                    f_curr = moment_value @ coeff_geom

                    if k < sizes[0]:
                        # equality constraint
                        con_geom_eq += [f_curr == 0]
                    else:
                        # inequality constraint <= 0
                        con_geom_ineq += [f_curr <= 0]
                else:
                    #classification constraint (data-dependent)
                    mi_curr = [mom["supp"].index(list(m)) for m in monom_poly[k]]
                    for ip in range(Np):

                        if Invalidation:
                            ind_s = -2
                            m_size = Ns + 2
                        else:
                            ind_s = -1
                            m_size = Ns + 1

                        m = cp.Variable((m_size, m_size), PSD=True)
                        con_mom_agree += [(m[:ind_s, :ind_s] == Mth[im])]
                        Mi.append(m)

                        con_bin += [(m[ind_1, ind_s] == m[ind_s, ind_s])]

                        s_curr = m[ind_1, ind_s]
                        if ip == 0:
                            S[im] = [s_curr]
                        else:
                            S[im] += [s_curr]

                        coeff_curr = mom["fb"][k](*Xp[:, ip])
                        rs_curr = m[mi_curr, -1]

                        f_curr = rs_curr @ coeff_curr

                        if Invalidation:
                            s_eps = m[ind_s, -1]
                            con_invalidation += [m[ind_1, -1] == eps]
                        else:
                            s_eps = s_curr * eps

                        if k < sizes[0]:
                            # equality constraint
                            con_classify_eq += [-s_eps <= f_curr, f_curr <= s_eps]
                        else:
                            # inequality constraint <= 0
                            con_classify_ineq += [f_curr <= s_eps]

            M.append(Mi)

        C = con_geom_eq + con_geom_ineq + con_classify_eq + con_classify_ineq + con_bin + con_mom_agree + con_invalidation

        # output: C constraints, M classification-blocks, S binary labels
        cvx_out = {"C": C, "M": M, "S": S, "Invalidation": Invalidation}

        return cvx_out


def AC_CVXPY(Xp, eps, var, P, mult, RwHopt):  # , , delta
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

    # This system will have cliques [1 parameter-monomials all data]
    # Will split up data later

    # Define dimension and number of points
    # x = var[0]
    # th = var[1]
    Mom_con = [ACmomentConstraint(p, var) for p in P]

    [D, Np] = Xp.shape
    Nx = len(var[0])
    Nth = len(var[1])

    Nclass = sum(mult)

    # number of monomials
    Nm = [len(m["monom_all"]) for m in Mom_con]
    Ns = [len(m["supp"]) for m in Mom_con]

    # Nm = 1 + Ns * (Np + D)  # Size of the overall matrix

    M = []  # moment matrix
    Mth = []  # moment matrix restricted to parameter entries (theta)
    C = []  # constraints

    con_assign_pre = np.zeros([Np, 1])
    con_classify = []
    con_one = []
    con_mom = []
    con_bin = []
    con_assign = []
    count = 0

    # R = [None]*Nclass
    S = [None] * Nclass
    TH_monom = [None] * Nclass
    TH = [None] * Nclass
    # S = cp.Variable([Np, Nclass])

    class_sum = []
    for ic in range(len(P)):

        # Moment Constraints
        mom = Mom_con[ic]

        TH_ind = [mom["cons"][tuple(i)][0] for i in np.eye(Nth).tolist()]

        for im in range(mult[ic]):
            m = cp.Variable([Ns[ic] + Np, Ns[ic] + Np], PSD=True)
            M.append(m)
            Mth.append(m[:Ns[ic], :Ns[ic]])

            # "corner" of moment matrix is 1
            # ind_1 = len(mom["monom_all"])-1
            ind_1 = mom["monom_all"].index(tuple([0] * Nth))
            con_one.append(M[count][ind_1, ind_1] == 1)
            # r_curr = M[count][ind_1, :Nm[ic]]
            # R[count] = r_curr

            TH_monom[count] = Mth[count][ind_1, :]
            TH[count] = [Mth[count][thi] for thi in TH_ind]

            # entries of moment matrix agree with each other
            # like (xy)(xy) == (x^2) (y^2)
            for monom, ind_agree in mom["cons"].items():
                for iagree in range(len(ind_agree) - 1):
                    iprev = ind_agree[iagree]
                    inext = ind_agree[iagree + 1]
                    con_mom.append(M[count][iprev] == M[count][inext])

            # each point is properly classified (function evaluation)
            for ip in range(Np):
                i = ip + Ns[ic]

                coeff_curr = mom["fb"][0](*Xp[:, ip])

                # f_curr = 0
                rs_curr = M[count][i, :Nm[ic]]
                s_curr = M[count][i, ind_1]

                # add in the new variable
                # con_S += [S[ip, count]  == s_curr]

                f_curr = coeff_curr @ rs_curr

                con_classify.append(-eps * s_curr <= f_curr)
                con_classify.append(f_curr <= eps * s_curr)

                if ip == 0:
                    S[count] = [s_curr]
                else:
                    S[count] += [s_curr]

                if count == 0:
                    class_sum += [s_curr]
                else:
                    class_sum[ip] += s_curr

                # all classifications are binary
                con_bin += [M[count][i, ind_1] == M[count][i, i]]

            # prepare for new moment matrix
            count += 1

    # unique classification of points

    for ip in range(Np):
        con_assign += [cp.sum(class_sum[ip]) == 1]

    # con_assign = [class_sum == 1]

    # C = con_one + con_assign + con_mom + con_bin

    # still need to add in extra constraints, instituting relations among/between classes
    # like if a set of planes are perpendicular, or to normalize a projective curve
    C = con_one + con_assign + con_mom + con_bin + con_classify

    # develop the objective
    W = [np.eye(mth.shape[0]) + 1e-1 * np.random.randn(mth.shape[0]) for mth in Mth]
    objective = 0
    # for i = 1:Nclass
    #    objective = cth[i] @ Mth[i]

    # add in tau[i]
    # objective = sum([cp.trace(cth[i] @ Mth[i]) for i in range(Nclass)])
    # objective = sum([cp.trace(mi) for mi in M])
    # objective = 0

    # prob = cp.Problem(cp.Minimize(objective), C)
    # sol = prob.solve(solver=cp.MOSEK, verbose = 1)

    # Start reweighted heuristic
    rank1ness = np.zeros([RwHopt.maxIter, 1])
    rank1curr = 0
    for iter in range(0, RwHopt.maxIter):
        print('   - R.H. iteration: ', iter)
        # objective = cp.Minimize(cp.trace(W.T * R))
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
    S_out = [[t.value for t in T] for T in S]

    return {"cost": cost, "W": W, "M": [m.value for m in M], "mom": Mom_con, "S": S_out, "TH": TH_out,
            "rank1ness": rank1ness}

    # # Define index entries
    # ind_r = lambda i, d: (1 + i * D + d)
    # ind_s = lambda i, j: (1 + Ns * D + j * Ns + i)
