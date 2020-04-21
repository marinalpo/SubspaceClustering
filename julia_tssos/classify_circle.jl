using DynamicPolynomials
using SparseArrays
using TSSOS

#Initialize points
cx = 2
cy = 3
Nsample = 6
Ntotal = 2*Nsample
theta_1 = rand(Nsample)
# x1 = [cos(th) for th in theta_1]
# y1 = [sin(th) for th in theta_1]
X1 = [(cos(th), sin(th)) for th in theta_1]

theta_2 = rand(Nsample)
# x2 = [cos(th) + cx for th in theta_2]
# y2 = [sin(th) + cy for th in theta_2]
X2 = [(cos(th), sin(th)) for th in theta_1]
# X = [x1 y1; x2 y2]
X = [X1; X2]

eps_test = 0.05

@polyvar x[1:2]
@polyvar th1[1:2]
@polyvar th2[1:2]

@polyvar s1[1:Ntotal]
@polyvar s2[1:Ntotal]

var = [th1; th2; s1; s2]

V1 = (x[1] - th1[1])^2 + (x[2] - th1[2])^2 - 1
V2 = (x[1] - th2[1])^2 + (x[2] - th2[2])^2 - 1

V1_sub = s1 .* [subs(V1, x=>Xi) for Xi in X]
V2_sub = s2 .* [subs(V2, x=>Xi) for Xi in X]

con_classify_1 = [V1_sub + s1.*eps_test; s1.*eps_test - V1_sub]
con_classify_2 = [V2_sub + s2.*eps_test; s2.*eps_test - V2_sub]

con_assign = [s1[i] + s2[i] - 1 for i in 1:Ntotal]
con_bin = [s1.^2-s1; s2.^2-s2]

# con_geom = []
con_sym = [th1[1] >= th2[1]]

con_eq = [con_assign; con_bin]
con_ineq = [con_sym; con_classify_1; con_classify_2]


f = 0

pop = [f; con_ineq; con_eq]
d = 2
# opt, sol, data = blockcpop_first(pop, var, d, numeq = length(con_eq), method="clique", solve=false, solution=false)

opt, sol, data = blockcpop_first(pop, var, d, numeq = length(con_eq), QUIET=false, method="clique", solve=true, solution=true)

# opt,sol,data =  blockcpop_higher!(data,QUIET=true, method="clique", solve=false, solution=false)
# opt,sol,data =  blockcpop_higher!(data,QUIET=false, method="clique", solve=true, solution=true)
