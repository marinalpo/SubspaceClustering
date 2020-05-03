using DynamicPolynomials
using SparseArrays
using TSSOS
using Plots

#Initialize points
cx = 2
cy = 3
Nsample = 10
Ntotal = 2*Nsample

eps_true = 0.05
R = 1

theta = rand(Ntotal) * 2 * pi
eps = rand(Ntotal) * 2 * eps_true .- eps_true
#r = 1 .+ eps
r_noisy = sqrt.(R^2 .+ eps)

#maybe this should be sqrt(eps) instead?

X = [r_noisy.*cos.(theta) r_noisy.*sin.(theta)]

X[1:Nsample, :] = X[1:Nsample, :] .+ [cx cy]

plot(X[:,1], X[:, 2], seriestype=:scatter)

# theta_1 = rand(Nsample)*2*pi
# x1 = [cos(th) for th in theta_1]
# y1 = [sin(th) for th in theta_1]
#X1 = [(cos(th), sin(th)) for th in theta_1]

# theta_2 = rand(Nsample)*2*pi
# x2 = [cos(th) + cx for th in theta_2]
# y2 = [sin(th) + cy for th in theta_2]
# X2 = [(cos(th) + cx, sin(th) + cy) for th in theta_1]
# X = [x1 y1; x2 y2]
# X = [X1; X2]
eps_test = 0.07

@polyvar x[1:2]
@polyvar th1[1:2]
@polyvar th2[1:2]

@polyvar s1[1:Ntotal]
@polyvar s2[1:Ntotal]

var = [th1; th2; s1; s2]

V1 = (x[1] - th1[1])^2 + (x[2] - th1[2])^2 - R^2
V2 = (x[1] - th2[1])^2 + (x[2] - th2[2])^2 - R^2

V1_sub = s1 .* [subs(V1, x=>Xi) for Xi in eachrow(X)]
V2_sub = s2 .* [subs(V2, x=>Xi) for Xi in eachrow(X)]

con_classify_1 = [V1_sub + s1.*eps_test; s1.*eps_test - V1_sub]
con_classify_2 = [V2_sub + s2.*eps_test; s2.*eps_test - V2_sub]

con_assign = [s1[i] + s2[i] - 1 for i in 1:Ntotal]
con_bin = [s1.^2-s1; s2.^2-s2]

# break symmetry?
con_sym = th1[1] - th2[1]
#con_sym = []


con_eq = [con_assign; con_bin]
con_ineq = [con_sym; con_classify_1; con_classify_2]


f = 0

pop = [f; con_ineq; con_eq]
d = 2
opt, sol, data = blockcpop_first(pop, var, d, numeq = length(con_eq), QUIET=false,
    method="chordal",chor_alg="greedy", solve=true, solution=true, reducebasis=true)
# chor_alg="greedy"

# opt,sol,data   =  blockcpop_higher!(data,QUIET=false, method="chordal", solve=true, solution=true)
