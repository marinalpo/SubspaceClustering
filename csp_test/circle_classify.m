rng(400, 'twister')
th0_true = [0,0];
th1_true = [2,3];

Nsample = 20;
Ntotal = 2*Nsample;

theta = rand(Ntotal, 1)*2*pi;
eps_true = 0.1;
eps_data = rand(Ntotal, 1)*eps_true*2 - eps_true;
X = [cos(theta), sin(theta)].*(1+eps_data);

X(1:Nsample,:) = X(1:Nsample,:) + th0_true;
X(Nsample+1:end,:) = X(Nsample+1:end,:) + th1_true;

th0 = sdpvar(2, 1);
th1 = sdpvar(2, 1);
s  = sdpvar(Ntotal, 2);
%s = sdpvar(Ntotal, 1); %only two classes, reduce # variables for testing
var = {th0, th1, s};

Vcirc = @(x,th)  sum((x - th).^2)-1;
eps_test = 0.15;
%constraints
con_assign = (sum(s, 2) == 1);
% con_assign = [];
con_bin = (s.^2 == s);
con_bound = [sum(th0.^2)<=100, sum(th1.^2)<=100];
% con_bound = [th0 <= 10; -10 <= th0; th1 <= 10; -10 <= th1];
con_classify = [];
for i = 1:Ntotal
    xcurr = X(i,:)';
    s0 = s(i, 1);
%    s0 = s(i);
    v0 = Vcirc(xcurr, th0);
    con_classify = [con_classify; -s0*eps_test <= s0*v0; s0*v0 <= s0*eps_test];
    
     s1 = s(i, 2);
%    s1 = 1-s(i);
    v1 = Vcirc(xcurr,th1);
    con_classify = [con_classify; -s1*eps_test <= s1*v1; s1*v1 <= s1*eps_test];
end

con_sym = (th0(1)<=th1(1));

%constraints = [con_bound; con_bin; con_sym; con_assign];
constraints = [con_sym; con_classify; con_bound; con_bin; con_assign];
%obj = 0;
obj = sum(sum(s_out .* (1+0.4*rand(size(s_out)))));
d = 2;
opts = sdpsettings('solver', 'SparsePOP', 'sparsepop.relaxOrder', d, ...
   'debug',1);
sol = optimize(constraints, obj, opts);

th0_out = value(th0);
th1_out = value(th1);
s_out = value(s);
s0_out = s_out(:, 1);
s1_out = s_out(:, 2);
%plot output
figure(1)
clf
hold on
%class0 = find(s_out > 0.5);
class0 = find(s0_out > 0.5);
class1 = find(s1_out > 0.5);
%class1 = find(s_out <= 0.5);
scatter(X(:, 1), X(:, 2), '.k')
scatter(X(class0, 1), X(class0, 2), 'or')

scatter(th0_out(1), th0_out(2), 200, 'xr')

scatter(X(class1, 1), X(class1, 2), 'ob')

scatter(th1_out(1), th1_out(2), 200,'xb')
hold off