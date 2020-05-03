[r, res] = mosekopt('read(circle_classify.task.gz)');
%[r, res] = mosekopt('read(circle_classify0.task.gz)');

At = full(res.prob.a)';

model = struct;
%[model.A, model.b, model.c, model.K] ...
%    = convert_mosek2sedumi(res.prob);
% 
%mosek2sedumi
%[F, obj] = sedumi2yalmip(model.A',model.b,model.c,model.K);
%[model2, ~] = export(F, obj, sdpsettings('solver','sedumi', 'removeequalities', 2));