% Satisfiable: binding chain with binary function — partial structural alignment.
% Unifying p(X,f(X,a)) with ~p(f(Y,a),Y): first args give X = f(Y,a),
% then second args require f(f(Y,a),a) = Y — the extra constant a makes
% the terms look plausibly matchable before the cycle surfaces.
cnf(c1, axiom, p(X, f(X, a))).
cnf(c2, axiom, ~p(f(Y, a), Y)).
