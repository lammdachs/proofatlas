% Satisfiable: cross-binding occurs check requires two substitution steps.
% Unifying p(X,f(X)) with ~p(f(Y),Y): first args give X = f(Y),
% then second args require f(f(Y)) = Y — cycle only visible after
% composing the binding from the first pair.
cnf(c1, axiom, p(X, f(X))).
cnf(c2, axiom, ~p(f(Y), Y)).
