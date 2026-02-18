% Satisfiable: diamond conflict — first argument pair succeeds, second conflicts.
% Unifying p(f(X,Y),f(Y,X)) with ~p(f(a,b),f(a,b)): first pair gives
% X = a, Y = b. Second pair requires Y = a, X = b — contradicts the
% committed bindings. The symmetric structure masks the contradiction.
cnf(c1, axiom, p(f(X, Y), f(Y, X))).
cnf(c2, axiom, ~p(f(a, b), f(a, b))).
