% Satisfiable: same outer function f, same arity, swapped variable and g(.).
% Unifying p(f(X,g(X))) with ~p(f(g(Y),Y)): first args give X = g(Y),
% then second args require g(g(Y)) = Y — occurs check two function
% applications deep. Identical outer structure masks the failure.
cnf(c1, axiom, p(f(X, g(X)))).
cnf(c2, axiom, ~p(f(g(Y), Y))).
