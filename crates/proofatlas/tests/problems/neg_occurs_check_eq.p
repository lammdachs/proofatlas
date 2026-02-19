% Satisfiable: f(X) = X has no solution (occurs check on X = f(X)).
% If the prover's equality resolution ignored occurs check, it would
% unsoundly delete the negative equality ~(f(X) = X) and derive empty clause.
cnf(c1, axiom, p(a)).
cnf(c2, axiom, ~p(X) | f(X) != X).
