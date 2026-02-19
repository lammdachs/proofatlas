% High-arity predicate: 4 arguments with cross-referencing variables.
% All four variables must bind simultaneously and consistently.
cnf(c1, axiom, ~r(X, Y, Z, W) | s(f(X, Z), g(Y, W))).
cnf(c2, axiom, r(a, b, c, d)).
cnf(c3, negated_conjecture, ~s(f(a, c), g(b, d))).
