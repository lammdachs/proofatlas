% Shared variables across literals: X, Y, Z each appear in two positions.
% Requires consistent multi-variable binding through 3 resolution steps.
cnf(c1, axiom, ~p(X, Y) | ~q(Y, Z) | r(X, Z)).
cnf(c2, axiom, p(a, f(b))).
cnf(c3, axiom, q(f(b), c)).
cnf(c4, negated_conjecture, ~r(a, c)).
