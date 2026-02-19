% Same variable name X in both clauses — tests scoped unification.
% The X in c1 and X in c2 are different variables; scoping must keep them apart.
cnf(c1, axiom, ~p(X) | q(X, f(X))).
cnf(c2, axiom, ~q(a, X) | r(X)).
cnf(c3, axiom, p(a)).
cnf(c4, negated_conjecture, ~r(f(a))).
