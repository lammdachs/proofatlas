% Horn clauses (at most one positive literal)
cnf(c1, axiom, p(a)).
cnf(c2, axiom, ~p(X) | q(X)).
cnf(c3, axiom, ~q(X) | r(X)).
cnf(c4, negated_conjecture, ~r(a)).
