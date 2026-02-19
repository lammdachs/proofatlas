% Chain of implications
cnf(c1, axiom, p(a)).
cnf(c2, axiom, ~p(X) | q(X)).
cnf(c3, axiom, ~q(X) | r(X)).
cnf(c4, axiom, ~r(X) | s(X)).
cnf(c5, negated_conjecture, ~s(a)).
