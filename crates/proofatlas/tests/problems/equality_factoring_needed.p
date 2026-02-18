% Needs equality factoring: two positive equalities
cnf(c1, axiom, f(X) = a | f(X) = b).
cnf(c2, negated_conjecture, f(c) != a).
cnf(c3, negated_conjecture, f(c) != b).
