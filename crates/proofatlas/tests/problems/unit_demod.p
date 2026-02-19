% Unit demodulation: rewriting chains
cnf(c1, axiom, f(a) = b).
cnf(c2, axiom, g(b) = c).
cnf(c3, axiom, p(g(f(a)))).
cnf(c4, negated_conjecture, ~p(c)).
