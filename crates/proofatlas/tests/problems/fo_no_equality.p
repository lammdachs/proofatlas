% First-order without equality: Socrates is mortal
cnf(c1, axiom, human(socrates)).
cnf(c2, axiom, ~human(X) | mortal(X)).
cnf(c3, negated_conjecture, ~mortal(socrates)).
