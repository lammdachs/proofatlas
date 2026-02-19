% Pure equality: injectivity of f implies a = b when f(a) = f(b)
cnf(c1, axiom, f(X) != f(Y) | X = Y).
cnf(c2, axiom, f(a) = f(b)).
cnf(c3, negated_conjecture, a != b).
