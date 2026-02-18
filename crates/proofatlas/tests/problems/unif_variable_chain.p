% Deep variable chain: binding X forces Y, Y forces Z, Z forces W.
% Resolution at each step propagates bindings through 4 levels.
cnf(c1, axiom, p(X, f(X))).
cnf(c2, axiom, ~p(g(Y), Z) | q(Y, Z)).
cnf(c3, axiom, ~q(h(W), V) | r(W, V)).
cnf(c4, negated_conjecture, ~r(a, f(g(h(a))))).
