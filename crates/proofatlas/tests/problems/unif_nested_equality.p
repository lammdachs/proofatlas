% Deep nested equality rewriting with variable unification.
% Superposition must match f(g(X)) inside h(f(g(a)),b) and propagate X=a.
cnf(c1, axiom, f(g(X)) = k(X)).
cnf(c2, axiom, p(h(f(g(a)), b))).
cnf(c3, negated_conjecture, ~p(h(k(a), b))).
