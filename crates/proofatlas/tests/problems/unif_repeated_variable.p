% Repeated variable: X appears 3 times in a literal, requiring consistent binding.
% c1+c2 unifies p(X,X,f(X)) with p(g(Y),g(Y),Z) giving X=g(Y), Z=f(g(Y)).
cnf(c1, axiom, p(X, X, f(X))).
cnf(c2, axiom, ~p(g(Y), g(Y), Z) | q(Y, Z)).
cnf(c3, negated_conjecture, ~q(a, f(g(a)))).
