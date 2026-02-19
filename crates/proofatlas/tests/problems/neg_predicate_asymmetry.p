% Satisfiable: predicates are not symmetric — p(a, b) does not imply p(b, a).
% Unlike equality, predicate argument order matters.
cnf(c1, axiom, p(a, b)).
cnf(c2, axiom, ~p(b, a)).
