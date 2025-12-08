% Right identity from group axioms
cnf(left_identity, axiom, mult(e,X) = X).
cnf(left_inverse, axiom, mult(inv(X),X) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).
cnf(not_right_identity, negated_conjecture, mult(c,e) != c).