% Inverse of inverse is the original element
cnf(left_identity, axiom, mult(e,X) = X).
cnf(right_identity, axiom, mult(X,e) = X).
cnf(left_inverse, axiom, mult(inv(X),X) = e).
cnf(right_inverse, axiom, mult(X,inv(X)) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).
cnf(not_inv_inv, negated_conjecture, inv(inv(a)) != a).