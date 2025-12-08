% Inverse of identity is identity
cnf(left_identity, axiom, mult(e,X) = X).
cnf(left_inverse, axiom, mult(inv(X),X) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).
cnf(not_inv_e_is_e, negated_conjecture, inv(e) != e).