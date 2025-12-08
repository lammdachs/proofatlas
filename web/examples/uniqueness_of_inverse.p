% Uniqueness of inverse element
cnf(left_identity, axiom, mult(e,X) = X).
cnf(right_identity, axiom, mult(X,e) = X).
cnf(left_inverse, axiom, mult(inv(X),X) = e).
cnf(right_inverse, axiom, mult(X,inv(X)) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).
% Assume b is another inverse of a
cnf(b_left_inverse, axiom, mult(b,a) = e).
cnf(not_unique, negated_conjecture, b != inv(a)).