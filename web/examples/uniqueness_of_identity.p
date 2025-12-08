% Uniqueness of identity element
cnf(left_identity, axiom, mult(e,X) = X).
cnf(right_identity, axiom, mult(X,e) = X).
cnf(left_inverse, axiom, mult(inv(X),X) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).
% Assume e1 is another identity
cnf(e1_left_identity, axiom, mult(e1,X) = X).
cnf(not_unique, negated_conjecture, e != e1).