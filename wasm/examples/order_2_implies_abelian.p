% Prove that a group in which every element has order 2 is abelian (commutative)
% In such a group, x * x = e for all x, which means x = inv(x) for all x

% Group axioms
cnf(left_identity, axiom, mult(e,X) = X).
cnf(right_identity, axiom, mult(X,e) = X).
cnf(left_inverse, axiom, mult(inv(X),X) = e).
cnf(right_inverse, axiom, mult(X,inv(X)) = e).
cnf(associativity, axiom, mult(mult(X,Y),Z) = mult(X,mult(Y,Z))).

% Every element has order 2 (x * x = e for all x)
cnf(order_2, axiom, mult(X,X) = e).

% Prove commutativity: a * b = b * a
cnf(not_commutative, negated_conjecture, mult(a,b) != mult(b,a)).