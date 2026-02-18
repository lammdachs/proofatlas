% Satisfiable: no single resolution step fails — occurs check only in derived clause.
% c1 + c2 resolve on p: ~p(X,Y)|q(f(Y),X) with p(f(Z),Z) gives q(f(Z),f(Z)).
% Then q(f(Z),f(Z)) with ~q(W,f(W)): W = f(Z) from first arg, then
% f(W) = f(f(Z)) must equal f(Z) — forces Z = f(Z), an occurs check
% that only appears after composing two valid inference steps.
cnf(c1, axiom, ~p(X, Y) | q(f(Y), X)).
cnf(c2, axiom, p(f(Z), Z)).
cnf(c3, axiom, ~q(W, f(W))).
