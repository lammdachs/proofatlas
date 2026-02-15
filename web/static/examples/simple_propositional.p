% Simple propositional logic example
cnf(p_implies_q, axiom, ~p | q).
cnf(q_implies_r, axiom, ~q | r).
cnf(p_true, axiom, p).
cnf(not_r, negated_conjecture, ~r).