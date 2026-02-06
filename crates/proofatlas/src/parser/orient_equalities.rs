//! Equality orientation for preprocessing
//!
//! Orients equality literals so that the larger term (according to KBO)
//! is on the left side. This improves superposition performance.

use crate::fol::{Clause, Interner, KBOConfig, TermOrdering, KBO};

/// Orient all equality literals in a clause
pub fn orient_clause_equalities(clause: &mut Clause, interner: &Interner) {
    let kbo = KBO::new(KBOConfig::default());

    for literal in &mut clause.literals {
        if literal.is_equality(interner) && literal.args.len() == 2 {
            let left = &literal.args[0];
            let right = &literal.args[1];

            // Compare terms using KBO
            match kbo.compare(left, right) {
                TermOrdering::Less => {
                    // Right is larger, swap the arguments
                    literal.args.swap(0, 1);
                }
                _ => {
                    // Keep as is (Greater, Equal, or Incomparable)
                    // For Incomparable, we keep the original order
                }
            }
        }
    }
}

/// Orient equalities in all clauses
pub fn orient_all_equalities(clauses: &mut [Clause], interner: &Interner) {
    for clause in clauses {
        orient_clause_equalities(clause, interner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Constant, Literal, PredicateSymbol, Term};

    #[test]
    fn test_orient_simple_equality() {
        let mut interner = Interner::new();
        let a_id = interner.intern_constant("a");
        let b_id = interner.intern_constant("b");
        let eq_id = interner.intern_predicate("=");

        let a = Term::Constant(Constant::new(a_id));
        let b = Term::Constant(Constant::new(b_id));

        let eq_pred = PredicateSymbol::new(eq_id, 2);

        // Create clause: a = b
        let mut clause = Clause::new(vec![Literal::positive(eq_pred, vec![a.clone(), b.clone()])]);

        orient_clause_equalities(&mut clause, &interner);

        // Should be reoriented to b = a (since b > a alphabetically)
        assert_eq!(clause.literals[0].args[0], b);
        assert_eq!(clause.literals[0].args[1], a);
    }

    #[test]
    fn test_keep_correct_orientation() {
        let mut interner = Interner::new();
        // Intern "a" first so it gets lower ID, then "c" gets higher ID
        // This means c > a in KBO precedence (higher ID = higher precedence)
        let a_id = interner.intern_constant("a");
        let c_id = interner.intern_constant("c");
        let eq_id = interner.intern_predicate("=");

        let c = Term::Constant(Constant::new(c_id));
        let a = Term::Constant(Constant::new(a_id));

        let eq_pred = PredicateSymbol::new(eq_id, 2);

        // Create clause: c = a (already correctly oriented since c > a)
        let mut clause = Clause::new(vec![Literal::positive(eq_pred, vec![c.clone(), a.clone()])]);

        orient_clause_equalities(&mut clause, &interner);

        // Should remain c = a (c has higher ID, so c > a in precedence)
        assert_eq!(clause.literals[0].args[0], c);
        assert_eq!(clause.literals[0].args[1], a);
    }
}
