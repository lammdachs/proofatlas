//! Centralized clause management: interner, literal selection, and term ordering.
//!
//! The `ClauseManager` provides a unified interface for operations that require
//! coordination between the symbol interner, literal selector, and term ordering:
//! equality orientation and clause normalization.

use super::{Clause, Interner, KBOConfig, TermOrdering, KBO};
use super::literal_selection::LiteralSelector;

/// Centralized clause management combining the symbol interner, literal
/// selection strategy, and term ordering.
pub struct ClauseManager {
    /// Symbol interner for resolving and creating symbol names
    pub interner: Interner,
    /// Literal selection strategy for inference rules
    pub literal_selector: Box<dyn LiteralSelector>,
    /// Term ordering (KBO) for equality orientation and ordering constraints
    pub term_ordering: KBO,
}

impl ClauseManager {
    /// Create a new ClauseManager with the given interner and literal selector.
    /// Uses default KBO configuration for term ordering.
    pub fn new(interner: Interner, literal_selector: Box<dyn LiteralSelector>) -> Self {
        ClauseManager {
            interner,
            literal_selector,
            term_ordering: KBO::new(KBOConfig::default()),
        }
    }

    /// Orient equality literals so the larger term (by KBO) is on the left.
    ///
    /// For each equality literal `s = t`, if `t ≻ s` by the term ordering,
    /// swap arguments so the larger term comes first. This improves superposition
    /// performance by ensuring rewrites go in the right direction.
    pub fn orient_equalities(&self, clause: &mut Clause) {
        for literal in &mut clause.literals {
            if literal.is_equality(&self.interner) && literal.args.len() == 2 {
                let left = &literal.args[0];
                let right = &literal.args[1];
                match self.term_ordering.compare(left, right) {
                    TermOrdering::Less => {
                        literal.args.swap(0, 1);
                    }
                    _ => {}
                }
            }
        }
    }

}
