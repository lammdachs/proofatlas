//! Saturation-based theorem proving using the given clause algorithm

mod state;
pub mod subsumption;

pub use state::{LiteralSelectionStrategy, SaturationConfig, SaturationResult, SaturationState};

use crate::core::CNFFormula;
use crate::selection::ClauseSelector;

/// Run saturation on a CNF formula
pub fn saturate(
    formula: CNFFormula,
    config: SaturationConfig,
    clause_selector: Box<dyn ClauseSelector>,
) -> SaturationResult {
    let state = SaturationState::new(formula.clauses, config, clause_selector);
    state.saturate()
}

/// Run saturation on a CNF formula with a step limit
pub fn saturate_with_steps(
    formula: CNFFormula,
    mut config: SaturationConfig,
    clause_selector: Box<dyn ClauseSelector>,
    steps: Option<usize>,
) -> SaturationResult {
    config.step_limit = steps;
    let state = SaturationState::new(formula.clauses, config, clause_selector);
    state.saturate()
}
