//! Saturation-based theorem proving using the given clause algorithm

mod state;
mod simplification;
pub mod subsumption;

pub use state::{SaturationState, SaturationConfig, SaturationResult, LiteralSelectionStrategy};

use crate::core::CNFFormula;

/// Run saturation on a CNF formula
pub fn saturate(formula: CNFFormula, config: SaturationConfig) -> SaturationResult {
    let state = SaturationState::new(formula.clauses, config);
    state.saturate()
}

/// Run saturation on a CNF formula with a step limit
pub fn saturate_with_steps(formula: CNFFormula, mut config: SaturationConfig, steps: Option<usize>) -> SaturationResult {
    config.step_limit = steps;
    let state = SaturationState::new(formula.clauses, config);
    state.saturate()
}