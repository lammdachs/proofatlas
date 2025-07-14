//! Saturation-based theorem proving using the given clause algorithm

mod state;
mod subsumption;
mod simplification;
mod custom;

pub use state::{SaturationState, SaturationConfig, SaturationResult, LiteralSelectionStrategy};
pub use custom::custom_saturate_with_trace;

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