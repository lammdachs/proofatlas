//! Saturation-based theorem proving using the given clause algorithm

mod state;
mod subsumption;
mod simplification;

pub use state::{SaturationState, SaturationConfig, SaturationResult, LiteralSelectionStrategy};

use crate::core::CNFFormula;

/// Run saturation on a CNF formula
pub fn saturate(formula: CNFFormula, config: SaturationConfig) -> SaturationResult {
    let state = SaturationState::new(formula.clauses, config);
    state.saturate()
}