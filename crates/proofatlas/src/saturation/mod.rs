//! Saturation-based theorem proving using the given clause algorithm

pub mod index;
pub mod profile;
pub mod rule;
mod state;
pub mod subsumption;
pub mod trace;

pub use profile::SaturationProfile;
pub use rule::{ProofStateChange, SaturationEventLog};
pub use state::{LiteralSelectionStrategy, SaturationConfig, SaturationResult, SaturationState};
pub use trace::{extract_proof_from_events, EventLogReplayer};

use crate::fol::{CNFFormula, Interner};
use crate::selection::ClauseSelector;

/// Run saturation on a CNF formula
pub fn saturate(
    formula: CNFFormula,
    config: SaturationConfig,
    clause_selector: Box<dyn ClauseSelector>,
    interner: Interner,
) -> (SaturationResult, Option<SaturationProfile>, SaturationEventLog, Interner) {
    let state = SaturationState::new(formula.clauses, config, clause_selector, interner);
    state.saturate()
}
