//! Saturation-based theorem proving using the given clause algorithm

pub mod index;
pub mod profile;
pub mod rule;
pub mod state;
pub mod subsumption;
pub mod trace;

pub use profile::SaturationProfile;
pub use rule::{StateChange, EventLog};
pub use state::{LiteralSelectionStrategy, ProverConfig, ProofResult, SaturationState};
pub use trace::{extract_proof_from_events, EventLogReplayer};

use crate::fol::{CNFFormula, Interner};
use crate::prover::ProofAtlas;
use crate::selection::ClauseSelector;

/// Run saturation on a CNF formula
pub fn saturate(
    formula: CNFFormula,
    config: ProverConfig,
    clause_selector: Box<dyn ClauseSelector>,
    interner: Interner,
) -> (ProofResult, Option<SaturationProfile>, EventLog, Interner) {
    let prover = ProofAtlas::new(formula.clauses, config, clause_selector, interner);
    prover.prove()
}
