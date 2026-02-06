//! Re-exports for backward compatibility.
//! Canonical locations: crate::state, crate::config, crate::profile, crate::trace,
//! crate::index, crate::simplifying, crate::generating, crate::prover

pub mod rule;
pub mod state;
pub mod subsumption;

pub use crate::index;
pub use crate::profile::SaturationProfile;
pub use crate::state::{StateChange, EventLog, ProofResult, SaturationState};
pub use crate::config::{LiteralSelectionStrategy, ProverConfig};
pub use crate::trace::{extract_proof_from_events, EventLogReplayer};
pub use crate::prover::saturate;
