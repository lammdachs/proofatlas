//! Simplifying inference rule implementations.
//!
//! Each module merges the algorithm from `inference/` with the rule adapter
//! from `saturation/rule.rs`.

pub mod tautology;
pub mod subsumption;
pub mod demodulation;

pub use tautology::TautologyRule;
pub use subsumption::SubsumptionRule;
pub use demodulation::DemodulationRule;
