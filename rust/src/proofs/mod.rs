//! Proof tracking and state management
//!
//! This module provides types for managing proof states and proof histories.

pub mod state;
pub mod proof;

pub use self::state::ProofState;
pub use self::proof::{Proof, ProofStep, RuleApplication};