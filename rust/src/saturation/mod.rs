//! Saturation loop and related algorithms
//! 
//! This module implements the given-clause saturation algorithm,
//! which is the core proof search procedure for superposition provers.
//! 
//! # Key Components
//! 
//! - **Saturation Loop** - Main given-clause algorithm
//! - **Literal Selection** - Strategies for constraining inference
//! - **Subsumption** - Redundancy elimination
//! - **Unification** - Fast array-based unification
//! 
//! # Algorithm Overview
//! 
//! The given-clause algorithm maintains two sets:
//! - **Processed** - Clauses that have been selected as given clauses
//! - **Unprocessed** - Clauses waiting to be processed
//! 
//! In each iteration:
//! 1. Select a clause from unprocessed (using clause selection)
//! 2. Apply all inference rules with processed clauses
//! 3. Eliminate redundant clauses (subsumption, tautology deletion)
//! 4. Add new clauses to unprocessed
//! 
//! # Example
//! 
//! ```rust
//! use proofatlas_rust::saturation::{saturate, SaturationConfig, SelectFirstNegative};
//! use proofatlas_rust::core::{Problem, SaturationResult};
//! 
//! let mut problem = Problem::new();
//! // ... load problem ...
//! 
//! let config = SaturationConfig {
//!     max_clauses: 10000,
//!     max_clause_size: 100,
//!     max_iterations: 1000,
//!     use_backward_subsumption: true,
//!     literal_selector: Box::new(SelectFirstNegative),
//! };
//! 
//! match saturate(&mut problem, &config) {
//!     SaturationResult::Proof(proof) => {
//!         println!("Found proof with {} steps", proof.steps.len());
//!     }
//!     SaturationResult::Saturated => {
//!         println!("Problem is satisfiable");
//!     }
//!     SaturationResult::ResourceLimit => {
//!         println!("Resource limit exceeded");
//!     }
//! }
//! ```

mod r#loop;
mod literal_selection;
mod subsumption;
mod unification;

pub use r#loop::{saturate, SaturationConfig};
pub use literal_selection::{LiteralSelector, apply_literal_selection, SelectAll, SelectNegative, SelectFirstNegative, SelectLargestNegative};
pub use subsumption::SubsumptionIndex;
pub use unification::unify_nodes;

