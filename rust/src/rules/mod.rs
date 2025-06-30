//! Inference rules for array-based theorem proving
//! 
//! This module implements the inference rules for superposition calculus,
//! a complete calculus for first-order logic with equality.
//! 
//! # Available Rules
//! 
//! - **Resolution** - Resolves complementary literals between clauses
//! - **Factoring** - Merges duplicate literals within a clause
//! - **Superposition** - Handles equality by rewriting terms
//! - **Equality Resolution** - Resolves negative equality literals
//! - **Equality Factoring** - Factors positive equality literals
//! 
//! # Design
//! 
//! Each rule follows a common pattern:
//! 1. Takes an `Problem` and clause indices as input
//! 2. Performs the inference operation
//! 3. Returns an `InferenceResult` with the new clause (if any)
//! 
//! The rules respect literal selection constraints when configured.
//! 
//! # Example
//! 
//! ```rust
//! use proofatlas_rust::rules::{resolve_clauses, InferenceResult};
//! use proofatlas_rust::core::Problem;
//! 
//! let mut problem = Problem::new();
//! // ... build clauses ...
//! 
//! let results = resolve_clauses(&mut problem, 0, 1);
//! for result in results {
//!     if let Some(new_clause) = result.new_clause_idx {
//!         println!("Resolution produced clause {}", new_clause);
//!     }
//! }
//! ```

mod common;
mod resolution;
mod factoring;
mod superposition;
mod equality_resolution;
mod equality_factoring;

pub use common::{InferenceResult, has_selected_literals};
pub use resolution::resolve_clauses;
pub use factoring::factor_clause;
pub use superposition::superpose_clauses;
pub use equality_resolution::equality_resolve;
pub use equality_factoring::equality_factor;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod variable_sharing_tests;

#[cfg(test)]
mod failure_tests;