//! Selection strategies for literal and clause selection
//!
//! This module provides trait-based abstractions for:
//! - Literal selection within clauses (for inference rules)
//! - Clause selection from the unprocessed set (for saturation)
//!
//! # Literal Selection
//!
//! Literal selection constrains which literals in a clause can participate
//! in inference rules. More restrictive selection reduces the search space
//! but may sacrifice completeness.
//!
//! Available strategies:
//! - [`SelectAll`] - No restriction, all literals eligible (complete)
//! - [`SelectMaxWeight`] - Only literals with maximum symbol count (incomplete)
//! - [`SelectLargestNegative`] - Largest negative literal, or all if none (complete)
//!
//! # Clause Selection
//!
//! Clause selection determines the order of clause processing in the
//! given-clause algorithm. Available strategies:
//!
//! ## Heuristic selectors:
//! - [`AgeWeightSelector`] - Classic age-weight ratio heuristic
//!
//! ## ML-based selectors (Burn framework):
//! - [`burn_gcn::BurnGcnSelector`] - GCN-based clause selection
//! - [`burn_mlp::BurnMlpSelector`] - MLP-based clause selection

pub mod age_weight;
pub mod burn_gcn;
pub mod burn_mlp;
pub mod clause;
pub mod literal;

pub use age_weight::AgeWeightSelector;
pub use burn_gcn::{
    create_ndarray_gcn_selector, load_ndarray_gcn_selector, BurnGcnSelector, GcnModel,
    NdarrayGcnSelector,
};
pub use burn_mlp::{
    create_ndarray_mlp_selector, load_ndarray_mlp_selector, BurnMlpSelector, MlpModel,
    NdarrayMlpSelector,
};
pub use clause::ClauseSelector;
pub use literal::{LiteralSelector, SelectAll, SelectLargestNegative, SelectMaxWeight};
