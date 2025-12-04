//! Clause selection strategies for the given-clause algorithm
//!
//! This module provides clause selectors that determine the order of clause
//! processing during saturation. The choice of clause selector significantly
//! impacts proof search efficiency.
//!
//! # Available Selectors
//!
//! ## Heuristic selectors:
//! - [`AgeWeightSelector`] - Classic age-weight ratio heuristic
//!
//! ## ML-based selectors (Burn framework):
//! - [`BurnGcnSelector`] / [`NdarrayGcnSelector`] - GCN-based clause selection
//! - [`BurnMlpSelector`] / [`NdarrayMlpSelector`] - MLP-based clause selection

pub mod age_weight;
pub mod burn_gcn;
pub mod burn_mlp;
pub mod clause;

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
