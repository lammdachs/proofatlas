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
//! given-clause algorithm. ONNX-based ML selection is used exclusively.
//!
//! - [`OnnxClauseSelector`] - ML-based clause selection using ONNX model

pub mod clause;
pub mod literal;

pub use clause::{ClauseSelector, OnnxClauseSelector};
pub use literal::{LiteralSelector, SelectAll, SelectLargestNegative, SelectMaxWeight};
