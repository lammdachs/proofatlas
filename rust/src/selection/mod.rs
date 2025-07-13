//! Selection strategies for literal and clause selection
//!
//! This module provides trait-based abstractions for:
//! - Literal selection within clauses (for inference rules)
//! - Clause selection from the unprocessed set (for saturation)

pub mod literal;
pub mod clause;
pub mod max_weight;

pub use literal::{LiteralSelector, NoSelection};
pub use clause::{ClauseSelector, FIFOSelector, SizeBasedSelector, AgeBasedSelector, AgeWeightRatioSelector};
pub use max_weight::SelectMaxWeight;