//! Selection strategies for literal and clause selection
//!
//! This module provides trait-based abstractions for:
//! - Literal selection within clauses (for inference rules)
//! - Clause selection from the unprocessed set (for saturation)

pub mod literal;
pub mod clause;

pub use literal::{LiteralSelector, NoSelection, SelectNegative, SelectMaxWeight};
pub use clause::{ClauseSelector, SizeBasedSelector, AgeBasedSelector, AgeWeightRatioSelector};