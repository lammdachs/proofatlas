//! Generating inference rule implementations.
//!
//! Each module merges the algorithm from `inference/` with the rule adapter
//! from `saturation/rule.rs`.

pub mod common;
pub mod resolution;
pub mod superposition;
pub mod factoring;
pub mod equality_resolution;
pub mod equality_factoring;

pub use common::{
    unify_atoms, unify_atoms_scoped, collect_literals_except, collect_scoped_literals_except,
    remove_duplicate_literals, is_ordered_greater,
};
pub use resolution::{resolution, ResolutionRule};
pub use superposition::{superposition, SuperpositionRule};
pub use factoring::{factoring, FactoringRule};
pub use equality_resolution::{equality_resolution, EqualityResolutionRule};
pub use equality_factoring::{equality_factoring, EqualityFactoringRule};
