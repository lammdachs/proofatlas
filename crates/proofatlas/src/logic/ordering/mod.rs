pub mod kbo;
pub mod orient_equalities;

#[cfg(test)]
mod proptest_tests;

pub use kbo::{KBOConfig, Ordering, KBO};
pub use orient_equalities::{orient_all_equalities, orient_clause_equalities};
