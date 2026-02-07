//! Prover configuration types.

use std::time::Duration;

/// Configuration for the saturation loop
#[derive(Debug, Clone)]
pub struct ProverConfig {
    pub max_clauses: usize,
    pub max_iterations: usize,
    pub max_clause_size: usize,
    pub timeout: Duration,
    pub literal_selection: LiteralSelectionStrategy,
    /// Memory limit for clause storage in MB
    pub memory_limit_mb: Option<usize>,
    /// Enable structured profiling (zero overhead when false)
    pub enable_profiling: bool,
}

/// Literal selection strategies (numbers match Vampire's --selection option)
///
/// From Hoder et al. "Selecting the selection" (2016):
/// - Sel0: Select all literals
/// - Sel20: Select all maximal literals
/// - Sel21: Select unique maximal, else max-weight negative, else all maximal
/// - Sel22: Select max-weight negative literal, else all maximal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralSelectionStrategy {
    /// Selection 0: Select all literals (no selection)
    Sel0,
    /// Selection 20: Select all maximal literals
    Sel20,
    /// Selection 21: Unique maximal, else max-weight negative, else all maximal
    Sel21,
    /// Selection 22: Max-weight negative literal, else all maximal
    Sel22,
}

impl Default for ProverConfig {
    fn default() -> Self {
        ProverConfig {
            max_clauses: 0,      // 0 means no limit
            max_iterations: 0,   // 0 means no limit
            max_clause_size: 100,
            timeout: Duration::from_secs(60),
            literal_selection: LiteralSelectionStrategy::Sel21,
            memory_limit_mb: None,
            enable_profiling: false,
        }
    }
}
