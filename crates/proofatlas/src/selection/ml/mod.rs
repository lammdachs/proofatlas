//! ML model implementations for clause selection

#[cfg(feature = "ml")]
pub mod gcn;
pub mod graph;
#[cfg(feature = "ml")]
pub mod sentence;
