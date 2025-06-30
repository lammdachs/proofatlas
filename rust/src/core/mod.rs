//! Core data structures and types for the theorem prover
//! 
//! This module provides the foundational data structures for representing
//! first-order logic formulas using an efficient array-based representation.
//! 
//! # Key Components
//! 
//! - [`Problem`] - Main data structure using CSR format for formulas
//! - [`SymbolTable`] - String interning for efficient symbol management  
//! - [`Builder`] - Builder for constructing graphs with out-of-order edges
//! - [`Proof`] - Proof tracking and step recording
//! 
//! # Example
//! 
//! ```rust
//! use proofatlas_rust::core::{Problem, Builder, NodeType};
//! 
//! let mut problem = Problem::new();
//! let mut builder = Builder::new(&mut problem);
//! 
//! // Build a clause: P(a) ∨ ¬Q(b)
//! let clause = builder.add_node(NodeType::Clause, "", 0, 2);
//! let lit1 = builder.add_node(NodeType::Literal, "", 1, 1);
//! let lit2 = builder.add_node(NodeType::Literal, "", -1, 1);
//! ```

mod problem;
mod symbol_table;
pub mod builder;
mod proof;
pub mod parser_convert;
pub mod ordering;

pub use problem::{Problem, NodeType, ClauseType, ArraySubstitution, CapacityError};
pub use symbol_table::SymbolTable;
pub use builder::Builder;
pub use proof::{ProofStep, Proof, InferenceRule, SaturationResult};