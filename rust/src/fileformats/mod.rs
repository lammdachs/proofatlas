//! File format handlers

// High-level API modules
pub mod tptp;

// Parser implementation modules
pub mod tptp_parser;
pub mod fof;
pub mod prescan;

// Re-export commonly used items
pub use tptp_parser::{parse_file, parse_string};

// Future formats can be added here
// pub mod smtlib;
// pub mod dimacs;