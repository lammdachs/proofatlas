//! TPTP parser module

pub mod tptp;
pub mod prescan;

// Re-export main parsing functions
pub use tptp::{parse_file, parse_string};
pub use prescan::prescan_file;