//! TPTP parser and FOF to CNF conversion

pub mod tptp;
pub mod fof;
pub mod cnf_conversion;

// Re-export main parsing functions
pub use tptp::{parse_tptp, parse_tptp_file};
pub use fof::{FOFFormula, Quantifier};
pub use cnf_conversion::fof_to_cnf;