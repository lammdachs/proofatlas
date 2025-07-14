//! TPTP parser and FOF to CNF conversion

pub mod tptp;
pub mod fof;
pub mod cnf_conversion;
pub mod orient_equalities;

// Re-export main parsing functions
pub use tptp::{parse_tptp, parse_tptp_file};
pub use fof::{FOFFormula, Quantifier};
pub use cnf_conversion::fof_to_cnf;
pub use orient_equalities::{orient_all_equalities, orient_clause_equalities};