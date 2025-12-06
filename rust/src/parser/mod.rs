//! TPTP parser and FOF to CNF conversion

pub mod cnf_conversion;
pub mod fof;
pub mod orient_equalities;
pub mod tptp;

// Re-export main parsing functions
pub use cnf_conversion::fof_to_cnf;
pub use fof::{FOFFormula, Quantifier};
pub use orient_equalities::{orient_all_equalities, orient_clause_equalities};
pub use tptp::{parse_tptp, parse_tptp_file, parse_tptp_with_includes};
