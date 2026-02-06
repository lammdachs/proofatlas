//! TPTP parser and FOF to CNF conversion

pub mod cnf_conversion;
pub mod fof;
pub mod tptp;

// Re-export main parsing functions and types
pub use cnf_conversion::{fof_to_cnf, fof_to_cnf_with_role, CNFConversionError};
pub use fof::{FOFFormula, Quantifier};
pub use crate::logic::ordering::{orient_all_equalities, orient_clause_equalities};
pub use tptp::{parse_tptp, parse_tptp_file, ParsedProblem};
