//! Wire protocol types for scoring server communication
//!
//! Defines the request/response enums and length-prefixed framing used
//! between RemoteSelector (client) and ScoringServer (server) over Unix
//! domain sockets.

use crate::logic::{Clause, Interner};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

/// Serializable representation of an Interner's symbol tables.
///
/// Since `Interner` doesn't implement Serde, we transfer the four
/// string arenas as `Vec<String>`. Reconstruction via `to_interner()`
/// produces identical ID assignments because `StringArena` assigns
/// sequential IDs starting from 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternedSymbols {
    pub variables: Vec<String>,
    pub constants: Vec<String>,
    pub functions: Vec<String>,
    pub predicates: Vec<String>,
}

impl InternedSymbols {
    /// Extract all symbol names from an `Interner`.
    pub fn from_interner(interner: &Interner) -> Self {
        let variables = (0..interner.variable_count())
            .map(|i| {
                interner
                    .resolve_variable(crate::logic::VariableId(i as u32))
                    .to_string()
            })
            .collect();
        let constants = (0..interner.constant_count())
            .map(|i| {
                interner
                    .resolve_constant(crate::logic::ConstantId(i as u32))
                    .to_string()
            })
            .collect();
        let functions = (0..interner.function_count())
            .map(|i| {
                interner
                    .resolve_function(crate::logic::FunctionId(i as u32))
                    .to_string()
            })
            .collect();
        let predicates = (0..interner.predicate_count())
            .map(|i| {
                interner
                    .resolve_predicate(crate::logic::PredicateId(i as u32))
                    .to_string()
            })
            .collect();
        Self {
            variables,
            constants,
            functions,
            predicates,
        }
    }

    /// Reconstruct an `Interner` from the stored symbol names.
    ///
    /// Interns each string in order, producing identical ID assignments
    /// to the original interner.
    pub fn to_interner(&self) -> Interner {
        let mut interner = Interner::new();
        for name in &self.variables {
            interner.intern_variable(name);
        }
        for name in &self.constants {
            interner.intern_constant(name);
        }
        for name in &self.functions {
            interner.intern_function(name);
        }
        for name in &self.predicates {
            interner.intern_predicate(name);
        }
        interner
    }
}

/// Request sent from worker to scoring server.
#[derive(Debug, Serialize, Deserialize)]
pub enum ScoringRequest {
    /// Initialize the connection with the problem's symbol table.
    Init { interner: InternedSymbols },
    /// Request scores for unprocessed clauses.
    ///
    /// `uncached` contains (clause_index, Clause) pairs the server hasn't seen.
    /// `unprocessed_indices` and `processed_indices` are the full U and P sets.
    Score {
        uncached: Vec<(usize, Clause)>,
        unprocessed_indices: Vec<usize>,
        processed_indices: Vec<usize>,
    },
    /// Clear all caches for a new problem.
    Reset,
    /// Shut down this connection.
    Shutdown,
}

/// Response sent from scoring server to worker.
#[derive(Debug, Serialize, Deserialize)]
pub enum ScoringResponse {
    /// Init succeeded.
    InitOk,
    /// Scores for the requested unprocessed clauses (same order as `unprocessed_indices`).
    Scores(Vec<f32>),
    /// Reset succeeded.
    ResetOk,
    /// An error occurred.
    Error(String),
}

/// Write a length-prefixed bincode message to a stream.
///
/// Format: 4-byte little-endian length + bincode payload.
pub fn write_message<W: Write, T: Serialize>(writer: &mut W, msg: &T) -> io::Result<()> {
    let payload =
        bincode::serialize(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}

/// Read a length-prefixed bincode message from a stream.
///
/// Format: 4-byte little-endian length + bincode payload.
pub fn read_message<R: Read, T: for<'de> Deserialize<'de>>(reader: &mut R) -> io::Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;

    bincode::deserialize(&payload).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, Literal, PredicateSymbol, Term};

    #[test]
    fn test_request_roundtrip_init() {
        let mut interner = Interner::new();
        interner.intern_variable("X");
        interner.intern_constant("a");
        interner.intern_function("f");
        interner.intern_predicate("p");

        let symbols = InternedSymbols::from_interner(&interner);
        let req = ScoringRequest::Init {
            interner: symbols.clone(),
        };

        let mut buf = Vec::new();
        write_message(&mut buf, &req).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let decoded: ScoringRequest = read_message(&mut cursor).unwrap();

        match decoded {
            ScoringRequest::Init { interner: syms } => {
                assert_eq!(syms.variables, vec!["X"]);
                assert_eq!(syms.constants, vec!["a"]);
                assert_eq!(syms.functions, vec!["f"]);
                assert_eq!(syms.predicates, vec!["p"]);
            }
            _ => panic!("Expected Init"),
        }
    }

    #[test]
    fn test_request_roundtrip_score() {
        let mut interner = Interner::new();
        let p = PredicateSymbol {
            id: interner.intern_predicate("P"),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            id: interner.intern_constant("a"),
        });
        let clause = Clause::new(vec![Literal::positive(p, vec![a])]);

        let req = ScoringRequest::Score {
            uncached: vec![(0, clause)],
            unprocessed_indices: vec![0, 1],
            processed_indices: vec![2],
        };

        let mut buf = Vec::new();
        write_message(&mut buf, &req).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let decoded: ScoringRequest = read_message(&mut cursor).unwrap();

        match decoded {
            ScoringRequest::Score {
                uncached,
                unprocessed_indices,
                processed_indices,
            } => {
                assert_eq!(uncached.len(), 1);
                assert_eq!(uncached[0].0, 0);
                assert_eq!(unprocessed_indices, vec![0, 1]);
                assert_eq!(processed_indices, vec![2]);
            }
            _ => panic!("Expected Score"),
        }
    }

    #[test]
    fn test_request_roundtrip_reset_shutdown() {
        for req in [ScoringRequest::Reset, ScoringRequest::Shutdown] {
            let mut buf = Vec::new();
            write_message(&mut buf, &req).unwrap();
            let mut cursor = std::io::Cursor::new(buf);
            let _: ScoringRequest = read_message(&mut cursor).unwrap();
        }
    }

    #[test]
    fn test_response_roundtrip() {
        let responses = vec![
            ScoringResponse::InitOk,
            ScoringResponse::Scores(vec![0.1, 0.5, 0.9]),
            ScoringResponse::ResetOk,
            ScoringResponse::Error("test error".into()),
        ];

        for resp in responses {
            let mut buf = Vec::new();
            write_message(&mut buf, &resp).unwrap();
            let mut cursor = std::io::Cursor::new(buf);
            let decoded: ScoringResponse = read_message(&mut cursor).unwrap();

            match (&resp, &decoded) {
                (ScoringResponse::InitOk, ScoringResponse::InitOk) => {}
                (ScoringResponse::Scores(a), ScoringResponse::Scores(b)) => assert_eq!(a, b),
                (ScoringResponse::ResetOk, ScoringResponse::ResetOk) => {}
                (ScoringResponse::Error(a), ScoringResponse::Error(b)) => assert_eq!(a, b),
                _ => panic!("Mismatch"),
            }
        }
    }

    #[test]
    fn test_interned_symbols_roundtrip() {
        let mut interner = Interner::new();
        let x = interner.intern_variable("X");
        let y = interner.intern_variable("Y");
        let a = interner.intern_constant("a");
        let b = interner.intern_constant("b");
        let f = interner.intern_function("f");
        let p = interner.intern_predicate("p");
        let q = interner.intern_predicate("q");

        let symbols = InternedSymbols::from_interner(&interner);
        let reconstructed = symbols.to_interner();

        // IDs should be identical
        assert_eq!(reconstructed.resolve_variable(x), "X");
        assert_eq!(reconstructed.resolve_variable(y), "Y");
        assert_eq!(reconstructed.resolve_constant(a), "a");
        assert_eq!(reconstructed.resolve_constant(b), "b");
        assert_eq!(reconstructed.resolve_function(f), "f");
        assert_eq!(reconstructed.resolve_predicate(p), "p");
        assert_eq!(reconstructed.resolve_predicate(q), "q");

        // Counts should match
        assert_eq!(reconstructed.variable_count(), interner.variable_count());
        assert_eq!(reconstructed.constant_count(), interner.constant_count());
        assert_eq!(reconstructed.function_count(), interner.function_count());
        assert_eq!(
            reconstructed.predicate_count(),
            interner.predicate_count()
        );
    }
}
