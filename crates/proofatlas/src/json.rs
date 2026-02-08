//! JSON serialization types for proof data

use crate::logic::{Clause, Interner, Literal, Term};
use crate::state::{clause_indices, Proof, ProofStep};
use serde::{Deserialize, Serialize};

/// JSON representation of a term
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TermJson {
    Variable { name: String },
    Constant { name: String },
    Function { name: String, args: Vec<TermJson> },
}

impl TermJson {
    pub fn from_term(term: &Term, interner: &Interner) -> Self {
        match term {
            Term::Variable(v) => TermJson::Variable {
                name: v.name(interner).to_string(),
            },
            Term::Constant(c) => TermJson::Constant {
                name: c.name(interner).to_string(),
            },
            Term::Function(func_sym, args) => TermJson::Function {
                name: func_sym.name(interner).to_string(),
                args: args.iter().map(|t| TermJson::from_term(t, interner)).collect(),
            },
        }
    }
}

/// JSON representation of an atom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomJson {
    pub predicate: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<TermJson>,
}

impl AtomJson {
    pub fn from_literal(lit: &Literal, interner: &Interner) -> Self {
        AtomJson {
            predicate: lit.predicate.name(interner).to_string(),
            args: lit.args.iter().map(|t| TermJson::from_term(t, interner)).collect(),
        }
    }
}

/// JSON representation of a literal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteralJson {
    pub polarity: bool,
    pub atom: AtomJson,
}

impl LiteralJson {
    pub fn from_literal(lit: &Literal, interner: &Interner) -> Self {
        LiteralJson {
            polarity: lit.polarity,
            atom: AtomJson::from_literal(lit, interner),
        }
    }
}

/// JSON representation of a clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<usize>,
    pub literals: Vec<LiteralJson>,
}

impl ClauseJson {
    pub fn from_clause(clause: &Clause, interner: &Interner) -> Self {
        ClauseJson {
            id: clause.id,
            literals: clause.literals.iter().map(|l| LiteralJson::from_literal(l, interner)).collect(),
        }
    }
}

/// JSON representation of a clause with training metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingClauseJson {
    /// The clause literals
    pub literals: Vec<LiteralJson>,
    /// Whether this clause is in the proof (1) or not (0)
    pub label: u8,
    /// Age of the clause (derivation step, 0 for input clauses)
    pub age: usize,
    /// Role: "axiom", "hypothesis", "definition", "negated_conjecture", "derived"
    pub role: String,
    /// Parent clause IDs (empty for input clauses)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parents: Vec<usize>,
    /// Inference rule used to derive this clause (empty for input clauses)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub rule: String,
    /// Maximum term nesting depth
    pub depth: usize,
    /// Total number of symbol occurrences (predicates + functions + constants)
    pub symbol_count: usize,
    /// Number of distinct symbols
    pub distinct_symbols: usize,
    /// Total number of variable occurrences
    pub variable_count: usize,
    /// Number of distinct variables
    pub distinct_variables: usize,
}

impl TrainingClauseJson {
    /// Create from a Clause with a label and optional derivation info
    pub fn from_clause(clause: &Clause, interner: &Interner, label: bool, parents: Vec<usize>, rule: String) -> Self {
        use std::collections::HashSet;

        let role = match clause.role {
            crate::logic::ClauseRole::Axiom => "axiom",
            crate::logic::ClauseRole::Hypothesis => "hypothesis",
            crate::logic::ClauseRole::Definition => "definition",
            crate::logic::ClauseRole::NegatedConjecture => "negated_conjecture",
            crate::logic::ClauseRole::Derived => "derived",
        };

        // Compute clause statistics
        let mut depth: usize = 0;
        let mut symbol_count: usize = 0;
        let mut variable_count: usize = 0;
        let mut distinct_symbols: HashSet<u64> = HashSet::new();
        let mut distinct_variables: HashSet<u64> = HashSet::new();

        for lit in &clause.literals {
            // Count predicate as a symbol
            symbol_count += 1;
            distinct_symbols.insert(lit.predicate.id.0 as u64 | (1u64 << 32)); // tag to avoid ID collisions
            for arg in &lit.args {
                let (d, sc, vc) = term_stats(arg, &mut distinct_symbols, &mut distinct_variables);
                depth = depth.max(d);
                symbol_count += sc;
                variable_count += vc;
            }
        }

        TrainingClauseJson {
            literals: clause.literals.iter().map(|l| LiteralJson::from_literal(l, interner)).collect(),
            label: if label { 1 } else { 0 },
            age: clause.age,
            role: role.to_string(),
            parents,
            rule,
            depth,
            symbol_count,
            distinct_symbols: distinct_symbols.len(),
            variable_count,
            distinct_variables: distinct_variables.len(),
        }
    }
}

/// Recursively compute term statistics.
/// Returns (depth, symbol_count, variable_count).
fn term_stats(
    term: &Term,
    distinct_symbols: &mut std::collections::HashSet<u64>,
    distinct_variables: &mut std::collections::HashSet<u64>,
) -> (usize, usize, usize) {
    match term {
        Term::Variable(v) => {
            distinct_variables.insert(v.id.0 as u64);
            (0, 0, 1)
        }
        Term::Constant(c) => {
            distinct_symbols.insert(c.id.0 as u64 | (2u64 << 32));
            (0, 1, 0)
        }
        Term::Function(f, args) => {
            distinct_symbols.insert(f.id.0 as u64 | (3u64 << 32));
            let mut max_depth = 0usize;
            let mut sc = 1usize; // count the function symbol itself
            let mut vc = 0usize;
            for arg in args {
                let (d, s, v) = term_stats(arg, distinct_symbols, distinct_variables);
                max_depth = max_depth.max(d);
                sc += s;
                vc += v;
            }
            (max_depth + 1, sc, vc)
        }
    }
}

/// A snapshot of the selection state at one activation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStateJson {
    /// The clause that was selected (activated)
    pub selected: usize,
    /// Indices of clauses in the unprocessed set at selection time
    pub unprocessed: Vec<usize>,
    /// Indices of clauses in the processed set at selection time
    pub processed: Vec<usize>,
}

/// JSON representation of a proof trace for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceJson {
    /// Whether a proof was found
    pub proof_found: bool,
    /// Time taken in seconds
    pub time_seconds: f64,
    /// All clauses with training labels
    pub clauses: Vec<TrainingClauseJson>,
    /// Selection state snapshots at each given clause selection
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selection_states: Vec<SelectionStateJson>,
}

/// JSON representation of an inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJson {
    pub rule: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub premises: Vec<usize>,
}

/// JSON representation of a proof step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStepJson {
    pub clause_idx: usize,
    pub inference: InferenceJson,
}

impl From<&ProofStep> for ProofStepJson {
    fn from(step: &ProofStep) -> Self {
        ProofStepJson {
            clause_idx: step.clause_idx,
            inference: InferenceJson {
                rule: step.rule_name.clone(),
                premises: clause_indices(&step.premises),
            },
        }
    }
}

/// JSON representation of a proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofJson {
    pub steps: Vec<ProofStepJson>,
    pub empty_clause_idx: usize,
}

impl From<&Proof> for ProofJson {
    fn from(proof: &Proof) -> Self {
        ProofJson {
            steps: proof.steps.iter().map(|s| s.into()).collect(),
            empty_clause_idx: proof.empty_clause_idx,
        }
    }
}

/// JSON representation of a saturation result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "result")]
pub enum ProofResultJson {
    Proof {
        proof: ProofJson,
        time_seconds: f64,
    },
    Saturated {
        final_clauses: Vec<ClauseJson>,
        proof_steps: Vec<ProofStepJson>,
        time_seconds: f64,
    },
    ResourceLimit {
        reason: String,
        final_clauses: Vec<ClauseJson>,
        proof_steps: Vec<ProofStepJson>,
        time_seconds: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Interner, Literal, PredicateSymbol, Term, Constant, Variable, FunctionSymbol};

    fn make_interner_and_clause(interner: &mut Interner) -> Clause {
        // Build: p(f(g(X), a))
        // predicate p, function f with args g(X) and constant a
        let p = PredicateSymbol {
            id: interner.intern_predicate("p"),
            arity: 1,
        };
        let f = FunctionSymbol::new(interner.intern_function("f"), 2);
        let g = FunctionSymbol::new(interner.intern_function("g"), 1);
        let x = Variable::new(interner.intern_variable("X"));
        let a = Constant::new(interner.intern_constant("a"));

        let term = Term::Function(
            f,
            vec![
                Term::Function(g, vec![Term::Variable(x)]),
                Term::Constant(a),
            ],
        );

        let lit = Literal::positive(p, vec![term]);
        Clause::new(vec![lit])
    }

    #[test]
    fn test_term_stats_variable() {
        let mut distinct_symbols = std::collections::HashSet::new();
        let mut distinct_variables = std::collections::HashSet::new();
        let mut interner = Interner::new();
        let x = Variable::new(interner.intern_variable("X"));
        let term = Term::Variable(x);

        let (depth, sc, vc) = term_stats(&term, &mut distinct_symbols, &mut distinct_variables);
        assert_eq!((depth, sc, vc), (0, 0, 1));
    }

    #[test]
    fn test_term_stats_constant() {
        let mut distinct_symbols = std::collections::HashSet::new();
        let mut distinct_variables = std::collections::HashSet::new();
        let mut interner = Interner::new();
        let a = Constant::new(interner.intern_constant("a"));
        let term = Term::Constant(a);

        let (depth, sc, vc) = term_stats(&term, &mut distinct_symbols, &mut distinct_variables);
        assert_eq!((depth, sc, vc), (0, 1, 0));
    }

    #[test]
    fn test_term_stats_nested_function() {
        // f(g(X), a) → depth=2, symbol_count=3 (f, g, a), variable_count=1 (X)
        let mut distinct_symbols = std::collections::HashSet::new();
        let mut distinct_variables = std::collections::HashSet::new();
        let mut interner = Interner::new();

        let f = FunctionSymbol::new(interner.intern_function("f"), 2);
        let g = FunctionSymbol::new(interner.intern_function("g"), 1);
        let x = Variable::new(interner.intern_variable("X"));
        let a = Constant::new(interner.intern_constant("a"));

        let term = Term::Function(
            f,
            vec![
                Term::Function(g, vec![Term::Variable(x)]),
                Term::Constant(a),
            ],
        );

        let (depth, sc, vc) = term_stats(&term, &mut distinct_symbols, &mut distinct_variables);
        assert_eq!(depth, 2);
        assert_eq!(sc, 3); // f + g + a
        assert_eq!(vc, 1); // X
    }

    #[test]
    fn test_training_clause_json_statistics() {
        let mut interner = Interner::new();
        let clause = make_interner_and_clause(&mut interner);

        let tcj = TrainingClauseJson::from_clause(&clause, &interner, true, vec![], String::new());

        // p(f(g(X), a)): depth=2 (f→g→X path, predicate not counted in term depth),
        // symbol_count: p(1) + f(1) + g(1) + a(1) = 4
        // variable_count: X(1) = 1
        // distinct_symbols: p, f, g, a = 4
        // distinct_variables: X = 1
        assert_eq!(tcj.depth, 2); // term_stats(f(g(X),a)) = 2; predicate not in term depth
        assert_eq!(tcj.symbol_count, 4); // predicate p + f + g + a
        assert_eq!(tcj.variable_count, 1);
        assert_eq!(tcj.distinct_symbols, 4);
        assert_eq!(tcj.distinct_variables, 1);
        assert_eq!(tcj.label, 1);
    }

    #[test]
    fn test_distinct_symbols_tagging() {
        // Verify that predicate/constant/function IDs don't collide
        // by checking distinct_symbols count with overlapping raw IDs
        let mut interner = Interner::new();

        // Create predicate "a", constant "a", function "a" — all ID 0
        let p = PredicateSymbol {
            id: interner.intern_predicate("a"),
            arity: 1,
        };
        let c = Constant::new(interner.intern_constant("a"));
        let f = FunctionSymbol::new(interner.intern_function("a"), 1);
        let x = Variable::new(interner.intern_variable("X"));

        // Clause: a(a(X), a)  — predicate a, function a, constant a
        let term = Term::Function(f, vec![Term::Variable(x)]);
        let lit = Literal::positive(p, vec![term, Term::Constant(c)]);
        let clause = Clause::new(vec![lit]);

        let tcj = TrainingClauseJson::from_clause(&clause, &interner, false, vec![], String::new());

        // distinct_symbols should count predicate-a, constant-a, function-a as 3 separate symbols
        // because the tag bits (1<<32, 2<<32, 3<<32) differentiate them
        assert_eq!(tcj.distinct_symbols, 3);
    }
}

/// Complete proof attempt data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAttemptJson {
    pub problem_file: String,
    pub initial_clauses: Vec<ClauseJson>,
    pub config: ConfigJson,
    pub result: ProofResultJson,
    pub statistics: StatisticsJson,
}

/// Configuration used for saturation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigJson {
    pub max_clauses: usize,
    pub max_iterations: usize,
    pub timeout_seconds: f64,
    pub literal_selection: String,
}

/// Statistics from the proof attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsJson {
    pub clauses_generated: usize,
    pub clauses_processed: usize,
    pub clauses_subsumed: usize,
    pub time_elapsed_seconds: f64,
}
