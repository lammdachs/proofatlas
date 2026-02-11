//! JSON serialization types for proof data

use crate::logic::{Clause, Interner, Literal, Term};
use crate::state::{clause_indices, ProofStep};
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

