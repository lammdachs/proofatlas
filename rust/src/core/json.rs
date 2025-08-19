//! JSON serialization for core data structures

use super::{Atom, Clause, Literal, Proof, ProofStep, Term};
use crate::inference::{InferenceResult, InferenceRule};
use serde::{Deserialize, Serialize};

/// JSON representation of a term
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TermJson {
    Variable { name: String },
    Constant { name: String },
    Function { name: String, args: Vec<TermJson> },
}

impl From<&Term> for TermJson {
    fn from(term: &Term) -> Self {
        match term {
            Term::Variable(v) => TermJson::Variable {
                name: v.name.clone(),
            },
            Term::Constant(c) => TermJson::Constant {
                name: c.name.clone(),
            },
            Term::Function(func_sym, args) => TermJson::Function {
                name: func_sym.name.clone(),
                args: args.iter().map(|t| t.into()).collect(),
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

impl From<&Atom> for AtomJson {
    fn from(atom: &Atom) -> Self {
        AtomJson {
            predicate: atom.predicate.name.clone(),
            args: atom.args.iter().map(|t| t.into()).collect(),
        }
    }
}

/// JSON representation of a literal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteralJson {
    pub polarity: bool,
    pub atom: AtomJson,
}

impl From<&Literal> for LiteralJson {
    fn from(lit: &Literal) -> Self {
        LiteralJson {
            polarity: lit.polarity,
            atom: (&lit.atom).into(),
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

impl From<&Clause> for ClauseJson {
    fn from(clause: &Clause) -> Self {
        ClauseJson {
            id: clause.id,
            literals: clause.literals.iter().map(|l| l.into()).collect(),
        }
    }
}

/// JSON representation of an inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJson {
    pub rule: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub premises: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conclusion: Option<ClauseJson>,
}

impl From<&InferenceResult> for InferenceJson {
    fn from(inf: &InferenceResult) -> Self {
        let rule_str = match inf.rule {
            InferenceRule::Input => "Input",
            InferenceRule::GivenClauseSelection => "GivenClauseSelection",
            InferenceRule::Resolution => "Resolution",
            InferenceRule::Factoring => "Factoring",
            InferenceRule::Superposition => "Superposition",
            InferenceRule::EqualityResolution => "EqualityResolution",
            InferenceRule::EqualityFactoring => "EqualityFactoring",
            InferenceRule::Demodulation => "Demodulation",
        }
        .to_string();

        // For GivenClauseSelection, we don't need to store the conclusion
        // since it's redundant with the clause already stored
        let conclusion = if inf.rule == InferenceRule::GivenClauseSelection {
            None
        } else {
            Some((&inf.conclusion).into())
        };

        InferenceJson {
            rule: rule_str,
            premises: inf.premises.clone(),
            conclusion,
        }
    }
}

/// JSON representation of a proof step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStepJson {
    pub clause_idx: usize,
    pub inference: InferenceJson,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unprocessed_count: Option<usize>,
}

impl From<&ProofStep> for ProofStepJson {
    fn from(step: &ProofStep) -> Self {
        ProofStepJson {
            clause_idx: step.clause_idx,
            inference: (&step.inference).into(),
            processed_count: None,
            unprocessed_count: None,
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
pub enum SaturationResultJson {
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
    Timeout {
        final_clauses: Vec<ClauseJson>,
        proof_steps: Vec<ProofStepJson>,
        time_seconds: f64,
    },
}

/// Complete proof attempt data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAttemptJson {
    pub problem_file: String,
    pub initial_clauses: Vec<ClauseJson>,
    pub config: ConfigJson,
    pub result: SaturationResultJson,
    pub statistics: StatisticsJson,
}

/// Configuration used for saturation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigJson {
    pub max_clauses: usize,
    pub max_iterations: usize,
    pub timeout_seconds: f64,
    pub use_superposition: bool,
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
