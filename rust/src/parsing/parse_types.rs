//! Minimal types used during TPTP parsing
//! These are temporary structures that get immediately converted to array representation

/// Temporary term representation for parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseTerm {
    Variable(String),
    Constant(String),
    Function {
        name: String,
        args: Vec<ParseTerm>,
    },
}

/// Temporary predicate representation for parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsePredicate {
    pub name: String,
    pub args: Vec<ParseTerm>,
}

/// Temporary literal representation for parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLiteral {
    pub predicate: ParsePredicate,
    pub polarity: bool,
}

/// Temporary clause representation for parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseClause {
    pub literals: Vec<ParseLiteral>,
}

/// Temporary problem representation for parsing
#[derive(Debug, Clone)]
pub struct ParseProblem {
    pub clauses: Vec<ParseClause>,
    pub conjecture_indices: Vec<usize>,
}

impl ParseProblem {
    pub fn new() -> Self {
        ParseProblem {
            clauses: Vec::new(),
            conjecture_indices: Vec::new(),
        }
    }
    
    pub fn add_clause(&mut self, clause: ParseClause, is_conjecture: bool) {
        if is_conjecture {
            self.conjecture_indices.push(self.clauses.len());
        }
        self.clauses.push(clause);
    }
}