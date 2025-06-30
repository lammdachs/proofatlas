//! Literal selection strategies for inference rules

use crate::core::Problem;

/// Trait for literal selection strategies
pub trait LiteralSelector: Send + Sync {
    /// Select literals from a clause for inference
    fn select_literals(&self, problem: &Problem, clause_idx: usize) -> Vec<usize>;
}

/// Select all negative literals
pub struct SelectNegative;

impl LiteralSelector for SelectNegative {
    fn select_literals(&self, problem: &Problem, clause_idx: usize) -> Vec<usize> {
        let literals = problem.clause_literals(clause_idx);
        literals.into_iter()
            .enumerate()
            .filter_map(|(idx, lit)| {
                if problem.node_polarities[lit] == -1 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Select all positive literals
pub struct SelectPositive;

impl LiteralSelector for SelectPositive {
    fn select_literals(&self, problem: &Problem, clause_idx: usize) -> Vec<usize> {
        let literals = problem.clause_literals(clause_idx);
        literals.into_iter()
            .enumerate()
            .filter_map(|(idx, lit)| {
                if problem.node_polarities[lit] == 1 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Select first literal
pub struct SelectFirst;

impl LiteralSelector for SelectFirst {
    fn select_literals(&self, problem: &Problem, clause_idx: usize) -> Vec<usize> {
        let literals = problem.clause_literals(clause_idx);
        if literals.is_empty() {
            vec![]
        } else {
            vec![0]
        }
    }
}

/// Select first negative literal, or first if no negative
pub struct SelectFirstNegative;

impl LiteralSelector for SelectFirstNegative {
    fn select_literals(&self, problem: &Problem, clause_idx: usize) -> Vec<usize> {
        let literals = problem.clause_literals(clause_idx);
        
        // Find first negative
        for (idx, &lit) in literals.iter().enumerate() {
            if problem.node_polarities[lit] == -1 {
                return vec![idx];
            }
        }
        
        // No negative found, select first
        if literals.is_empty() {
            vec![]
        } else {
            vec![0]
        }
    }
}

/// Select all literals (no restriction)
pub struct SelectAll;

impl LiteralSelector for SelectAll {
    fn select_literals(&self, problem: &Problem, clause_idx: usize) -> Vec<usize> {
        let literals = problem.clause_literals(clause_idx);
        (0..literals.len()).collect()
    }
}

/// Select the negative literal with the most arguments
pub struct SelectLargestNegative;

impl LiteralSelector for SelectLargestNegative {
    fn select_literals(&self, problem: &Problem, clause_idx: usize) -> Vec<usize> {
        let literals = problem.clause_literals(clause_idx);
        
        let mut best_idx = None;
        let mut best_arity = 0;
        
        // Find negative literal with most arguments
        for (idx, &lit) in literals.iter().enumerate() {
            if problem.node_polarities[lit] == -1 {
                // Get the predicate for this literal
                let children = problem.node_children(lit);
                if !children.is_empty() {
                    let pred = children[0];
                    let arity = problem.node_arities[pred];
                    
                    if arity > best_arity {
                        best_arity = arity;
                        best_idx = Some(idx);
                    }
                }
            }
        }
        
        // Return the best negative literal, or first literal if no negative
        if let Some(idx) = best_idx {
            vec![idx]
        } else if literals.is_empty() {
            vec![]
        } else {
            vec![0]
        }
    }
}

/// Apply literal selection to a clause
pub fn apply_literal_selection(
    problem: &mut Problem,
    clause_idx: usize,
    selector: &dyn LiteralSelector,
) {
    let selected = selector.select_literals(problem, clause_idx);
    let literals = problem.clause_literals(clause_idx);
    
    // Mark selected literals
    for (idx, &lit) in literals.iter().enumerate() {
        if selected.contains(&idx) {
            problem.node_selected[lit] = true;
        } else {
            problem.node_selected[lit] = false;
        }
    }
}

// Tests temporarily disabled - need to create literal_selection_tests.rs
// #[cfg(test)]
// #[path = "literal_selection_tests.rs"]
// mod tests;