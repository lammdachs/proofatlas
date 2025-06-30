//! Subsumption checking and indexing

use crate::core::{Problem, ArraySubstitution};
use std::collections::HashMap;

/// Index for efficient subsumption checking
pub struct SubsumptionIndex {
    // For now, a simple implementation
    // TODO: Implement discrimination tree for efficiency
    clauses: Vec<usize>,
}

impl SubsumptionIndex {
    pub fn new() -> Self {
        SubsumptionIndex {
            clauses: Vec::new(),
        }
    }
    
    /// Insert a clause into the index
    pub fn insert(&mut self, clause_idx: usize, _problem: &Problem) {
        self.clauses.push(clause_idx);
    }
    
    /// Remove a clause from the index
    pub fn remove(&mut self, clause_idx: usize) {
        self.clauses.retain(|&idx| idx != clause_idx);
    }
    
    /// Find a clause that subsumes the given clause
    pub fn find_subsuming(&self, clause_idx: usize, problem: &Problem) -> Option<usize> {
        for &existing_idx in &self.clauses {
            if existing_idx != clause_idx && subsumes(problem, existing_idx, clause_idx) {
                return Some(existing_idx);
            }
        }
        None
    }
    
    /// Find all clauses subsumed by the given clause
    pub fn find_subsumed_by(&self, clause_idx: usize, problem: &Problem) -> Vec<usize> {
        let mut subsumed = Vec::new();
        for &existing_idx in &self.clauses {
            if existing_idx != clause_idx && subsumes(problem, clause_idx, existing_idx) {
                subsumed.push(existing_idx);
            }
        }
        subsumed
    }
}

/// Check if clause1 subsumes clause2
/// A clause C subsumes clause D if there exists a substitution σ such that Cσ ⊆ D
pub fn subsumes(problem: &Problem, clause1_idx: usize, clause2_idx: usize) -> bool {
    let lits1 = problem.clause_literals(clause1_idx);
    let lits2 = problem.clause_literals(clause2_idx);
    
    // Quick check: subsumer must have fewer or equal literals
    if lits1.len() > lits2.len() {
        return false;
    }
    
    // Try to find a mapping from lits1 to lits2 with a consistent substitution
    let mut subst = ArraySubstitution::new();
    subsumes_recursive(problem, &lits1, &lits2, 0, &mut HashMap::new(), &mut subst)
}

/// Recursive helper for subsumption checking
fn subsumes_recursive(
    problem: &Problem,
    lits1: &[usize],
    lits2: &[usize],
    idx: usize,
    used: &mut HashMap<usize, usize>,
    subst: &mut ArraySubstitution,
) -> bool {
    if idx >= lits1.len() {
        return true;  // All literals from clause1 have been matched
    }
    
    let lit1 = lits1[idx];
    
    // Try to match lit1 with each unused literal in lits2
    for (i, &lit2) in lits2.iter().enumerate() {
        if used.contains_key(&i) {
            continue;  // Already used this literal
        }
        
        // Check if literals have same polarity
        if problem.node_polarities[lit1] != problem.node_polarities[lit2] {
            continue;
        }
        
        // Check if predicates match with current substitution
        if literals_match_with_subst(problem, lit1, lit2, subst) {
            used.insert(i, lit1);
            // Save the substitution state
            let saved_subst = subst.clone();
            if subsumes_recursive(problem, lits1, lits2, idx + 1, used, subst) {
                return true;
            }
            // Restore substitution on backtrack
            *subst = saved_subst;
            used.remove(&i);
        }
    }
    
    false
}

/// Check if two literals match with the given substitution
fn literals_match_with_subst(
    problem: &Problem, 
    lit1: usize, 
    lit2: usize,
    subst: &mut ArraySubstitution
) -> bool {
    // Get predicate nodes
    let pred1 = get_literal_predicate(problem, lit1);
    let pred2 = get_literal_predicate(problem, lit2);
    
    match (pred1, pred2) {
        (Some(p1), Some(p2)) => {
            // Check symbol first
            if problem.node_symbols[p1] != problem.node_symbols[p2] {
                return false;
            }
            
            // Check arity
            let args1 = problem.node_children(p1);
            let args2 = problem.node_children(p2);
            if args1.len() != args2.len() {
                return false;
            }
            
            // Try to unify the predicates with the current substitution
            // For subsumption, only variables from the first literal can be bound
            unify_for_subsumption(problem, p1, p2, subst)
        }
        _ => false,
    }
}


/// Get the predicate node of a literal
fn get_literal_predicate(problem: &Problem, lit_node: usize) -> Option<usize> {
    let children = problem.node_children(lit_node);
    if !children.is_empty() {
        Some(children[0])
    } else {
        None
    }
}

/// Unify for subsumption - only allows binding variables from term1
fn unify_for_subsumption(
    problem: &Problem,
    term1: usize,
    term2: usize,
    subst: &mut ArraySubstitution,
) -> bool {
    use crate::core::NodeType;
    
    // Stack for iterative traversal
    let mut stack = Vec::new();
    stack.push((term1, term2));
    
    #[cfg(test)]
    {
        eprintln!("unify_for_subsumption({}, {})", term1, term2);
    }
    
    while let Some((t1, t2)) = stack.pop() {
        // Same node - trivially unifiable
        if t1 == t2 {
            continue;
        }
        
        let type1 = problem.node_types[t1];
        let type2 = problem.node_types[t2];
        
        match (type1, type2) {
            (typ1, _) if typ1 == NodeType::Variable as u8 => {
                // Variable from subsumer can be bound
                // First check if any variable with the same symbol is already bound
                let var_symbol = problem.node_symbols[t1];
                
                // Check all existing bindings for variables with the same symbol
                let mut found_binding = None;
                for i in 0..problem.num_nodes {
                    if problem.node_types[i] == NodeType::Variable as u8 && 
                       problem.node_symbols[i] == var_symbol {
                        if let Some(binding) = subst.get(i) {
                            found_binding = Some(binding);
                            break;
                        }
                    }
                }
                
                if let Some(binding) = found_binding {
                    #[cfg(test)]
                    {
                        eprintln!("  Variable {} (symbol {}) already has binding {}, checking against {}", 
                                  t1, var_symbol, binding, t2);
                    }
                    // Variable with same symbol is already bound - check consistency
                    if binding != t2 {
                        // Need to check if binding and t2 can unify
                        stack.push((binding, t2));
                    }
                } else {
                    #[cfg(test)]
                    {
                        eprintln!("  Binding variable {} (symbol {}) to {}", t1, var_symbol, t2);
                    }
                    // Bind the variable
                    subst.bind(t1, t2);
                }
            }
            
            (_, typ2) if typ2 == NodeType::Variable as u8 => {
                // Variable from subsumee cannot be bound - fail
                return false;
            }
            
            (typ1, typ2) if typ1 == NodeType::Constant as u8 && typ2 == NodeType::Constant as u8 => {
                // Constants must have the same symbol
                if problem.node_symbols[t1] != problem.node_symbols[t2] {
                    return false;
                }
            }
            
            (typ1, typ2) if (typ1 == NodeType::Function as u8 && typ2 == NodeType::Function as u8) || 
                       (typ1 == NodeType::Predicate as u8 && typ2 == NodeType::Predicate as u8) => {
                // Must have same symbol and arity
                if problem.node_symbols[t1] != problem.node_symbols[t2] {
                    return false;
                }
                
                if problem.node_arities[t1] != problem.node_arities[t2] {
                    return false;
                }
                
                // Add arguments to stack for unification
                let args1 = problem.node_children(t1);
                let args2 = problem.node_children(t2);
                
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    stack.push((*a1, *a2));
                }
            }
            
            _ => {
                // Type mismatch
                return false;
            }
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::tptp_parser::parse_string;
    
    #[test]
    fn test_subsumption_basic() {
        // P(X) subsumes P(a)
        let input = r#"
            cnf(c1, axiom, p(X)).
            cnf(c2, axiom, p(a)).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Debug: print clause structures
        println!("Clause 0 literals: {:?}", problem.clause_literals(0));
        println!("Clause 1 literals: {:?}", problem.clause_literals(1));
        
        // Clause 0 (P(X)) should subsume clause 1 (P(a))
        let result01 = subsumes(&problem, 0, 1);
        println!("subsumes(0, 1) = {}", result01);
        assert!(result01);
        
        // But clause 1 (P(a)) should NOT subsume clause 0 (P(X))
        let result10 = subsumes(&problem, 1, 0);
        println!("subsumes(1, 0) = {}", result10);
        assert!(!result10);
    }
    
    #[test]
    fn test_subsumption_equality() {
        // Test that c=b does NOT subsume b=a
        let input = r#"
            cnf(c1, axiom, c = b).
            cnf(c2, axiom, b = a).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Neither should subsume the other
        assert!(!subsumes(&problem, 0, 1));
        assert!(!subsumes(&problem, 1, 0));
    }
    
    #[test]
    fn test_subsumption_with_variables() {
        // X=Y subsumes a=b
        let input = r#"
            cnf(c1, axiom, X = Y).
            cnf(c2, axiom, a = b).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Clause 0 (X=Y) should subsume clause 1 (a=b)
        assert!(subsumes(&problem, 0, 1));
        
        // But not the other way
        assert!(!subsumes(&problem, 1, 0));
    }
    
    #[test]
    fn test_subsumption_multiple_literals() {
        // P(X) | Q(Y) subsumes P(a) | Q(b) | R(c)
        let input = r#"
            cnf(c1, axiom, p(X) | q(Y)).
            cnf(c2, axiom, p(a) | q(b) | r(c)).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Clause 0 should subsume clause 1
        assert!(subsumes(&problem, 0, 1));
        
        // But not the other way
        assert!(!subsumes(&problem, 1, 0));
    }
    
    #[test]
    fn test_subsumption_consistent_substitution() {
        // P(X,X) should NOT subsume P(a,b) because X can't be both a and b
        let input = r#"
            cnf(c1, axiom, p(X,X)).
            cnf(c2, axiom, p(a,b)).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Debug output
        println!("Testing P(X,X) subsumes P(a,b)");
        let result = subsumes(&problem, 0, 1);
        println!("Result: {}", result);
        
        // Should not subsume because X can't unify with both a and b
        assert!(!result);
    }
    
    #[test]
    fn test_subsumption_reuses_variables() {
        // P(X,X) should subsume P(a,a)
        let input = r#"
            cnf(c1, axiom, p(X,X)).
            cnf(c2, axiom, p(a,a)).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Should subsume because X can be a
        assert!(subsumes(&problem, 0, 1));
    }
}