//! Array-based unification algorithm

use crate::core::{Problem, NodeType, ArraySubstitution};

/// Substitution for proof tracking - using ArraySubstitution internally
pub type Substitution = ArraySubstitution;

impl Substitution {
    /// Create an empty substitution
    pub fn empty() -> Self {
        Self::new()
    }
}

/// Stack frame for iterative unification
#[derive(Debug, Clone, Copy)]
struct UnifyFrame {
    term1: usize,
    term2: usize,
}

/// Unify two terms represented as node indices
pub fn unify_nodes(
    problem: &Problem,
    node1: usize,
    node2: usize,
    subst: &mut ArraySubstitution,
) -> bool {
    // Use a stack for iterative traversal
    let mut stack = Vec::new();
    stack.push(UnifyFrame {
        term1: node1,
        term2: node2,
    });
    
    while let Some(frame) = stack.pop() {
        let t1 = frame.term1;
        let t2 = frame.term2;
        
        
        // Same node - trivially unifiable
        if t1 == t2 {
            continue;
        }
        
        let type1 = problem.node_types[t1];
        let type2 = problem.node_types[t2];
        
        match (type1, type2) {
            (typ1, _) if typ1 == NodeType::Variable as u8 => {
                // Check if variable is already bound
                if let Some(binding) = subst.get(t1) {
                    // Follow the binding
                    stack.push(UnifyFrame {
                        term1: binding,
                        term2: t2,
                    });
                } else {
                    // Bind the variable
                    subst.bind(t1, t2);
                }
            }
            
            (_, typ2) if typ2 == NodeType::Variable as u8 => {
                // Check if variable is already bound
                if let Some(binding) = subst.get(t2) {
                    // Follow the binding
                    stack.push(UnifyFrame {
                        term1: t1,
                        term2: binding,
                    });
                } else {
                    // Bind the variable
                    subst.bind(t2, t1);
                }
            }
            
            (typ1, typ2) if typ1 == NodeType::Constant as u8 && typ2 == NodeType::Constant as u8 => {
                // Constants must have the same symbol
                if problem.node_symbols[t1] != problem.node_symbols[t2] {
                    return false;
                }
            }
            
            (typ1, typ2) if typ1 == NodeType::Function as u8 && typ2 == NodeType::Function as u8 => {
                // Functions must have same symbol and arity
                if problem.node_symbols[t1] != problem.node_symbols[t2] {
                    return false;
                }
                
                if problem.node_arities[t1] != problem.node_arities[t2] {
                    return false;
                }
                
                // Add arguments to stack for unification
                let children1 = problem.node_children(t1);
                let children2 = problem.node_children(t2);
                
                for (c1, c2) in children1.iter().zip(children2.iter()) {
                    stack.push(UnifyFrame {
                        term1: *c1,
                        term2: *c2,
                    });
                }
            }
            
            (typ1, typ2) if typ1 == NodeType::Predicate as u8 && typ2 == NodeType::Predicate as u8 => {
                // Predicates must have same symbol and arity
                if problem.node_symbols[t1] != problem.node_symbols[t2] {
                    return false;
                }
                
                if problem.node_arities[t1] != problem.node_arities[t2] {
                    return false;
                }
                
                // Add arguments to stack for unification
                let children1 = problem.node_children(t1);
                let children2 = problem.node_children(t2);
                
                for (c1, c2) in children1.iter().zip(children2.iter()) {
                    stack.push(UnifyFrame {
                        term1: *c1,
                        term2: *c2,
                    });
                }
            }
            
            _ => {
                // Different types cannot unify
                return false;
            }
        }
    }
    
    
    true
}

/// Apply a substitution to a term, creating new nodes
#[allow(dead_code)]
pub fn apply_substitution(
    problem: &mut Problem,
    term_node: usize,
    subst: &ArraySubstitution,
) -> usize {
    // For now, we'll implement a simple version that modifies in place
    // A full implementation would create new nodes
    
    let node_type = problem.node_types[term_node];
    
    match node_type {
        t if t == NodeType::Variable as u8 => {
            // Check if this variable has a substitution
            if let Some(replacement) = subst.get(term_node) {
                replacement
            } else {
                term_node
            }
        }
        _ => {
            // For non-variables, recursively apply to children
            // This is a placeholder - full implementation would create new nodes
            term_node
        }
    }
}

// Tests temporarily disabled during refactoring - need to be rewritten with new structure

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::tptp_parser::parse_string;
    
    #[test]
    fn test_unification_shared_variables() {
        // Test case: P(X,X) should unify with P(Y,Y) with X->Y
        // But should NOT unify with P(a,b)
        let input = r#"
            cnf(c1, axiom, p(X,X)).
            cnf(c2, axiom, p(Y,Y)).
            cnf(c3, axiom, p(a,b)).
            cnf(c4, axiom, p(a,a)).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Find predicate nodes
        let mut predicates = Vec::new();
        for i in 0..problem.num_nodes {
            if problem.node_types[i] == crate::core::NodeType::Predicate as u8 {
                predicates.push(i);
            }
        }
        
        let pred1 = predicates[0];  // P(X,X)
        let pred2 = predicates[1];  // P(Y,Y)
        let pred3 = predicates[2];  // P(a,b)
        let pred4 = predicates[3];  // P(a,a)
        
        // Test 1: P(X,X) should unify with P(Y,Y)
        let mut subst1 = ArraySubstitution::new();
        let result1 = unify_nodes(&problem, pred1, pred2, &mut subst1);
        assert!(result1, "P(X,X) should unify with P(Y,Y)");
        
        // Test 2: P(X,X) should NOT unify with P(a,b)
        let mut subst2 = ArraySubstitution::new();
        let result2 = unify_nodes(&problem, pred1, pred3, &mut subst2);
        assert!(!result2, "P(X,X) should NOT unify with P(a,b)");
        
        // Test 3: P(X,X) should unify with P(a,a)
        let mut subst3 = ArraySubstitution::new();
        let result3 = unify_nodes(&problem, pred1, pred4, &mut subst3);
        assert!(result3, "P(X,X) should unify with P(a,a)");
    }
    
    #[test]
    fn test_unification_shared_between_terms() {
        // Test case: P(X,f(X)) should unify with P(a,f(a)) but not P(a,f(b))
        let input = r#"
            cnf(c1, axiom, p(X,f(X))).
            cnf(c2, axiom, p(a,f(a))).
            cnf(c3, axiom, p(a,f(b))).
        "#;
        
        let problem = parse_string(input).expect("Failed to parse");
        
        // Find predicate nodes
        let mut predicates = Vec::new();
        for i in 0..problem.num_nodes {
            if problem.node_types[i] == crate::core::NodeType::Predicate as u8 {
                predicates.push(i);
            }
        }
        
        let pred1 = predicates[0];  // P(X,f(X))
        let pred2 = predicates[1];  // P(a,f(a))
        let pred3 = predicates[2];  // P(a,f(b))
        
        // P(X,f(X)) should unify with P(a,f(a))
        let mut subst1 = ArraySubstitution::new();
        let result1 = unify_nodes(&problem, pred1, pred2, &mut subst1);
        assert!(result1, "P(X,f(X)) should unify with P(a,f(a))");
        
        // P(X,f(X)) should NOT unify with P(a,f(b))
        let mut subst2 = ArraySubstitution::new();
        let result2 = unify_nodes(&problem, pred1, pred3, &mut subst2);
        assert!(!result2, "P(X,f(X)) should NOT unify with P(a,f(b))");
    }
}