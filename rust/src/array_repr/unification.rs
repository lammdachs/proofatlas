//! Array-based unification algorithm

use crate::array_repr::types::{ArrayProblem, NodeType, ArraySubstitution};

/// Stack frame for iterative unification
#[derive(Debug, Clone, Copy)]
struct UnifyFrame {
    term1: usize,
    term2: usize,
}

/// Unify two terms represented as node indices
pub fn unify_nodes(
    problem: &ArrayProblem,
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
            (NodeType::Variable, _) => {
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
            
            (_, NodeType::Variable) => {
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
            
            (NodeType::Constant, NodeType::Constant) => {
                // Constants must have the same symbol
                if problem.node_symbols[t1] != problem.node_symbols[t2] {
                    return false;
                }
            }
            
            (NodeType::Function, NodeType::Function) => {
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
            
            (NodeType::Predicate, NodeType::Predicate) => {
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
pub fn apply_substitution(
    problem: &mut ArrayProblem,
    term_node: usize,
    subst: &ArraySubstitution,
) -> usize {
    // For now, we'll implement a simple version that modifies in place
    // A full implementation would create new nodes
    
    let node_type = problem.node_types[term_node];
    
    match node_type {
        NodeType::Variable => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_repr::builder::ArrayBuilder;
    use crate::core::logic::{Term, Predicate, Literal, Clause};
    
    #[test]
    fn test_unify_constants() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create two constants
        let a = Term::Constant("a".to_string());
        let b = Term::Constant("b".to_string());
        
        // Create dummy predicates to hold them
        let p_a = Predicate::new("P".to_string(), vec![a.clone()]);
        let p_b = Predicate::new("P".to_string(), vec![b.clone()]);
        
        let lit1 = Literal::positive(p_a);
        let lit2 = Literal::positive(p_b);
        
        let clause1 = Clause::new(vec![lit1]);
        let clause2 = Clause::new(vec![lit2]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Get the constant nodes (they should be children of predicates)
        let pred1_node = problem.node_children(problem.clause_literals(0)[0])[0];
        let pred2_node = problem.node_children(problem.clause_literals(1)[0])[0];
        
        let const1_node = problem.node_children(pred1_node)[0];
        let const2_node = problem.node_children(pred2_node)[0];
        
        let mut subst = ArraySubstitution::new();
        
        // Same constant should unify
        assert!(unify_nodes(&problem, const1_node, const1_node, &mut subst));
        
        // Different constants should not unify
        assert!(!unify_nodes(&problem, const1_node, const2_node, &mut subst));
    }
    
    #[test]
    fn test_unify_variable() {
        let mut problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut problem);
        
        // Create a variable and a constant
        let x = Term::Variable("X".to_string());
        let a = Term::Constant("a".to_string());
        
        // Create predicates
        let p_x = Predicate::new("P".to_string(), vec![x]);
        let p_a = Predicate::new("P".to_string(), vec![a]);
        
        let lit1 = Literal::positive(p_x);
        let lit2 = Literal::positive(p_a);
        
        let clause1 = Clause::new(vec![lit1]);
        let clause2 = Clause::new(vec![lit2]);
        
        builder.add_clause(&clause1);
        builder.add_clause(&clause2);
        
        // Get the term nodes
        let pred1_node = problem.node_children(problem.clause_literals(0)[0])[0];
        let pred2_node = problem.node_children(problem.clause_literals(1)[0])[0];
        
        let var_node = problem.node_children(pred1_node)[0];
        let const_node = problem.node_children(pred2_node)[0];
        
        let mut subst = ArraySubstitution::new();
        
        // Variable should unify with constant
        assert!(unify_nodes(&problem, var_node, const_node, &mut subst));
        
        // Check substitution
        assert_eq!(subst.get(var_node), Some(const_node));
    }
}

#[cfg(test)]
#[path = "unification_tests.rs"]
mod extended_tests;