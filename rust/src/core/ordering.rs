//! Term ordering for superposition calculus
//! 
//! This module implements a basic Knuth-Bendix Ordering (KBO) with uniform weights.
//! KBO is used to constrain superposition inferences and ensure completeness.

use crate::core::{Problem, NodeType};
use std::cmp::Ordering;

/// Calculate the weight of a term (number of symbols)
pub fn term_weight(problem: &Problem, term: usize) -> usize {
    match problem.node_types[term] {
        t if t == NodeType::Variable as u8 => 1,
        t if t == NodeType::Constant as u8 => 1,
        t if t == NodeType::Function as u8 => {
            let children = problem.node_children(term);
            1 + children.iter().map(|&child| term_weight(problem, child)).sum::<usize>()
        }
        t if t == NodeType::Predicate as u8 => {
            let children = problem.node_children(term);
            1 + children.iter().map(|&child| term_weight(problem, child)).sum::<usize>()
        }
        _ => 0, // Literals and clauses don't have weight
    }
}

/// Count occurrences of each variable in a term
pub fn count_variables(problem: &Problem, term: usize) -> Vec<(u32, usize)> {
    let mut counts = Vec::new();
    count_variables_recursive(problem, term, &mut counts);
    counts.sort_by_key(|&(var, _)| var);
    counts
}

fn count_variables_recursive(problem: &Problem, term: usize, counts: &mut Vec<(u32, usize)>) {
    match problem.node_types[term] {
        t if t == NodeType::Variable as u8 => {
            let var_symbol = problem.node_symbols[term];
            if let Some(entry) = counts.iter_mut().find(|(sym, _)| *sym == var_symbol) {
                entry.1 += 1;
            } else {
                counts.push((var_symbol, 1));
            }
        }
        t if t == NodeType::Function as u8 || t == NodeType::Predicate as u8 => {
            let children = problem.node_children(term);
            for &child in children.iter() {
                count_variables_recursive(problem, child, counts);
            }
        }
        _ => {}
    }
}

/// Compare two terms using KBO (Knuth-Bendix Ordering)
/// Returns Ordering::Greater if term1 > term2
/// Returns Ordering::Less if term1 < term2  
/// Returns Ordering::Equal if terms are equivalent
pub fn kbo_compare(problem: &Problem, term1: usize, term2: usize) -> Ordering {
    // First, handle the special case of variables
    let is_var1 = problem.node_types[term1] == NodeType::Variable as u8;
    let is_var2 = problem.node_types[term2] == NodeType::Variable as u8;
    
    match (is_var1, is_var2) {
        (true, true) => {
            // Both are variables - compare their symbols
            let sym1 = problem.node_symbols[term1];
            let sym2 = problem.node_symbols[term2];
            if sym1 == sym2 {
                Ordering::Equal
            } else {
                // Compare variable names lexicographically
                let name1 = problem.symbols.get(sym1).unwrap_or("");
                let name2 = problem.symbols.get(sym2).unwrap_or("");
                name1.cmp(name2)
            }
        }
        (true, false) => {
            // Variable vs non-variable: variable is always smaller
            Ordering::Less
        }
        (false, true) => {
            // Non-variable vs variable: non-variable is always greater
            Ordering::Greater
        }
        (false, false) => {
            // Neither is a variable - proceed with standard KBO
            
            // Check variable occurrences
            let vars1 = count_variables(problem, term1);
            let vars2 = count_variables(problem, term2);
            
            // Check if all variables in term2 occur in term1 with at least the same multiplicity
            for (var2, count2) in &vars2 {
                let count1 = vars1.iter()
                    .find(|(var1, _)| var1 == var2)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);
                if count1 < *count2 {
                    return Ordering::Less; // term1 < term2
                }
            }
            
            // Check if term1 has variables that term2 doesn't
            for (var1, _) in &vars1 {
                if !vars2.iter().any(|(var2, _)| var1 == var2) {
                    return Ordering::Greater; // term1 > term2
                }
            }
            
            // Compare weights
            let weight1 = term_weight(problem, term1);
            let weight2 = term_weight(problem, term2);
            
            match weight1.cmp(&weight2) {
                Ordering::Greater => Ordering::Greater,
                Ordering::Less => Ordering::Less,
                Ordering::Equal => {
                    // Same weight, use lexicographic comparison
                    lexicographic_compare(problem, term1, term2)
                }
            }
        }
    }
}

/// Lexicographic comparison of terms with same weight
fn lexicographic_compare(problem: &Problem, term1: usize, term2: usize) -> Ordering {
    // Compare node types first
    let type1 = problem.node_types[term1];
    let type2 = problem.node_types[term2];
    
    match (type1, type2) {
        (t1, t2) if t1 == NodeType::Variable as u8 && t2 == NodeType::Variable as u8 => {
            // Compare variable symbol strings
            let sym1 = problem.symbols.get(problem.node_symbols[term1]).unwrap_or("");
            let sym2 = problem.symbols.get(problem.node_symbols[term2]).unwrap_or("");
            sym1.cmp(sym2)
        }
        (t1, t2) if t1 == NodeType::Constant as u8 && t2 == NodeType::Constant as u8 => {
            // Compare constant symbol strings, not IDs!
            let sym1 = problem.symbols.get(problem.node_symbols[term1]).unwrap_or("");
            let sym2 = problem.symbols.get(problem.node_symbols[term2]).unwrap_or("");
            let result = sym1.cmp(sym2);
            result
        }
        (t1, t2) if t1 == NodeType::Function as u8 && t2 == NodeType::Function as u8 => {
            // First compare function symbol strings
            let sym1 = problem.symbols.get(problem.node_symbols[term1]).unwrap_or("");
            let sym2 = problem.symbols.get(problem.node_symbols[term2]).unwrap_or("");
            match sym1.cmp(sym2) {
                Ordering::Equal => {
                    // Same function, compare arguments lexicographically
                    let children1 = problem.node_children(term1);
                    let children2 = problem.node_children(term2);
                    
                    for (c1, c2) in children1.iter().zip(children2.iter()) {
                        match kbo_compare(problem, *c1, *c2) {
                            Ordering::Equal => continue,
                            other => return other,
                        }
                    }
                    
                    // All arguments equal
                    Ordering::Equal
                }
                other => other,
            }
        }
        // Variables are handled in kbo_compare, so this shouldn't happen
        // But if it does, maintain consistency: Variables < Constants < Functions
        (t1, t2) if t1 == NodeType::Constant as u8 && t2 == NodeType::Function as u8 => Ordering::Less,
        (t1, t2) if t1 == NodeType::Function as u8 && t2 == NodeType::Constant as u8 => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::tptp_parser::parse_string;
    
    #[test]
    fn test_term_weight() {
        // Use parser to create terms
        let problem = parse_string("
            cnf(test, axiom, (p(X) | q(a) | r(f(a)) | s(g(X, f(a))))).
        ").expect("Failed to parse");
        
        // The clause has 4 literals with different terms
        let clause_lits = problem.clause_literals(0);
        
        // First literal p(X) - get X
        let pred1 = problem.node_children(clause_lits[0])[0];
        let x_idx = problem.node_children(pred1)[0];
        assert_eq!(term_weight(&problem, x_idx), 1); // variable weight
        
        // Second literal q(a) - get a
        let pred2 = problem.node_children(clause_lits[1])[0];
        let a_idx = problem.node_children(pred2)[0];
        assert_eq!(term_weight(&problem, a_idx), 1); // constant weight
        
        // Third literal r(f(a)) - get f(a)
        let pred3 = problem.node_children(clause_lits[2])[0];
        let f_a_idx = problem.node_children(pred3)[0];
        assert_eq!(term_weight(&problem, f_a_idx), 2); // f + a
        
        // Fourth literal s(g(X, f(a))) - get g(X, f(a))
        let pred4 = problem.node_children(clause_lits[3])[0];
        let g_idx = problem.node_children(pred4)[0];
        assert_eq!(term_weight(&problem, g_idx), 4); // g + X + f + a
    }
    
    #[test]
    fn test_variable_counting() {
        // Use parser to create f(X, X)
        let problem = parse_string("
            cnf(test, axiom, p(f(X, X))).
        ").expect("Failed to parse");
        
        // Get f(X, X) term
        let clause_lits = problem.clause_literals(0);
        let pred = problem.node_children(clause_lits[0])[0];
        let f_xx_idx = problem.node_children(pred)[0];
        
        let vars = count_variables(&problem, f_xx_idx);
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].1, 2); // X appears twice
    }
    
    #[test] 
    fn test_variable_counting_multiple() {
        // Use parser to create g(X, Y)
        let problem = parse_string("
            cnf(test, axiom, p(g(X, Y))).
        ").expect("Failed to parse");
        
        // Get g(X, Y) term
        let clause_lits = problem.clause_literals(0);
        let pred = problem.node_children(clause_lits[0])[0];
        let g_xy_idx = problem.node_children(pred)[0];
        
        let vars = count_variables(&problem, g_xy_idx);
        assert_eq!(vars.len(), 2);
    }
    
    #[test]
    fn test_kbo_basic() {
        // Create terms using parser
        let problem = parse_string("
            cnf(c1, axiom, p(X)).
            cnf(c2, axiom, q(a)).
            cnf(c3, axiom, r(f(X))).
        ").expect("Failed to parse");
        
        // Get terms from predicates
        let c1_lits = problem.clause_literals(0);
        let pred1 = problem.node_children(c1_lits[0])[0];
        let x_idx = problem.node_children(pred1)[0];
        
        let c2_lits = problem.clause_literals(1);
        let pred2 = problem.node_children(c2_lits[0])[0];
        let a_idx = problem.node_children(pred2)[0];
        
        let c3_lits = problem.clause_literals(2);
        let pred3 = problem.node_children(c3_lits[0])[0];
        let f_x_idx = problem.node_children(pred3)[0];
        
        // f(X) > X (contains X as subterm)
        assert_eq!(kbo_compare(&problem, f_x_idx, x_idx), Ordering::Greater);
        
        // a and X: since X doesn't occur in a but a doesn't contain X, a < X
        assert_eq!(kbo_compare(&problem, a_idx, x_idx), Ordering::Less);
    }
    
    #[test]
    fn test_kbo_variable_condition() {
        // Build f(X, Y) and g(X)
        let problem = parse_string("
            cnf(c1, axiom, p(f(X, Y))).
            cnf(c2, axiom, q(g(X))).
        ").expect("Failed to parse");
        
        // Get f(X, Y) from first clause
        let c1_lits = problem.clause_literals(0);
        let pred1 = problem.node_children(c1_lits[0])[0];
        let f_xy_idx = problem.node_children(pred1)[0];
        
        // Get g(X) from second clause
        let c2_lits = problem.clause_literals(1);
        let pred2 = problem.node_children(c2_lits[0])[0];
        let g_x_idx = problem.node_children(pred2)[0];
        
        // f(X,Y) > g(X) because f(X,Y) has variable Y that g(X) doesn't have
        assert_eq!(kbo_compare(&problem, f_xy_idx, g_x_idx), Ordering::Greater);
        
        // g(X) cannot be greater than f(X,Y) due to missing Y
        assert_eq!(kbo_compare(&problem, g_x_idx, f_xy_idx), Ordering::Less);
    }
}