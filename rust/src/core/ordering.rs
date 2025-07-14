//! Term ordering implementation (Knuth-Bendix Ordering)

use super::{Term, Variable};
use std::collections::HashMap;

/// Configuration for Knuth-Bendix Ordering
#[derive(Debug, Clone)]
pub struct KBOConfig {
    /// Weight of each function/constant symbol (default weight is 1)
    pub symbol_weights: HashMap<String, usize>,
    /// Precedence ordering of symbols (higher value = higher precedence)
    pub symbol_precedence: HashMap<String, usize>,
    /// Weight of variables (must be positive)
    pub variable_weight: usize,
}

impl Default for KBOConfig {
    fn default() -> Self {
        KBOConfig {
            symbol_weights: HashMap::new(),
            symbol_precedence: HashMap::new(),
            variable_weight: 1,
        }
    }
}

/// Result of comparing two terms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    Greater,
    Less,
    Equal,
    Incomparable,
}

/// Knuth-Bendix Ordering implementation
pub struct KBO {
    config: KBOConfig,
}

impl KBO {
    pub fn new(config: KBOConfig) -> Self {
        KBO { config }
    }
    
    /// Get weight of a symbol (default is 1)
    fn symbol_weight(&self, name: &str) -> usize {
        self.config.symbol_weights.get(name).copied().unwrap_or(1)
    }
    
    /// Get precedence of a symbol (default is 0)
    fn symbol_precedence(&self, name: &str) -> usize {
        self.config.symbol_precedence.get(name).copied().unwrap_or(0)
    }
    
    /// Calculate the weight of a term
    pub fn term_weight(&self, term: &Term) -> usize {
        match term {
            Term::Variable(_) => self.config.variable_weight,
            Term::Constant(c) => self.symbol_weight(&c.name),
            Term::Function(f, args) => {
                let func_weight = self.symbol_weight(&f.name);
                let args_weight: usize = args.iter().map(|t| self.term_weight(t)).sum();
                func_weight + args_weight
            }
        }
    }
    
    /// Count occurrences of each variable in a term
    pub fn count_variables(&self, term: &Term) -> HashMap<Variable, usize> {
        let mut counts = HashMap::new();
        self.count_variables_rec(term, &mut counts);
        counts
    }
    
    fn count_variables_rec(&self, term: &Term, counts: &mut HashMap<Variable, usize>) {
        match term {
            Term::Variable(v) => {
                *counts.entry(v.clone()).or_insert(0) += 1;
            }
            Term::Constant(_) => {}
            Term::Function(_, args) => {
                for arg in args {
                    self.count_variables_rec(arg, counts);
                }
            }
        }
    }
    
    /// Compare two terms using KBO
    pub fn compare(&self, s: &Term, t: &Term) -> Ordering {
        // First check if terms are syntactically equal
        if s == t {
            return Ordering::Equal;
        }
        
        // Check variable condition for s > t
        let vars_s = self.count_variables(s);
        let vars_t = self.count_variables(t);
        
        // For s > t, need #(x, s) ≥ #(x, t) for all variables x
        let s_gt_t_var_cond = vars_t.iter().all(|(var, count_t)| {
            let count_s = vars_s.get(var).copied().unwrap_or(0);
            count_s >= *count_t
        });
        
        // For t > s, need #(x, t) ≥ #(x, s) for all variables x
        let t_gt_s_var_cond = vars_s.iter().all(|(var, count_s)| {
            let count_t = vars_t.get(var).copied().unwrap_or(0);
            count_t >= *count_s
        });
        
        // Compare weights
        let weight_s = self.term_weight(s);
        let weight_t = self.term_weight(t);
        
        if weight_s > weight_t && s_gt_t_var_cond {
            Ordering::Greater
        } else if weight_t > weight_s && t_gt_s_var_cond {
            Ordering::Less
        } else if weight_s == weight_t {
            // Equal weight, check lexicographic ordering
            if s_gt_t_var_cond && t_gt_s_var_cond {
                // Both variable conditions hold, use pure lexicographic
                self.compare_lex(s, t)
            } else if s_gt_t_var_cond {
                // Only s > t possible
                let lex = self.compare_lex(s, t);
                if lex == Ordering::Greater || lex == Ordering::Equal {
                    lex
                } else {
                    Ordering::Incomparable
                }
            } else if t_gt_s_var_cond {
                // Only t > s possible
                let lex = self.compare_lex(s, t);
                if lex == Ordering::Less || lex == Ordering::Equal {
                    lex
                } else {
                    Ordering::Incomparable
                }
            } else {
                // Neither variable condition holds
                Ordering::Incomparable
            }
        } else {
            Ordering::Incomparable
        }
    }
    
    /// Lexicographic comparison for terms of equal weight
    fn compare_lex(&self, s: &Term, t: &Term) -> Ordering {
        match (s, t) {
            (Term::Variable(v1), Term::Variable(v2)) => {
                if v1 == v2 {
                    Ordering::Equal
                } else {
                    // Variables are totally ordered by name
                    if v1.name > v2.name {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
            }
            // Variable vs non-variable: variable is always smaller in lex ordering
            (Term::Variable(_), _) => Ordering::Less,
            (_, Term::Variable(_)) => Ordering::Greater,
            (Term::Constant(c1), Term::Constant(c2)) => {
                if c1.name == c2.name {
                    Ordering::Equal
                } else {
                    let prec1 = self.symbol_precedence(&c1.name);
                    let prec2 = self.symbol_precedence(&c2.name);
                    if prec1 > prec2 {
                        Ordering::Greater
                    } else if prec1 < prec2 {
                        Ordering::Less
                    } else {
                        // Same precedence, use name comparison
                        if c1.name > c2.name {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                }
            }
            (Term::Function(f1, args1), Term::Function(f2, args2)) => {
                if f1.name != f2.name {
                    // Different function symbols
                    let prec1 = self.symbol_precedence(&f1.name);
                    let prec2 = self.symbol_precedence(&f2.name);
                    if prec1 > prec2 {
                        Ordering::Greater
                    } else if prec1 < prec2 {
                        Ordering::Less
                    } else {
                        // Same precedence, use name comparison
                        if f1.name > f2.name {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                } else {
                    // Same function symbol, compare arguments lexicographically
                    for (arg1, arg2) in args1.iter().zip(args2.iter()) {
                        match self.compare(arg1, arg2) {
                            Ordering::Equal => continue,
                            other => return other,
                        }
                    }
                    Ordering::Equal
                }
            }
            (Term::Function(_, _), Term::Constant(_)) => Ordering::Greater,
            (Term::Constant(_), Term::Function(_, _)) => Ordering::Less,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Constant, FunctionSymbol};
    
    #[test]
    fn test_term_weight() {
        let kbo = KBO::new(KBOConfig::default());
        
        // Variable
        let x = Term::Variable(Variable { name: "X".to_string() });
        assert_eq!(kbo.term_weight(&x), 1);
        
        // Constant
        let a = Term::Constant(Constant { name: "a".to_string() });
        assert_eq!(kbo.term_weight(&a), 1);
        
        // Function f(a, X)
        let f = FunctionSymbol { name: "f".to_string(), arity: 2 };
        let fa_x = Term::Function(f, vec![a.clone(), x.clone()]);
        assert_eq!(kbo.term_weight(&fa_x), 3); // f(1) + a(1) + X(1)
    }
    
    #[test]
    fn test_variable_condition() {
        let kbo = KBO::new(KBOConfig::default());
        
        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });
        
        // X > Y should be incomparable (different variables)
        assert_eq!(kbo.compare(&x, &y), Ordering::Incomparable);
        
        // a > X is incomparable (variable condition not satisfied)
        assert_eq!(kbo.compare(&a, &x), Ordering::Incomparable);
        
        // f(X) > X should be valid
        let f = FunctionSymbol { name: "f".to_string(), arity: 1 };
        let fx = Term::Function(f, vec![x.clone()]);
        assert_eq!(kbo.compare(&fx, &x), Ordering::Greater);
    }
    
    #[test]
    fn test_precedence() {
        let mut config = KBOConfig::default();
        config.symbol_precedence.insert("f".to_string(), 2);
        config.symbol_precedence.insert("g".to_string(), 1);
        
        let kbo = KBO::new(config);
        
        let a = Term::Constant(Constant { name: "a".to_string() });
        let f = FunctionSymbol { name: "f".to_string(), arity: 1 };
        let g = FunctionSymbol { name: "g".to_string(), arity: 1 };
        
        let fa = Term::Function(f, vec![a.clone()]);
        let ga = Term::Function(g, vec![a.clone()]);
        
        // f(a) > g(a) because f has higher precedence
        assert_eq!(kbo.compare(&fa, &ga), Ordering::Greater);
    }
}