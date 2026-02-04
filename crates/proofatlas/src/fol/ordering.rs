//! Term ordering implementation (Knuth-Bendix Ordering)

use super::interner::{ConstantId, FunctionId, Interner, VariableId};
use super::Term;
use std::collections::HashMap;

/// Configuration for Knuth-Bendix Ordering
#[derive(Debug, Clone)]
pub struct KBOConfig {
    /// Weight of each function/constant symbol by ID (default weight is 1)
    pub function_weights: HashMap<FunctionId, usize>,
    pub constant_weights: HashMap<ConstantId, usize>,
    /// Precedence ordering of symbols by ID (higher value = higher precedence)
    pub function_precedence: HashMap<FunctionId, usize>,
    pub constant_precedence: HashMap<ConstantId, usize>,
    /// Weight of variables (must be positive)
    pub variable_weight: usize,
}

impl Default for KBOConfig {
    fn default() -> Self {
        KBOConfig {
            function_weights: HashMap::new(),
            constant_weights: HashMap::new(),
            function_precedence: HashMap::new(),
            constant_precedence: HashMap::new(),
            variable_weight: 1,
        }
    }
}

impl KBOConfig {
    /// Create a KBOConfig from string-based weights/precedences using an interner
    pub fn from_strings(
        interner: &Interner,
        symbol_weights: &HashMap<String, usize>,
        symbol_precedence: &HashMap<String, usize>,
        variable_weight: usize,
    ) -> Self {
        let mut config = KBOConfig {
            variable_weight,
            ..Default::default()
        };

        for (name, &weight) in symbol_weights {
            if let Some(fid) = interner.get_function(name) {
                config.function_weights.insert(fid, weight);
            }
            if let Some(cid) = interner.get_constant(name) {
                config.constant_weights.insert(cid, weight);
            }
        }

        for (name, &prec) in symbol_precedence {
            if let Some(fid) = interner.get_function(name) {
                config.function_precedence.insert(fid, prec);
            }
            if let Some(cid) = interner.get_constant(name) {
                config.constant_precedence.insert(cid, prec);
            }
        }

        config
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

    /// Get weight of a function symbol (default is 1)
    fn function_weight(&self, id: FunctionId) -> usize {
        self.config.function_weights.get(&id).copied().unwrap_or(1)
    }

    /// Get weight of a constant symbol (default is 1)
    fn constant_weight(&self, id: ConstantId) -> usize {
        self.config.constant_weights.get(&id).copied().unwrap_or(1)
    }

    /// Get precedence of a function symbol (default is 0)
    fn function_precedence(&self, id: FunctionId) -> usize {
        self.config
            .function_precedence
            .get(&id)
            .copied()
            .unwrap_or(0)
    }

    /// Get precedence of a constant symbol (default is 0)
    fn constant_precedence(&self, id: ConstantId) -> usize {
        self.config
            .constant_precedence
            .get(&id)
            .copied()
            .unwrap_or(0)
    }

    /// Calculate the weight of a term
    pub fn term_weight(&self, term: &Term) -> usize {
        match term {
            Term::Variable(_) => self.config.variable_weight,
            Term::Constant(c) => self.constant_weight(c.id),
            Term::Function(f, args) => {
                let func_weight = self.function_weight(f.id);
                let args_weight: usize = args.iter().map(|t| self.term_weight(t)).sum();
                func_weight + args_weight
            }
        }
    }

    /// Count occurrences of each variable in a term
    pub fn count_variables(&self, term: &Term) -> HashMap<VariableId, usize> {
        let mut counts = HashMap::new();
        self.count_variables_rec(term, &mut counts);
        counts
    }

    fn count_variables_rec(&self, term: &Term, counts: &mut HashMap<VariableId, usize>) {
        match term {
            Term::Variable(v) => {
                *counts.entry(v.id).or_insert(0) += 1;
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
        let s_gt_t_var_cond = vars_t.iter().all(|(var_id, count_t)| {
            let count_s = vars_s.get(var_id).copied().unwrap_or(0);
            count_s >= *count_t
        });

        // For t > s, need #(x, t) ≥ #(x, s) for all variables x
        let t_gt_s_var_cond = vars_s.iter().all(|(var_id, count_s)| {
            let count_t = vars_t.get(var_id).copied().unwrap_or(0);
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
                    // Variables are totally ordered by ID
                    if v1.id > v2.id {
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
                if c1.id == c2.id {
                    Ordering::Equal
                } else {
                    let prec1 = self.constant_precedence(c1.id);
                    let prec2 = self.constant_precedence(c2.id);
                    if prec1 > prec2 {
                        Ordering::Greater
                    } else if prec1 < prec2 {
                        Ordering::Less
                    } else {
                        // Same precedence, use ID comparison
                        if c1.id > c2.id {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    }
                }
            }
            (Term::Function(f1, args1), Term::Function(f2, args2)) => {
                if f1.id != f2.id {
                    // Different function symbols
                    let prec1 = self.function_precedence(f1.id);
                    let prec2 = self.function_precedence(f2.id);
                    if prec1 > prec2 {
                        Ordering::Greater
                    } else if prec1 < prec2 {
                        Ordering::Less
                    } else {
                        // Same precedence, use ID comparison
                        if f1.id > f2.id {
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
    use crate::fol::{Constant, FunctionSymbol, Interner, Variable};

    #[test]
    fn test_term_weight() {
        let mut interner = Interner::new();
        let x_id = interner.intern_variable("X");
        let a_id = interner.intern_constant("a");
        let f_id = interner.intern_function("f");

        let kbo = KBO::new(KBOConfig::default());

        // Variable
        let x = Term::Variable(Variable::new(x_id));
        assert_eq!(kbo.term_weight(&x), 1);

        // Constant
        let a = Term::Constant(Constant::new(a_id));
        assert_eq!(kbo.term_weight(&a), 1);

        // Function f(a, X)
        let f = FunctionSymbol::new(f_id, 2);
        let fa_x = Term::Function(f, vec![a.clone(), x.clone()]);
        assert_eq!(kbo.term_weight(&fa_x), 3); // f(1) + a(1) + X(1)
    }

    #[test]
    fn test_variable_condition() {
        let mut interner = Interner::new();
        let x_id = interner.intern_variable("X");
        let y_id = interner.intern_variable("Y");
        let a_id = interner.intern_constant("a");
        let f_id = interner.intern_function("f");

        let kbo = KBO::new(KBOConfig::default());

        let x = Term::Variable(Variable::new(x_id));
        let y = Term::Variable(Variable::new(y_id));
        let a = Term::Constant(Constant::new(a_id));

        // X > Y should be incomparable (different variables)
        assert_eq!(kbo.compare(&x, &y), Ordering::Incomparable);

        // a > X is incomparable (variable condition not satisfied)
        assert_eq!(kbo.compare(&a, &x), Ordering::Incomparable);

        // f(X) > X should be valid
        let f = FunctionSymbol::new(f_id, 1);
        let fx = Term::Function(f, vec![x.clone()]);
        assert_eq!(kbo.compare(&fx, &x), Ordering::Greater);
    }

    #[test]
    fn test_precedence() {
        let mut interner = Interner::new();
        let a_id = interner.intern_constant("a");
        let f_id = interner.intern_function("f");
        let g_id = interner.intern_function("g");

        let mut config = KBOConfig::default();
        config.function_precedence.insert(f_id, 2);
        config.function_precedence.insert(g_id, 1);

        let kbo = KBO::new(config);

        let a = Term::Constant(Constant::new(a_id));
        let f = FunctionSymbol::new(f_id, 1);
        let g = FunctionSymbol::new(g_id, 1);

        let fa = Term::Function(f, vec![a.clone()]);
        let ga = Term::Function(g, vec![a.clone()]);

        // f(a) > g(a) because f has higher precedence
        assert_eq!(kbo.compare(&fa, &ga), Ordering::Greater);
    }
}
