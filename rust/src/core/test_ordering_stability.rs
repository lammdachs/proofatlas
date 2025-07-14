//! Test that demonstrates the KBO stability issue

#[cfg(test)]
mod tests {
    use crate::core::{Term, Variable, Constant, FunctionSymbol, KBO, KBOConfig, TermOrdering, Substitution};
    
    #[test]
    fn test_kbo_stability_fixed() {
        let kbo = KBO::new(KBOConfig::default());
        
        // Create terms: f(X) and a
        let x = Term::Variable(Variable { name: "X".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });
        let f = FunctionSymbol { name: "f".to_string(), arity: 1 };
        let fx = Term::Function(f.clone(), vec![x.clone()]);
        
        // Check f(X) > a
        let cmp1 = kbo.compare(&fx, &a);
        println!("f(X) vs a: {:?}", cmp1);
        
        // Create substitution X -> b
        let b = Term::Constant(Constant { name: "b".to_string() });
        let mut subst = Substitution::new();
        subst.insert(Variable { name: "X".to_string() }, b.clone());
        
        // Apply substitution: f(X)σ = f(b), aσ = a
        let fx_sigma = fx.apply_substitution(&subst);
        let a_sigma = a.apply_substitution(&subst);
        
        println!("After substitution X -> b:");
        println!("f(X)σ = {}", fx_sigma);
        println!("aσ = {}", a_sigma);
        
        // Check f(b) > a
        let cmp2 = kbo.compare(&fx_sigma, &a_sigma);
        println!("f(b) vs a: {:?}", cmp2);
        
        // This should pass: if f(X) > a, then f(b) > a
        // But it might fail due to the variable handling issue
        match (cmp1, cmp2) {
            (TermOrdering::Greater, TermOrdering::Greater) => {
                println!("✓ Stability holds");
            }
            (TermOrdering::Greater, _) => {
                panic!("✗ Stability violated! f(X) > a but f(b) is not > a");
            }
            _ => {
                println!("Original comparison was not Greater: {:?}", cmp1);
            }
        }
    }
    
    #[test] 
    fn test_simple_stability() {
        let kbo = KBO::new(KBOConfig::default());
        
        // Even simpler: b > a
        let a = Term::Constant(Constant { name: "a".to_string() });
        let b = Term::Constant(Constant { name: "b".to_string() });
        
        let cmp = kbo.compare(&b, &a);
        assert_eq!(cmp, TermOrdering::Greater, "b should be > a");
        
        // Now with variables: X > a should be Incomparable or handle properly
        let x = Term::Variable(Variable { name: "X".to_string() });
        let cmp_var = kbo.compare(&x, &a);
        println!("X vs a: {:?}", cmp_var);
    }
    
    #[test]
    fn test_stability_with_equal_weight() {
        let kbo = KBO::new(KBOConfig::default());
        
        // Test case where weight is equal and lexicographic comparison matters
        // f(X, a) vs f(Y, b) where X, Y are variables
        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        let a = Term::Constant(Constant { name: "a".to_string() });
        let b = Term::Constant(Constant { name: "b".to_string() });
        let f = FunctionSymbol { name: "f".to_string(), arity: 2 };
        
        let fx_a = Term::Function(f.clone(), vec![x.clone(), a.clone()]);
        let fy_b = Term::Function(f.clone(), vec![y.clone(), b.clone()]);
        
        // Compare f(X, a) vs f(Y, b) - should be incomparable due to different variables
        let cmp1 = kbo.compare(&fx_a, &fy_b);
        println!("f(X, a) vs f(Y, b): {:?}", cmp1);
        
        // Now with substitution X -> c, Y -> d
        let c = Term::Constant(Constant { name: "c".to_string() });
        let d = Term::Constant(Constant { name: "d".to_string() });
        let mut subst = Substitution::new();
        subst.insert(Variable { name: "X".to_string() }, c.clone());
        subst.insert(Variable { name: "Y".to_string() }, d.clone());
        
        let fx_a_sigma = fx_a.apply_substitution(&subst);
        let fy_b_sigma = fy_b.apply_substitution(&subst);
        
        println!("After substitution:");
        println!("f(X, a)σ = {}", fx_a_sigma);
        println!("f(Y, b)σ = {}", fy_b_sigma);
        
        let cmp2 = kbo.compare(&fx_a_sigma, &fy_b_sigma);
        println!("f(c, a) vs f(d, b): {:?}", cmp2);
        
        // Test problematic case: g(X) vs f(a, b)
        let g = FunctionSymbol { name: "g".to_string(), arity: 1 };
        let gx = Term::Function(g.clone(), vec![x.clone()]);
        let fab = Term::Function(f.clone(), vec![a.clone(), b.clone()]);
        
        let cmp_problem = kbo.compare(&gx, &fab);
        println!("\ng(X) vs f(a, b): {:?}", cmp_problem);
        
        // This is where the issue shows: if they have equal weight,
        // the lexicographic comparison will make g(X) incomparable with f(a,b)
        // because X is incomparable with any non-variable term
    }
    
    #[test]
    fn test_correct_variable_ordering() {
        let kbo = KBO::new(KBOConfig::default());
        
        // Test 1: Different variables are incomparable in KBO
        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        
        assert_eq!(kbo.compare(&x, &y), TermOrdering::Incomparable, "X and Y are incomparable");
        assert_eq!(kbo.compare(&y, &x), TermOrdering::Incomparable, "Y and X are incomparable");
        
        // Test 2: Constant vs variable - incomparable due to variable condition
        let a = Term::Constant(Constant { name: "a".to_string() });
        assert_eq!(kbo.compare(&a, &x), TermOrdering::Incomparable, "a and X are incomparable");
        assert_eq!(kbo.compare(&x, &a), TermOrdering::Incomparable, "X and a are incomparable");
        
        // Test 3: f(X, Y) vs f(Y, X) should be incomparable (different variable sets)
        let f = FunctionSymbol { name: "f".to_string(), arity: 2 };
        let fxy = Term::Function(f.clone(), vec![x.clone(), y.clone()]);
        let fyx = Term::Function(f.clone(), vec![y.clone(), x.clone()]);
        
        // These are incomparable because they don't satisfy the variable condition
        assert_eq!(kbo.compare(&fxy, &fyx), TermOrdering::Incomparable);
        assert_eq!(kbo.compare(&fyx, &fxy), TermOrdering::Incomparable);
        
        // Test 4: Substitution should preserve certain properties
        let mut subst = Substitution::new();
        subst.insert(Variable { name: "X".to_string() }, a.clone());
        let b = Term::Constant(Constant { name: "b".to_string() });
        subst.insert(Variable { name: "Y".to_string() }, b.clone());
        
        let fxy_sigma = fxy.apply_substitution(&subst);
        let fyx_sigma = fyx.apply_substitution(&subst);
        
        // f(a, b) < f(b, a) since a < b
        assert_eq!(kbo.compare(&fxy_sigma, &fyx_sigma), TermOrdering::Less);
    }
    
    #[test]
    fn test_variable_condition() {
        let kbo = KBO::new(KBOConfig::default());
        
        // f(X) vs g(Y) - incomparable (different variables)
        let x = Term::Variable(Variable { name: "X".to_string() });
        let y = Term::Variable(Variable { name: "Y".to_string() });
        let f = FunctionSymbol { name: "f".to_string(), arity: 1 };
        let g = FunctionSymbol { name: "g".to_string(), arity: 1 };
        
        let fx = Term::Function(f.clone(), vec![x.clone()]);
        let gy = Term::Function(g.clone(), vec![y.clone()]);
        
        assert_eq!(kbo.compare(&fx, &gy), TermOrdering::Incomparable,
                   "f(X) and g(Y) should be incomparable (different variables)");
        
        // f(X) > X (variable condition satisfied)
        assert_eq!(kbo.compare(&fx, &x), TermOrdering::Greater,
                   "f(X) > X");
    }
}