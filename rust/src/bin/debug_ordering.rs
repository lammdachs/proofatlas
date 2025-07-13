//! Debug ordering constraints

use proofatlas::core::{Term, Constant, FunctionSymbol, KBO, KBOConfig, TermOrdering};

fn main() {
    let kbo = KBO::new(KBOConfig::default());
    
    // Test constants
    let a = Term::Constant(Constant { name: "a".to_string() });
    let b = Term::Constant(Constant { name: "b".to_string() });
    let c = Term::Constant(Constant { name: "c".to_string() });
    
    println!("Constants:");
    println!("  a vs b: {:?}", kbo.compare(&a, &b));
    println!("  b vs a: {:?}", kbo.compare(&b, &a));
    println!("  a vs c: {:?}", kbo.compare(&a, &c));
    println!("  b vs c: {:?}", kbo.compare(&b, &c));
    
    // Test functions
    let f = FunctionSymbol { name: "f".to_string(), arity: 1 };
    let g = FunctionSymbol { name: "g".to_string(), arity: 1 };
    let h = FunctionSymbol { name: "h".to_string(), arity: 1 };
    
    let fa = Term::Function(f.clone(), vec![a.clone()]);
    let gb = Term::Function(g.clone(), vec![b.clone()]);
    let hc = Term::Function(h.clone(), vec![c.clone()]);
    
    println!("\nFunctions:");
    println!("  f(a) vs g(b): {:?}", kbo.compare(&fa, &gb));
    println!("  g(b) vs f(a): {:?}", kbo.compare(&gb, &fa));
    println!("  f(a) vs h(c): {:?}", kbo.compare(&fa, &hc));
    println!("  g(b) vs h(c): {:?}", kbo.compare(&gb, &hc));
}