//! Debug what literals get selected by different strategies

use proofatlas::{parse_tptp_file, LiteralSelectionStrategy};
use proofatlas::selection::{LiteralSelector, SelectAll, SelectMaxWeight, SelectLargestNegative};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <tptp_file> [--include <dir>]", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];
    let mut include_dirs: Vec<&str> = Vec::new();

    // Parse include directories
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--include" && i + 1 < args.len() {
            include_dirs.push(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    // Parse the TPTP file
    let cnf_formula = match parse_tptp_file(filename, &include_dirs) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    println!("Parsed {} clauses from '{}'", cnf_formula.clauses.len(), filename);
    println!();

    let select_all = SelectAll;
    let select_max_weight = SelectMaxWeight::new();
    let select_largest_neg = SelectLargestNegative::new();

    // Analyze first 10 clauses
    for (idx, clause) in cnf_formula.clauses.iter().take(10).enumerate() {
        println!("Clause [{}]: {}", idx, clause);
        println!("  Literals: {}", clause.literals.len());

        let all_selected = select_all.select(clause);
        let max_weight_selected = select_max_weight.select(clause);
        let largest_neg_selected = select_largest_neg.select(clause);

        println!("  SelectAll: {} literals", all_selected.len());
        println!("  SelectMaxWeight: {} literals ({}%)",
                 max_weight_selected.len(),
                 (max_weight_selected.len() * 100) / clause.literals.len());
        println!("  SelectLargestNegative: {} literals ({}%)",
                 largest_neg_selected.len(),
                 (largest_neg_selected.len() * 100) / clause.literals.len());

        // Show weights
        print!("  Literal weights: [");
        for (i, lit) in clause.literals.iter().enumerate() {
            let weight = 1 + lit.atom.args.iter()
                .map(|term| count_symbols(term))
                .sum::<usize>();
            print!("{}", weight);
            if i < clause.literals.len() - 1 {
                print!(", ");
            }
        }
        println!("]");
        println!();
    }
}

fn count_symbols(term: &proofatlas::core::Term) -> usize {
    use proofatlas::core::Term;
    match term {
        Term::Variable(_) => 1,
        Term::Constant(_) => 1,
        Term::Function(_, args) => {
            1 + args.iter().map(|t| count_symbols(t)).sum::<usize>()
        }
    }
}
