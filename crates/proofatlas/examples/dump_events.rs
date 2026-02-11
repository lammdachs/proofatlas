use proofatlas::{
    parse_tptp_file, AgeWeightSelector, ClauseSelector, LiteralSelectionStrategy,
    ProverConfig, ProofResult, ProofAtlas, StateChange,
};
use std::collections::{HashMap, BTreeSet};
use std::time::Duration;

fn main() {
    let parsed = parse_tptp_file(
        "crates/proofatlas/tests/problems/right_identity.p", &[], None, None
    ).expect("Failed to parse");

    let config = ProverConfig {
        max_clauses: 10000,
        max_iterations: 100,
        timeout: Duration::from_secs(10),
        literal_selection: LiteralSelectionStrategy::Sel0,
        ..Default::default()
    };

    let selector: Box<dyn ClauseSelector> = Box::new(AgeWeightSelector::default());
    let mut prover = ProofAtlas::new(parsed.formula.clauses, config, selector, parsed.interner);
    let result = prover.prove();

    let events = prover.event_log();
    let interner = prover.interner();

    // Dump all events
    println!("=== RAW EVENT LOG ({} events) ===", events.len());
    for (i, event) in events.iter().enumerate() {
        match event {
            StateChange::Add(clause, rule, premises) => {
                let prem: Vec<usize> = premises.iter().map(|p| p.clause).collect();
                println!("{:3}: Add(id={:?}, rule={}, premises={:?}, clause={})",
                    i, clause.id, rule, prem, clause.display(interner));
            }
            StateChange::Simplify(idx, repl, rule, premises) => {
                let prem: Vec<usize> = premises.iter().map(|p| p.clause).collect();
                let repl_info = repl.as_ref().map(|c| format!("Some(id={:?}, {})", c.id, c.display(interner)));
                println!("{:3}: Simplify(idx={}, repl={}, rule={}, premises={:?})",
                    i, idx, repl_info.unwrap_or("None".into()), rule, prem);
            }
            StateChange::Transfer(idx) => {
                println!("{:3}: Transfer({})", i, idx);
            }
            StateChange::Activate(idx) => {
                println!("{:3}: Activate({})", i, idx);
            }
        }
    }

    let status = match &result {
        ProofResult::Proof { .. } => "proof",
        ProofResult::Saturated => "saturated",
        ProofResult::ResourceLimit => "resource_limit",
    };
    println!("\nResult: {}", status);

    // === Simulate the WASM trace conversion (events_to_js_value) ===
    println!("\n=== WASM TRACE CONVERSION ===");

    // First pass: collect clauses
    let mut clauses: HashMap<usize, String> = HashMap::new();
    let mut initial_clause_count = 0;
    for event in events {
        match event {
            StateChange::Add(clause, rule_name, _) => {
                if let Some(idx) = clause.id {
                    clauses.insert(idx, clause.display(interner).to_string());
                    if rule_name == "Input" {
                        initial_clause_count = initial_clause_count.max(idx + 1);
                    }
                }
            }
            StateChange::Simplify(_, Some(clause), _, _) => {
                if let Some(idx) = clause.id {
                    clauses.insert(idx, clause.display(interner).to_string());
                }
            }
            _ => {}
        }
    }
    println!("Initial clauses: {}", initial_clause_count);
    println!("Total clauses in map: {}", clauses.len());

    // Second pass: build iterations (mirrors WASM logic)
    #[derive(Debug)]
    struct TraceEvent {
        clause_idx: usize,
        rule: String,
        premises: Vec<usize>,
    }
    #[derive(Debug)]
    struct Iteration {
        simplification: Vec<TraceEvent>,
        selection: Option<TraceEvent>,
        generation: Vec<TraceEvent>,
    }

    let mut iterations: Vec<Iteration> = Vec::new();
    let mut current_simplification: Vec<TraceEvent> = Vec::new();
    let mut current_generation: Vec<TraceEvent> = Vec::new();
    let mut current_selection: Option<TraceEvent> = None;
    let mut in_generation_phase = false;

    let flush = |iterations: &mut Vec<Iteration>,
                 simplification: &mut Vec<TraceEvent>,
                 selection: &mut Option<TraceEvent>,
                 generation: &mut Vec<TraceEvent>| {
        if selection.is_some() || !simplification.is_empty() || !generation.is_empty() {
            iterations.push(Iteration {
                simplification: std::mem::take(simplification),
                selection: selection.take(),
                generation: std::mem::take(generation),
            });
        }
    };

    for event in events {
        match event {
            StateChange::Add(clause, rule_name, premises) => {
                if let Some(idx) = clause.id {
                    let premise_indices: Vec<usize> = premises.iter().map(|p| p.clause).collect();
                    if rule_name == "Input" { continue; }
                    current_generation.push(TraceEvent {
                        clause_idx: idx,
                        rule: rule_name.clone(),
                        premises: premise_indices,
                    });
                }
            }
            StateChange::Simplify(clause_idx, replacement, rule_name, premises) => {
                if in_generation_phase {
                    flush(&mut iterations, &mut current_simplification, &mut current_selection, &mut current_generation);
                    in_generation_phase = false;
                }
                let premise_indices: Vec<usize> = premises.iter().map(|p| p.clause).collect();

                if let Some(repl) = replacement {
                    let repl_idx = repl.id.unwrap_or(0);
                    current_simplification.push(TraceEvent {
                        clause_idx: repl_idx,
                        rule: rule_name.clone(),
                        premises: premise_indices,
                    });
                    current_simplification.push(TraceEvent {
                        clause_idx: *clause_idx,
                        rule: format!("{}Deletion", rule_name),
                        premises: vec![],
                    });
                } else {
                    let rule = match rule_name.as_str() {
                        "Tautology" => "TautologyDeletion",
                        "Subsumption" => "SubsumptionDeletion",
                        _ => "SubsumptionDeletion",
                    };
                    current_simplification.push(TraceEvent {
                        clause_idx: *clause_idx,
                        rule: rule.to_string(),
                        premises: vec![],
                    });
                }
            }
            StateChange::Transfer(clause_idx) => {
                if in_generation_phase {
                    flush(&mut iterations, &mut current_simplification, &mut current_selection, &mut current_generation);
                    in_generation_phase = false;
                }
                current_simplification.push(TraceEvent {
                    clause_idx: *clause_idx,
                    rule: "Transfer".to_string(),
                    premises: vec![],
                });
            }
            StateChange::Activate(clause_idx) => {
                if in_generation_phase {
                    flush(&mut iterations, &mut current_simplification, &mut current_selection, &mut current_generation);
                }
                current_selection = Some(TraceEvent {
                    clause_idx: *clause_idx,
                    rule: "GivenClauseSelection".to_string(),
                    premises: vec![],
                });
                in_generation_phase = true;
            }
        }
    }

    // Flush remaining
    if !current_simplification.is_empty() || !current_generation.is_empty() || current_selection.is_some() {
        iterations.push(Iteration {
            simplification: current_simplification,
            selection: current_selection,
            generation: current_generation,
        });
    }

    println!("Total iterations: {}\n", iterations.len());

    // === Replay the trace like JS ProofInspector does ===
    println!("=== JS PROOF INSPECTOR REPLAY ===");

    let mut new_clauses: BTreeSet<usize> = BTreeSet::new();
    let mut unprocessed: BTreeSet<usize> = BTreeSet::new();
    let mut processed: BTreeSet<usize> = BTreeSet::new();

    // Initial clauses start in N
    for i in 0..initial_clause_count {
        new_clauses.insert(i);
    }
    println!("After init: N={:?}, U={:?}, P={:?}", new_clauses, unprocessed, processed);

    for (iter_num, iter) in iterations.iter().enumerate() {
        println!("\n--- Iteration {} ---", iter_num);
        let flat_events: Vec<&TraceEvent> = iter.simplification.iter()
            .chain(iter.selection.iter())
            .chain(iter.generation.iter())
            .collect();

        for ev in &flat_events {
            let rule = &ev.rule;
            match rule.as_str() {
                "TautologyDeletion" | "SubsumptionDeletion" | "DemodulationDeletion"
                | "ForwardSubsumptionDeletion" | "BackwardSubsumptionDeletion"
                | "ForwardDemodulation" => {
                    new_clauses.remove(&ev.clause_idx);
                    unprocessed.remove(&ev.clause_idx);
                    processed.remove(&ev.clause_idx);
                }
                "Transfer" => {
                    new_clauses.remove(&ev.clause_idx);
                    unprocessed.insert(ev.clause_idx);
                }
                "Demodulation" => {
                    new_clauses.insert(ev.clause_idx);
                }
                "GivenClauseSelection" => {
                    unprocessed.remove(&ev.clause_idx);
                    processed.insert(ev.clause_idx);
                }
                _ => {
                    // Generation: add to N
                    new_clauses.insert(ev.clause_idx);
                }
            }
            println!("  {:30} [{}] → N={:?} U={:?} P={:?}",
                ev.rule, ev.clause_idx, new_clauses, unprocessed, processed);
        }
    }

    // Now also replay using direct event log (the ground truth)
    println!("\n\n=== DIRECT EVENT LOG REPLAY (ground truth) ===");
    let mut n: Vec<usize> = Vec::new();
    let mut u: BTreeSet<usize> = BTreeSet::new();
    let mut p: BTreeSet<usize> = BTreeSet::new();

    for (i, event) in events.iter().enumerate() {
        match event {
            StateChange::Add(clause, _, _) => {
                if let Some(idx) = clause.id {
                    n.push(idx);
                }
            }
            StateChange::Simplify(clause_idx, replacement, _, _) => {
                // Remove from whichever set
                if n.last() == Some(clause_idx) {
                    n.pop();
                } else if u.remove(clause_idx) {
                    // removed from U
                } else {
                    p.remove(clause_idx);
                }
                // Add replacement to N if any
                if let Some(repl) = replacement {
                    if let Some(idx) = repl.id {
                        n.push(idx);
                    }
                }
            }
            StateChange::Transfer(idx) => {
                if n.last() == Some(idx) {
                    n.pop();
                }
                u.insert(*idx);
            }
            StateChange::Activate(idx) => {
                u.remove(idx);
                p.insert(*idx);
            }
        }
        let n_set: BTreeSet<usize> = n.iter().cloned().collect();
        println!("{:3}: N={:?} U={:?} P={:?}", i, n_set, u, p);
    }
}
