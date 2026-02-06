//! Extract training data from completed proofs

use crate::inference::Proof;
use crate::saturation::{extract_proof_from_events, StateChange, EventLog};
use std::collections::HashSet;

/// Training example: a clause with its label
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Index of the clause
    pub clause_idx: usize,
    /// Label: 1 if clause is in proof, 0 otherwise
    pub label: u8,
}

/// Extract training data from a proof
pub fn extract_training_data(proof: &Proof) -> Vec<TrainingExample> {
    // Find all clauses that are part of the proof derivation
    let proof_clauses = extract_proof_dag(proof);

    // All proof steps are real derivations (no trace-only events to filter)
    let all_clauses: HashSet<usize> = proof.steps.iter().map(|s| s.clause_idx).collect();

    // Create training examples
    let mut examples = Vec::new();
    for &clause_idx in &all_clauses {
        examples.push(TrainingExample {
            clause_idx,
            label: if proof_clauses.contains(&clause_idx) { 1 } else { 0 },
        });
    }

    examples
}

/// Extract the set of clause indices that are part of the proof DAG
/// (clauses that led to the empty clause)
fn extract_proof_dag(proof: &Proof) -> HashSet<usize> {
    let mut proof_clauses = HashSet::new();
    let mut to_visit = vec![proof.empty_clause_idx];

    // Backward search from empty clause
    while let Some(clause_idx) = to_visit.pop() {
        if proof_clauses.contains(&clause_idx) {
            continue; // Already visited
        }
        proof_clauses.insert(clause_idx);

        // Find the step that derived this clause
        if let Some(step) = proof.steps.iter().find(|s| s.clause_idx == clause_idx) {
            // Add premises (parent clauses) to the search
            to_visit.extend(step.derivation.clause_indices());
        }
    }

    proof_clauses
}

/// Statistics about proof and training data
#[derive(Debug, Clone)]
pub struct ProofStatistics {
    /// Total number of clauses generated
    pub total_clauses: usize,
    /// Number of clauses in the proof DAG
    pub proof_clauses: usize,
    /// Percentage of clauses in proof
    pub proof_percentage: f64,
}

/// Compute statistics about a proof
pub fn compute_proof_statistics(proof: &Proof) -> ProofStatistics {
    let proof_clauses = extract_proof_dag(proof);
    let total_clauses = proof.steps.len();
    let proof_clause_count = proof_clauses.len();

    ProofStatistics {
        total_clauses,
        proof_clauses: proof_clause_count,
        proof_percentage: (proof_clause_count as f64 / total_clauses as f64) * 100.0,
    }
}

// =============================================================================
// Training Data Extraction via Event Log Replay
// =============================================================================

/// A training example capturing clause selection context
#[derive(Debug, Clone)]
pub struct SelectionTrainingExample {
    /// The clause that was selected
    pub selected_idx: usize,
    /// All candidate clauses that were available at selection time
    pub candidates: Vec<usize>,
    /// Whether the selected clause is in the proof (computed after proof completion)
    pub selected_in_proof: bool,
    /// For each candidate, whether it's in the proof
    pub candidate_labels: Vec<bool>,
}

/// Extract training examples from event log by replaying and tracking selections
pub fn extract_training_from_events(events: &EventLog) -> Vec<SelectionTrainingExample> {
    let mut examples = Vec::new();
    let mut u: HashSet<usize> = HashSet::new();
    let mut n: HashSet<usize> = HashSet::new();

    // First pass: replay to collect selection contexts
    struct SelectionContext {
        selected_idx: usize,
        candidates: Vec<usize>,
    }
    let mut contexts = Vec::new();

    for event in events {
        match event {
            StateChange::Add { clause, derivation: _ } => {
                if let Some(idx) = clause.id {
                    n.insert(idx);
                }
            }
            StateChange::Delete { clause_idx, rule_name: _ } => {
                // Remove from whichever set contains it
                if !n.remove(clause_idx) {
                    u.remove(clause_idx);
                }
                // Note: P removals are also handled implicitly
            }
            StateChange::Transfer { clause_idx } => {
                n.remove(clause_idx);
                u.insert(*clause_idx);
            }
            StateChange::Activate { clause_idx } => {
                // This is the given clause selection - N should be empty at this point
                if n.is_empty() {
                    let candidates: Vec<usize> = u.iter().copied().collect();
                    contexts.push(SelectionContext {
                        selected_idx: *clause_idx,
                        candidates,
                    });
                }
                // Implicit: removed from U, added to P
                u.remove(clause_idx);
            }
        }
    }

    // Extract proof clauses
    let proof_clause_set: HashSet<usize> = extract_proof_from_events(events)
        .map(|v| v.into_iter().collect())
        .unwrap_or_default();

    // Build training examples
    for ctx in contexts {
        let selected_in_proof = proof_clause_set.contains(&ctx.selected_idx);
        let candidate_labels: Vec<bool> = ctx
            .candidates
            .iter()
            .map(|&c| proof_clause_set.contains(&c))
            .collect();

        examples.push(SelectionTrainingExample {
            selected_idx: ctx.selected_idx,
            candidates: ctx.candidates,
            selected_in_proof,
            candidate_labels,
        });
    }

    examples
}

/// Extract clause-level training data with proof membership labels from event log
pub fn extract_clause_labels_from_events(events: &EventLog) -> Vec<TrainingExample> {
    // Get proof clause set
    let proof_clause_set: HashSet<usize> = extract_proof_from_events(events)
        .map(|v| v.into_iter().collect())
        .unwrap_or_default();

    // Get all clauses that were ever added
    let mut all_clauses = HashSet::new();
    for event in events {
        if let StateChange::Add { clause, derivation: _ } = event {
            if let Some(idx) = clause.id {
                all_clauses.insert(idx);
            }
        }
    }

    // Build examples
    all_clauses
        .into_iter()
        .map(|clause_idx| TrainingExample {
            clause_idx,
            label: if proof_clause_set.contains(&clause_idx) {
                1
            } else {
                0
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fol::{Clause, Position};
    use crate::inference::{Derivation, ProofStep};

    fn make_step(clause_idx: usize, premises: Vec<usize>) -> ProofStep {
        let positions: Vec<Position> = premises.iter().map(|&p| Position::clause(p)).collect();
        let derivation = if premises.is_empty() {
            Derivation::input()
        } else if premises.len() == 1 {
            Derivation { rule_name: "Factoring".into(), premises: positions }
        } else {
            Derivation { rule_name: "Resolution".into(), premises: positions }
        };
        ProofStep {
            clause_idx,
            derivation,
            conclusion: Clause::new(vec![]), // Empty for simplicity
        }
    }

    #[test]
    fn test_extract_proof_dag() {
        // Create a simple proof:
        // Step 0: input
        // Step 1: input
        // Step 2: resolution from 0 and 1 (empty clause)

        let steps = vec![
            make_step(0, vec![]),
            make_step(1, vec![]),
            make_step(2, vec![0, 1]),
        ];

        let proof = Proof {
            steps,
            empty_clause_idx: 2,
            all_clauses: vec![],  // Not used for proof DAG extraction
        };

        let proof_clauses = extract_proof_dag(&proof);

        // All clauses should be in the proof
        assert_eq!(proof_clauses.len(), 3);
        assert!(proof_clauses.contains(&0));
        assert!(proof_clauses.contains(&1));
        assert!(proof_clauses.contains(&2));
    }

    #[test]
    fn test_extract_training_data_with_unused_clause() {
        // Create proof with an unused clause:
        // Step 0: input
        // Step 1: input
        // Step 2: input (unused)
        // Step 3: resolution from 0 and 1 (empty clause)

        let steps = vec![
            make_step(0, vec![]),
            make_step(1, vec![]),
            make_step(2, vec![]),       // Unused
            make_step(3, vec![0, 1]),    // Uses 0 and 1, not 2
        ];

        let proof = Proof {
            steps,
            empty_clause_idx: 3,
            all_clauses: vec![],  // Not used for this test
        };

        let examples = extract_training_data(&proof);

        // Should have 4 examples
        assert_eq!(examples.len(), 4);

        // Check labels
        let labels: std::collections::HashMap<usize, u8> =
            examples.iter().map(|e| (e.clause_idx, e.label)).collect();

        assert_eq!(labels[&0], 1); // In proof
        assert_eq!(labels[&1], 1); // In proof
        assert_eq!(labels[&2], 0); // NOT in proof (unused)
        assert_eq!(labels[&3], 1); // In proof (empty clause)
    }

    #[test]
    fn test_compute_statistics() {
        // Create proof with unused clauses
        let steps = vec![
            make_step(0, vec![]),
            make_step(1, vec![]),
            make_step(2, vec![]),       // Unused
            make_step(3, vec![0, 1]),    // Empty clause
        ];

        let proof = Proof {
            steps,
            empty_clause_idx: 3,
            all_clauses: vec![],  // Not used for statistics
        };

        let stats = compute_proof_statistics(&proof);

        assert_eq!(stats.total_clauses, 4);
        assert_eq!(stats.proof_clauses, 3); // 0, 1, 3 (not 2)
        assert!((stats.proof_percentage - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_complex_proof_dag() {
        // Create a more complex proof:
        // Step 0, 1, 2: inputs
        // Step 3: from 0, 1
        // Step 4: from 2 (unused branch)
        // Step 5: from 3, 2
        // Step 6: from 5 (empty clause)

        let steps = vec![
            make_step(0, vec![]),
            make_step(1, vec![]),
            make_step(2, vec![]),
            make_step(3, vec![0, 1]),
            make_step(4, vec![2]),       // Unused
            make_step(5, vec![3, 2]),
            make_step(6, vec![5]),       // Empty clause
        ];

        let proof = Proof {
            steps,
            empty_clause_idx: 6,
            all_clauses: vec![],  // Not used for proof DAG
        };

        let proof_clauses = extract_proof_dag(&proof);

        // Should include: 0, 1, 2, 3, 5, 6 (not 4)
        assert_eq!(proof_clauses.len(), 6);
        assert!(proof_clauses.contains(&0));
        assert!(proof_clauses.contains(&1));
        assert!(proof_clauses.contains(&2));
        assert!(proof_clauses.contains(&3));
        assert!(!proof_clauses.contains(&4)); // Not in proof
        assert!(proof_clauses.contains(&5));
        assert!(proof_clauses.contains(&6));
    }
}
