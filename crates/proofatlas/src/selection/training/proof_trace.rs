//! Extract training data from completed proofs

use crate::state::ProofStep;
use crate::state::clause_indices;
use std::collections::HashSet;

/// Training example: a clause with its label
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Index of the clause
    pub clause_idx: usize,
    /// Label: 1 if clause is in proof, 0 otherwise
    pub label: u8,
}

/// Extract training data from proof steps
pub fn extract_training_data(steps: &[ProofStep], empty_clause_idx: usize) -> Vec<TrainingExample> {
    // Find all clauses that are part of the proof derivation
    let proof_clauses = extract_proof_dag(steps, empty_clause_idx);

    // All proof steps are real derivations (no trace-only events to filter)
    let all_clauses: HashSet<usize> = steps.iter().map(|s| s.clause_idx).collect();

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
fn extract_proof_dag(steps: &[ProofStep], empty_clause_idx: usize) -> HashSet<usize> {
    let mut proof_clauses = HashSet::new();
    let mut to_visit = vec![empty_clause_idx];

    // Backward search from empty clause
    while let Some(clause_idx) = to_visit.pop() {
        if proof_clauses.contains(&clause_idx) {
            continue; // Already visited
        }
        proof_clauses.insert(clause_idx);

        // Find the step that derived this clause
        if let Some(step) = steps.iter().find(|s| s.clause_idx == clause_idx) {
            // Add premises (parent clauses) to the search
            to_visit.extend(clause_indices(&step.premises));
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
pub fn compute_proof_statistics(steps: &[ProofStep], empty_clause_idx: usize) -> ProofStatistics {
    let proof_clauses = extract_proof_dag(steps, empty_clause_idx);
    let total_clauses = steps.len();
    let proof_clause_count = proof_clauses.len();

    ProofStatistics {
        total_clauses,
        proof_clauses: proof_clause_count,
        proof_percentage: (proof_clause_count as f64 / total_clauses as f64) * 100.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Clause, Position};

    fn make_step(clause_idx: usize, premises: Vec<usize>) -> ProofStep {
        let positions: Vec<Position> = premises.iter().map(|&p| Position::clause(p)).collect();
        let rule_name = if premises.is_empty() {
            "Input".into()
        } else if premises.len() == 1 {
            "Factoring".into()
        } else {
            "Resolution".into()
        };
        ProofStep {
            clause_idx,
            rule_name,
            premises: positions,
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

        let proof_clauses = extract_proof_dag(&steps, 2);

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

        let examples = extract_training_data(&steps, 3);

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

        let stats = compute_proof_statistics(&steps, 3);

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

        let proof_clauses = extract_proof_dag(&steps, 6);

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
