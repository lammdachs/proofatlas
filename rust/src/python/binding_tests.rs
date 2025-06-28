//! Tests for Python bindings

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    
    #[test]
    fn test_problem_binding() {
        Python::with_gil(|py| {
            // Create a Problem through Python binding
            let problem_class = py.import("proofatlas_rust.core")
                .unwrap()
                .getattr("Problem")
                .unwrap();
            
            let problem = problem_class.call0().unwrap();
            
            // Test __len__
            let len: usize = problem.len().unwrap();
            assert_eq!(len, 0);
            
            // Test __repr__
            let repr: String = problem.repr().unwrap().extract().unwrap();
            assert!(repr.contains("Problem"));
            
            // Test properties
            let clauses = problem.getattr("clauses").unwrap();
            assert!(clauses.is_instance_of::<PyList>());
            
            let indices = problem.getattr("conjecture_indices").unwrap();
            assert!(indices.is_instance_of::<PyList>());
        });
    }
    
    #[test] 
    fn test_proofstate_binding() {
        Python::with_gil(|py| {
            let proofstate_class = py.import("proofatlas_rust.proofs")
                .unwrap()
                .getattr("ProofState")
                .unwrap();
            
            // Create empty lists for processed/unprocessed
            let empty_list = PyList::empty(py);
            let state = proofstate_class.call1((empty_list, empty_list)).unwrap();
            
            // Test properties
            let processed = state.getattr("processed").unwrap();
            assert!(processed.is_instance_of::<PyList>());
            
            let contains_empty: bool = state.getattr("contains_empty_clause")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(contains_empty, false);
        });
    }
    
    #[test]
    fn test_proof_binding() {
        Python::with_gil(|py| {
            let proof_class = py.import("proofatlas_rust.proofs")
                .unwrap()
                .getattr("Proof")
                .unwrap();
            
            let proof = proof_class.call0().unwrap();
            
            // Test length
            let length: usize = proof.getattr("length").unwrap().extract().unwrap();
            assert_eq!(length, 0);
            
            // Test steps
            let steps = proof.getattr("steps").unwrap();
            assert!(steps.is_instance_of::<PyList>());
            
            // Test methods
            let found: bool = proof.call_method0("found_contradiction")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(found, false);
        });
    }
}