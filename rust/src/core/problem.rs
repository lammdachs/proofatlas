//! Core array types for logical structures

use crate::core::symbol_table::SymbolTable;
use std::fmt;

/// Error when capacity is exceeded
#[derive(Debug, Clone)]
pub struct CapacityError {
    pub resource: &'static str,
    pub requested: usize,
    pub capacity: usize,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Capacity exceeded for {}: requested {} but capacity is {}", 
               self.resource, self.requested, self.capacity)
    }
}

impl std::error::Error for CapacityError {}

/// Node types in the graph representation
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Variable = 0,
    Constant = 1,
    Function = 2,
    Predicate = 3,
    Literal = 4,
    Clause = 5,
}


/// Clause types for tracking origin
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseType {
    Axiom = 0,
    NegatedConjecture = 1,
    Derived = 2,
}

/// Array-based representation of a logical problem
/// 
/// Uses pre-allocated Box<[T]> arrays for stable memory addresses,
/// enabling zero-copy access from Python.
#[derive(Debug, Clone)]
pub struct Problem {
    // Node data - stored as primitives for zero-copy Python interface
    pub node_types: Box<[u8]>,         // Type of each node (0=Variable, 1=Constant, etc.)
    pub node_symbols: Box<[u32]>,      // Symbol table index
    pub node_polarities: Box<[i8]>,    // For literals: 1 (positive), -1 (negative), 0 (n/a)
    pub node_arities: Box<[u32]>,      // Number of arguments/children
    pub node_selected: Box<[bool]>,    // For literal selection
    
    // Edge data (CSR format)
    pub edge_row_offsets: Box<[usize]>, // Start index for each node's edges
    pub edge_col_indices: Box<[u32]>,   // Target node indices
    
    // Hierarchical structure
    pub clause_boundaries: Box<[usize]>, // Start/end indices for clauses
    pub clause_types: Box<[u8]>,        // Type of each clause
    pub literal_boundaries: Box<[usize]>, // Start/end indices for literals
    
    // Symbol table (remains as SymbolTable for flexibility)
    pub symbols: SymbolTable,
    
    // Metadata - tracks actual usage within pre-allocated arrays
    pub num_nodes: usize,
    pub num_clauses: usize,
    pub num_literals: usize,
    pub num_edges: usize,  // Track edge usage
    
    // Capacity tracking
    pub max_nodes: usize,
    pub max_clauses: usize,
    pub max_edges: usize,
}

impl Problem {
    /// Create a new array problem with default capacity
    pub fn new() -> Self {
        // Default capacities for small problems
        Self::with_capacity(1000, 100, 5000)
    }
    
    /// Create a new array problem with specified capacity
    pub fn with_capacity(max_nodes: usize, max_clauses: usize, max_edges: usize) -> Self {
        // Pre-allocate all arrays with zeros
        Problem {
            node_types: vec![0; max_nodes].into_boxed_slice(),
            node_symbols: vec![0; max_nodes].into_boxed_slice(),
            node_polarities: vec![0; max_nodes].into_boxed_slice(),
            node_arities: vec![0; max_nodes].into_boxed_slice(),
            node_selected: vec![false; max_nodes].into_boxed_slice(),
            
            edge_row_offsets: vec![0; max_nodes + 1].into_boxed_slice(),
            edge_col_indices: vec![0; max_edges].into_boxed_slice(),
            
            clause_boundaries: vec![0; max_clauses + 1].into_boxed_slice(),
            clause_types: vec![0; max_clauses].into_boxed_slice(),
            literal_boundaries: vec![0; max_nodes].into_boxed_slice(), // Worst case: each node is a literal
            
            symbols: SymbolTable::new(),
            
            num_nodes: 0,
            num_clauses: 0,
            num_literals: 0,
            num_edges: 0,
            
            max_nodes,
            max_clauses,
            max_edges,
        }
    }
    
    /// Get the node index for a clause
    pub fn clause_node(&self, clause_idx: usize) -> Option<usize> {
        if clause_idx < self.num_clauses {
            // The clause node is the first node in the clause's range
            let start = self.clause_boundaries[clause_idx];
            let end = self.clause_boundaries[clause_idx + 1];
            // Find the first node in this range that is a Clause type
            for i in start..end {
                if self.node_types[i] == NodeType::Clause as u8 {
                    return Some(i);
                }
            }
            None
        } else {
            None
        }
    }
    
    /// Get the range of nodes for a clause
    pub fn clause_node_range(&self, clause_idx: usize) -> Option<(usize, usize)> {
        if clause_idx < self.num_clauses {
            Some((self.clause_boundaries[clause_idx], self.clause_boundaries[clause_idx + 1]))
        } else {
            None
        }
    }
    
    /// Get literal nodes in a clause
    pub fn clause_literals(&self, clause_idx: usize) -> Vec<usize> {
        if let Some(clause_node) = self.clause_node(clause_idx) {
            // Get children of the clause node - these should be literals
            let children = self.node_children(clause_node);
            // Filter to ensure we only get literals (defensive programming)
            children.into_iter()
                .filter(|&node| self.node_types[node] == NodeType::Literal as u8)
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get children of a node using edge data
    pub fn node_children(&self, node_idx: usize) -> Vec<usize> {
        if node_idx >= self.num_nodes {
            return Vec::new();
        }
        
        let start = self.edge_row_offsets[node_idx];
        let end = self.edge_row_offsets[node_idx + 1];
        
        self.edge_col_indices[start..end]
            .iter()
            .map(|&idx| idx as usize)
            .collect()
    }
    
    
    /// Check if the problem contains the empty clause
    pub fn has_empty_clause(&self) -> bool {
        for clause_idx in 0..self.num_clauses {
            if self.clause_literals(clause_idx).is_empty() {
                return true;
            }
        }
        false
    }
}

/// Substitution represented as parallel arrays
#[derive(Debug, Clone)]
pub struct ArraySubstitution {
    pub var_indices: Vec<usize>,    // Variable node indices
    pub term_indices: Vec<usize>,   // What they map to
}

impl ArraySubstitution {
    pub fn new() -> Self {
        ArraySubstitution {
            var_indices: Vec::new(),
            term_indices: Vec::new(),
        }
    }
    
    /// Add a variable binding
    pub fn bind(&mut self, var_idx: usize, term_idx: usize) {
        self.var_indices.push(var_idx);
        self.term_indices.push(term_idx);
    }
    
    /// Look up a variable binding
    pub fn get(&self, var_idx: usize) -> Option<usize> {
        self.var_indices.iter()
            .position(|&v| v == var_idx)
            .map(|pos| self.term_indices[pos])
    }
    
    /// Clear all bindings
    pub fn clear(&mut self) {
        self.var_indices.clear();
        self.term_indices.clear();
    }
}

#[cfg(test)]
#[path = "problem_tests.rs"]
mod tests;