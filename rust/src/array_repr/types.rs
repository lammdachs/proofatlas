//! Core array types for logical structures

use crate::array_repr::symbol_table::SymbolTable;

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

/// Edge types in the graph representation
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    HasArgument = 0,    // Function/predicate → term
    HasLiteral = 1,     // Clause → literal
    HasPredicate = 2,   // Literal → predicate
}

/// Array-based representation of a logical problem
#[derive(Debug)]
pub struct ArrayProblem {
    // Node data
    pub node_types: Vec<NodeType>,      // Type of each node
    pub node_symbols: Vec<u32>,         // Symbol table index
    pub node_polarities: Vec<i8>,       // For literals: 1 (positive), -1 (negative), 0 (n/a)
    pub node_arities: Vec<u32>,         // Number of arguments/children
    
    // Edge data (CSR format)
    pub edge_row_offsets: Vec<usize>,   // Start index for each node's edges
    pub edge_col_indices: Vec<u32>,     // Target node indices
    pub edge_types: Vec<EdgeType>,      // Type of each edge
    
    // Hierarchical structure
    pub clause_boundaries: Vec<usize>,   // Start/end indices for clauses
    pub literal_boundaries: Vec<usize>,  // Start/end indices for literals
    
    // Symbol table
    pub symbols: SymbolTable,
    
    // Metadata
    pub num_nodes: usize,
    pub num_clauses: usize,
    pub num_literals: usize,
}

impl ArrayProblem {
    /// Create a new empty array problem
    pub fn new() -> Self {
        ArrayProblem {
            node_types: Vec::new(),
            node_symbols: Vec::new(),
            node_polarities: Vec::new(),
            node_arities: Vec::new(),
            edge_row_offsets: vec![0], // Start with one offset
            edge_col_indices: Vec::new(),
            edge_types: Vec::new(),
            clause_boundaries: vec![0],
            literal_boundaries: vec![0],
            symbols: SymbolTable::new(),
            num_nodes: 0,
            num_clauses: 0,
            num_literals: 0,
        }
    }
    
    /// Get the node index for a clause
    pub fn clause_node(&self, clause_idx: usize) -> Option<usize> {
        if clause_idx < self.num_clauses {
            Some(self.clause_boundaries[clause_idx])
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
        let mut literals = Vec::new();
        
        if let Some((start, end)) = self.clause_node_range(clause_idx) {
            for node in start..end {
                if self.node_types[node] == NodeType::Literal {
                    literals.push(node);
                }
            }
        }
        
        literals
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
#[path = "types_tests.rs"]
mod tests;