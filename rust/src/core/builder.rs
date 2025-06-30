//! Graph builder that collects edges and converts to CSR format

use super::{Problem, NodeType, CapacityError};
use std::collections::BTreeMap;

/// Builder that collects edges in a flexible format before converting to CSR
pub struct Builder<'a> {
    pub problem: &'a mut Problem,
    edges: BTreeMap<usize, Vec<usize>>,  // from -> [to1, to2, ...]
}

impl<'a> Builder<'a> {
    /// Create a new graph builder
    pub fn new(problem: &'a mut Problem) -> Self {
        Builder {
            problem,
            edges: BTreeMap::new(),
        }
    }
    
    /// Add a node to the problem
    pub fn add_node(&mut self, node_type: NodeType, symbol: &str, polarity: i8, arity: u32) -> Result<usize, CapacityError> {
        let node_idx = self.problem.num_nodes;
        
        // Check capacity
        if node_idx >= self.problem.max_nodes {
            return Err(CapacityError {
                resource: "nodes",
                requested: node_idx + 1,
                capacity: self.problem.max_nodes,
            });
        }
        
        // Set node data in pre-allocated arrays
        self.problem.node_types[node_idx] = node_type as u8;
        
        let symbol_id = if symbol.is_empty() {
            0
        } else {
            self.problem.symbols.intern(symbol)
        };
        self.problem.node_symbols[node_idx] = symbol_id;
        
        self.problem.node_polarities[node_idx] = polarity;
        self.problem.node_arities[node_idx] = arity;
        self.problem.node_selected[node_idx] = false;
        
        self.problem.num_nodes += 1;
        Ok(node_idx)
    }
    
    /// Add an edge between two nodes (can be called in any order)
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), CapacityError> {
        self.edges.entry(from).or_insert_with(Vec::new).push(to);
        Ok(())
    }
    
    /// Finalize the graph by converting edges to CSR format
    pub fn finalize(self) -> Result<(), CapacityError> {
        // Count total NEW edges
        let new_edges: usize = self.edges.values().map(|v| v.len()).sum();
        
        // Check edge capacity
        if self.problem.num_edges + new_edges > self.problem.max_edges {
            return Err(CapacityError {
                resource: "edges",
                requested: self.problem.num_edges + new_edges,
                capacity: self.problem.max_edges,
            });
        }
        
        // Start from the current edge count
        let mut edge_offset = self.problem.num_edges;
        
        // Only process new edges from nodes that have edges in our map
        let min_node = self.edges.keys().min().copied().unwrap_or(self.problem.num_nodes);
        
        // Update row offsets only for nodes that might have new edges
        for node_idx in min_node..=self.problem.num_nodes {
            self.problem.edge_row_offsets[node_idx] = edge_offset;
            
            if let Some(neighbors) = self.edges.get(&node_idx) {
                for &neighbor in neighbors {
                    if edge_offset < self.problem.max_edges {
                        self.problem.edge_col_indices[edge_offset] = neighbor as u32;
                        edge_offset += 1;
                    }
                }
            }
        }
        
        self.problem.num_edges = edge_offset;
        Ok(())
    }
}