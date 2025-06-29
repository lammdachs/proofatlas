//! Builder for constructing array representations

use crate::core::logic::{Term, Literal, Clause, Predicate};
use crate::array_repr::types::{ArrayProblem, NodeType, EdgeType};

/// Builder for constructing array representations
pub struct ArrayBuilder<'a> {
    problem: &'a mut ArrayProblem,
}

impl<'a> ArrayBuilder<'a> {
    /// Create a new builder
    pub fn new(problem: &'a mut ArrayProblem) -> Self {
        ArrayBuilder { problem }
    }
    
    /// Add a clause and return its index
    pub fn add_clause(&mut self, clause: &Clause) -> usize {
        let clause_idx = self.problem.num_clauses;
        let clause_node = self.add_node(NodeType::Clause, "", 0, clause.literals.len() as u32);
        
        // Mark clause boundary
        self.problem.clause_boundaries.push(self.problem.num_nodes);
        
        // Add literals
        for literal in &clause.literals {
            self.add_literal(literal, clause_node);
        }
        
        self.problem.num_clauses += 1;
        clause_idx
    }
    
    /// Add a literal and connect it to its parent
    fn add_literal(&mut self, literal: &Literal, parent_node: usize) -> usize {
        let polarity = if literal.polarity { 1 } else { -1 };
        let lit_node = self.add_node(
            NodeType::Literal,
            "",
            polarity,
            1, // One child: the predicate
        );
        
        // Connect clause to literal
        self.add_edge(parent_node, lit_node, EdgeType::HasLiteral);
        
        // Add predicate
        self.add_predicate(&literal.predicate, lit_node);
        
        self.problem.num_literals += 1;
        lit_node
    }
    
    /// Add a predicate and connect it to its parent
    fn add_predicate(&mut self, predicate: &Predicate, parent_node: usize) -> usize {
        let symbol_id = self.problem.symbols.intern(&predicate.name);
        let pred_node = self.add_node(
            NodeType::Predicate,
            &predicate.name,
            0,
            predicate.args.len() as u32,
        );
        
        // Connect literal to predicate
        self.add_edge(parent_node, pred_node, EdgeType::HasPredicate);
        
        // Add arguments
        for (idx, term) in predicate.args.iter().enumerate() {
            self.add_term(term, pred_node, idx);
        }
        
        pred_node
    }
    
    /// Add a term and connect it to its parent
    fn add_term(&mut self, term: &Term, parent_node: usize, position: usize) -> usize {
        match term {
            Term::Variable(name) => {
                let symbol_id = self.problem.symbols.intern(name);
                let var_node = self.add_node(NodeType::Variable, name, 0, 0);
                self.add_edge(parent_node, var_node, EdgeType::HasArgument);
                var_node
            }
            Term::Constant(name) => {
                let symbol_id = self.problem.symbols.intern(name);
                let const_node = self.add_node(NodeType::Constant, name, 0, 0);
                self.add_edge(parent_node, const_node, EdgeType::HasArgument);
                const_node
            }
            Term::Function { name, args } => {
                let symbol_id = self.problem.symbols.intern(name);
                let func_node = self.add_node(
                    NodeType::Function,
                    name,
                    0,
                    args.len() as u32,
                );
                self.add_edge(parent_node, func_node, EdgeType::HasArgument);
                
                // Add function arguments
                for (idx, arg) in args.iter().enumerate() {
                    self.add_term(arg, func_node, idx);
                }
                
                func_node
            }
        }
    }
    
    /// Add a node to the problem
    fn add_node(&mut self, node_type: NodeType, symbol: &str, polarity: i8, arity: u32) -> usize {
        let node_idx = self.problem.num_nodes;
        
        // Add node data
        self.problem.node_types.push(node_type);
        
        let symbol_id = if symbol.is_empty() {
            0
        } else {
            self.problem.symbols.intern(symbol)
        };
        self.problem.node_symbols.push(symbol_id);
        
        self.problem.node_polarities.push(polarity);
        self.problem.node_arities.push(arity);
        
        // Prepare edge offset for this node
        let last_offset = *self.problem.edge_row_offsets.last().unwrap();
        self.problem.edge_row_offsets.push(last_offset);
        
        self.problem.num_nodes += 1;
        node_idx
    }
    
    /// Add an edge between two nodes
    fn add_edge(&mut self, from: usize, to: usize, edge_type: EdgeType) {
        // Add edge data
        self.problem.edge_col_indices.push(to as u32);
        self.problem.edge_types.push(edge_type);
        
        // Update the offset for the next node
        let last_offset_idx = self.problem.edge_row_offsets.len() - 1;
        self.problem.edge_row_offsets[last_offset_idx] += 1;
    }
}

#[cfg(test)]
#[path = "builder_tests.rs"]
mod tests;