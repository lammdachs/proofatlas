//! Graph representation of logical clauses for GNN training
//!
//! Feature layout (12 dimensions):
//! - 0-5: Node type one-hot (clause, literal, predicate, function, variable, constant)
//! - 6: Arity (for predicates and functions)
//! - 7: Depth in the clause tree
//! - 8: Clause age (normalized, 0-1 range based on max_age parameter)
//! - 9: Clause role (0=axiom, 1=hypothesis, 2=definition, 3=negated_conjecture, 4=derived)
//! - 10: Literal polarity (1=positive, 0=negative)
//! - 11: Is equality predicate

use crate::core::{Clause, Literal, Term};

/// Node type constants
pub const NODE_TYPE_CLAUSE: u8 = 0;
pub const NODE_TYPE_LITERAL: u8 = 1;
pub const NODE_TYPE_PREDICATE: u8 = 2;
pub const NODE_TYPE_FUNCTION: u8 = 3;
pub const NODE_TYPE_VARIABLE: u8 = 4;
pub const NODE_TYPE_CONSTANT: u8 = 5;

pub const NODE_TYPES: [&str; 6] = [
    "clause",
    "literal",
    "predicate",
    "function",
    "variable",
    "constant",
];

/// Feature dimension
pub const FEATURE_DIM: usize = 12;

/// Feature indices
pub const FEAT_NODE_TYPE_START: usize = 0;
pub const FEAT_ARITY: usize = 6;
pub const FEAT_DEPTH: usize = 7;
pub const FEAT_AGE: usize = 8;
pub const FEAT_ROLE: usize = 9;
pub const FEAT_POLARITY: usize = 10;
pub const FEAT_IS_EQUALITY: usize = 11;

/// Sparse graph representation of a clause
#[derive(Debug, Clone)]
pub struct ClauseGraph {
    /// Number of nodes in graph
    pub num_nodes: usize,

    /// Edge list: (source_idx, target_idx) pairs (COO format)
    pub edge_indices: Vec<(usize, usize)>,

    /// Node feature matrix: (num_nodes, FEATURE_DIM)
    pub node_features: Vec<[f32; FEATURE_DIM]>,

    /// Node types: (num_nodes,)
    pub node_types: Vec<u8>,

    /// Node names for debugging
    pub node_names: Vec<String>,
}

/// Builder for constructing clause graphs
pub struct GraphBuilder {
    /// Next node ID to assign
    node_id: usize,

    /// Edge list
    edges: Vec<(usize, usize)>,

    /// Node features
    features: Vec<[f32; FEATURE_DIM]>,

    /// Node types
    node_types: Vec<u8>,

    /// Node names (for debugging)
    node_names: Vec<String>,

    /// Depth of each node (for feature computation)
    node_depths: Vec<usize>,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        GraphBuilder {
            node_id: 0,
            edges: Vec::new(),
            features: Vec::new(),
            node_types: Vec::new(),
            node_names: Vec::new(),
            node_depths: Vec::new(),
        }
    }

    /// Build graph from a clause (with default max_age of 1000)
    pub fn build_from_clause(clause: &Clause) -> ClauseGraph {
        Self::build_from_clause_with_context(clause, 1000)
    }

    /// Build graph from a clause with context information
    ///
    /// # Arguments
    /// * `clause` - The clause to build a graph for
    /// * `max_age` - Maximum age for normalization (age will be divided by this)
    pub fn build_from_clause_with_context(clause: &Clause, max_age: usize) -> ClauseGraph {
        let mut builder = GraphBuilder::new();

        // Add clause root node
        let clause_node = builder.add_node(
            NODE_TYPE_CLAUSE,
            "clause_root",
            0, // depth
        );

        // Update clause root features including age and role
        builder.update_clause_features(clause_node, clause, max_age);

        // Process each literal
        for literal in &clause.literals {
            builder.add_literal(literal, clause_node, 1);
        }

        ClauseGraph {
            num_nodes: builder.node_id,
            edge_indices: builder.edges,
            node_features: builder.features,
            node_types: builder.node_types,
            node_names: builder.node_names,
        }
    }

    /// Add a node and return its ID
    fn add_node(&mut self, node_type: u8, name: &str, depth: usize) -> usize {
        let id = self.node_id;
        self.node_id += 1;

        self.node_types.push(node_type);
        self.node_names.push(name.to_string());
        self.node_depths.push(depth);

        // Initialize features with type one-hot encoding
        let mut features = [0.0f32; FEATURE_DIM];

        // Type one-hot (indices 0-5)
        if (node_type as usize) < 6 {
            features[FEAT_NODE_TYPE_START + node_type as usize] = 1.0;
        }

        // Depth
        features[FEAT_DEPTH] = depth as f32;

        self.features.push(features);

        id
    }

    /// Add an edge
    fn add_edge(&mut self, source: usize, target: usize) {
        self.edges.push((source, target));
    }

    /// Update clause root node features
    fn update_clause_features(&mut self, node_id: usize, clause: &Clause, max_age: usize) {
        let features = &mut self.features[node_id];

        // Age: normalized to 0-1 range
        let normalized_age = if max_age > 0 {
            (clause.age as f32) / (max_age as f32)
        } else {
            0.0
        };
        features[FEAT_AGE] = normalized_age.min(1.0); // Clamp to 1.0

        // Role: numeric encoding of clause role (goal can be derived from role == 3)
        features[FEAT_ROLE] = clause.role.to_feature_value();
    }

    /// Add a literal node
    fn add_literal(&mut self, literal: &Literal, parent: usize, depth: usize) -> usize {
        let lit_node = self.add_node(NODE_TYPE_LITERAL, "literal", depth);
        self.add_edge(parent, lit_node);

        // Update literal features
        {
            let features = &mut self.features[lit_node];
            features[FEAT_POLARITY] = if literal.polarity { 1.0 } else { 0.0 };
        }

        // Add predicate node
        let pred_node = self.add_predicate(&literal.atom.predicate.name, lit_node, depth + 1);

        // Update predicate features
        {
            let features = &mut self.features[pred_node];
            features[FEAT_ARITY] = literal.atom.args.len() as f32;
            features[FEAT_IS_EQUALITY] = if literal.atom.is_equality() { 1.0 } else { 0.0 };
        }

        // Process arguments
        for term in &literal.atom.args {
            self.add_term(term, pred_node, depth + 2);
        }

        lit_node
    }

    /// Add a predicate node
    fn add_predicate(&mut self, name: &str, parent: usize, depth: usize) -> usize {
        let pred_node = self.add_node(NODE_TYPE_PREDICATE, name, depth);
        self.add_edge(parent, pred_node);
        pred_node
    }

    /// Add a term node (variable, constant, or function)
    fn add_term(&mut self, term: &Term, parent: usize, depth: usize) -> usize {
        match term {
            Term::Variable(var) => {
                let node = self.add_node(NODE_TYPE_VARIABLE, &var.name, depth);
                self.add_edge(parent, node);
                node
            }

            Term::Constant(c) => {
                let node = self.add_node(NODE_TYPE_CONSTANT, &c.name, depth);
                self.add_edge(parent, node);
                node
            }

            Term::Function(func, args) => {
                let func_node = self.add_node(NODE_TYPE_FUNCTION, &func.name, depth);
                self.add_edge(parent, func_node);

                self.features[func_node][FEAT_ARITY] = args.len() as f32;

                // Process arguments recursively
                for arg in args {
                    self.add_term(arg, func_node, depth + 1);
                }

                func_node
            }
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atom, Constant, FunctionSymbol, Literal, PredicateSymbol, Variable};

    #[test]
    fn test_simple_variable() {
        let mut builder = GraphBuilder::new();
        let root = builder.add_node(NODE_TYPE_CLAUSE, "root", 0);

        let var = Term::Variable(Variable {
            name: "x".to_string(),
        });
        let var_node = builder.add_term(&var, root, 1);

        assert_eq!(builder.node_types[var_node], NODE_TYPE_VARIABLE);
        assert_eq!(builder.node_names[var_node], "x");
        assert_eq!(builder.node_depths[var_node], 1);

        // Check edge
        assert_eq!(builder.edges.len(), 1);
        assert_eq!(builder.edges[0], (root, var_node));
    }

    #[test]
    fn test_simple_function() {
        let mut builder = GraphBuilder::new();
        let root = builder.add_node(NODE_TYPE_CLAUSE, "root", 0);

        // f(x, a)
        let x = Term::Variable(Variable {
            name: "x".to_string(),
        });
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });
        let f = Term::Function(
            FunctionSymbol {
                name: "f".to_string(),
                arity: 2,
            },
            vec![x, a],
        );

        let func_node = builder.add_term(&f, root, 1);

        assert_eq!(builder.node_types[func_node], NODE_TYPE_FUNCTION);
        assert_eq!(builder.node_names[func_node], "f");

        // Check arity feature
        assert_eq!(builder.features[func_node][6], 2.0);

        // Check children (x and a)
        assert_eq!(builder.node_id, 4); // root + f + x + a
    }

    #[test]
    fn test_nested_function() {
        let mut builder = GraphBuilder::new();
        let root = builder.add_node(NODE_TYPE_CLAUSE, "root", 0);

        // f(g(x))
        let x = Term::Variable(Variable {
            name: "x".to_string(),
        });
        let g = Term::Function(
            FunctionSymbol {
                name: "g".to_string(),
                arity: 1,
            },
            vec![x],
        );
        let f = Term::Function(
            FunctionSymbol {
                name: "f".to_string(),
                arity: 1,
            },
            vec![g],
        );

        builder.add_term(&f, root, 1);

        // Nodes: root, f, g, x
        assert_eq!(builder.node_id, 4);

        // Check depths
        assert_eq!(builder.node_depths[0], 0); // root
        assert_eq!(builder.node_depths[1], 1); // f
        assert_eq!(builder.node_depths[2], 2); // g
        assert_eq!(builder.node_depths[3], 3); // x
    }

    #[test]
    fn test_simple_clause() {
        // P(x)
        let x = Term::Variable(Variable {
            name: "x".to_string(),
        });
        let atom = Atom {
            predicate: PredicateSymbol {
                name: "P".to_string(),
                arity: 1,
            },
            args: vec![x],
        };
        let literal = Literal {
            atom,
            polarity: true,
        };
        let clause = Clause::new(vec![literal]);

        let graph = GraphBuilder::build_from_clause(&clause);

        // Nodes: clause, literal, predicate, variable
        assert_eq!(graph.num_nodes, 4);

        // Edges: clause→literal, literal→predicate, predicate→variable
        assert_eq!(graph.edge_indices.len(), 3);

        // Check node types
        assert_eq!(graph.node_types[0], NODE_TYPE_CLAUSE);
        assert_eq!(graph.node_types[1], NODE_TYPE_LITERAL);
        assert_eq!(graph.node_types[2], NODE_TYPE_PREDICATE);
        assert_eq!(graph.node_types[3], NODE_TYPE_VARIABLE);
    }

    #[test]
    fn test_clause_with_two_literals() {
        // P(x) | Q(a)
        let x = Term::Variable(Variable {
            name: "x".to_string(),
        });
        let a = Term::Constant(Constant {
            name: "a".to_string(),
        });

        let lit1 = Literal {
            atom: Atom {
                predicate: PredicateSymbol {
                    name: "P".to_string(),
                    arity: 1,
                },
                args: vec![x],
            },
            polarity: true,
        };

        let lit2 = Literal {
            atom: Atom {
                predicate: PredicateSymbol {
                    name: "Q".to_string(),
                    arity: 1,
                },
                args: vec![a],
            },
            polarity: false,
        };

        let clause = Clause::new(vec![lit1, lit2]);

        let graph = GraphBuilder::build_from_clause(&clause);

        // Nodes: clause + 2*(literal + predicate + term) = 1 + 6 = 7
        assert_eq!(graph.num_nodes, 7);

        // Check polarities
        let lit1_node = 1;
        let lit2_node = 4;
        assert_eq!(graph.node_features[lit1_node][FEAT_POLARITY], 1.0); // positive
        assert_eq!(graph.node_features[lit2_node][FEAT_POLARITY], 0.0); // negative
    }

    #[test]
    fn test_clause_age_and_role_features() {
        use crate::core::ClauseRole;

        // Create a simple clause P(x)
        let x = Term::Variable(Variable {
            name: "x".to_string(),
        });
        let atom = Atom {
            predicate: PredicateSymbol {
                name: "P".to_string(),
                arity: 1,
            },
            args: vec![x],
        };
        let literal = Literal {
            atom,
            polarity: true,
        };

        // Create clause with specific age and role
        let mut clause = Clause::new(vec![literal]);
        clause.age = 500;
        clause.role = ClauseRole::NegatedConjecture;

        // Build graph with max_age = 1000
        let graph = GraphBuilder::build_from_clause_with_context(&clause, 1000);

        // Check age feature (should be 0.5 = 500/1000)
        assert!((graph.node_features[0][FEAT_AGE] - 0.5).abs() < 0.001);

        // Check role feature (NegatedConjecture = 3.0)
        assert_eq!(graph.node_features[0][FEAT_ROLE], 3.0);

        // Test with Axiom role
        let mut axiom_clause = Clause::new(vec![Literal {
            atom: Atom {
                predicate: PredicateSymbol {
                    name: "Q".to_string(),
                    arity: 0,
                },
                args: vec![],
            },
            polarity: true,
        }]);
        axiom_clause.age = 0;
        axiom_clause.role = ClauseRole::Axiom;

        let axiom_graph = GraphBuilder::build_from_clause_with_context(&axiom_clause, 1000);

        // Check age feature (should be 0.0)
        assert_eq!(axiom_graph.node_features[0][FEAT_AGE], 0.0);

        // Check role feature (Axiom = 0.0)
        assert_eq!(axiom_graph.node_features[0][FEAT_ROLE], 0.0);
    }
}
