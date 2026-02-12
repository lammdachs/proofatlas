//! Graph representation of logical clauses for GNN training
//!
//! ## Feature Layout (8 dimensions)
//!
//! Raw feature values - encoding (one-hot, sinusoidal) is done in the model.
//!
//! | Index | Feature | Type | Description |
//! |-------|---------|------|-------------|
//! | 0 | Node type | int 0-5 | clause, literal, predicate, function, variable, constant |
//! | 1 | Arity | int | Number of arguments (for predicates/functions) |
//! | 2 | Arg position | int | 0-indexed position in parent's argument list |
//! | 3 | Depth | int | Depth in the clause tree |
//! | 4 | Age | float 0-1 | Normalized clause age (age / max_age) |
//! | 5 | Role | int 0-4 | axiom, hypothesis, definition, negated_conjecture, derived |
//! | 6 | Polarity | binary | 1=positive literal, 0=negative |
//! | 7 | Is equality | binary | 1 if equality predicate |
//!
//! The model's FeatureEmbedding layer converts these to a richer representation:
//! - Node type/role → one-hot or learned embeddings
//! - Arity/position/depth/age → sinusoidal encoding

use crate::logic::{Clause, Interner, Literal, Term};

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

/// Feature dimension (raw features, encoding done in model)
pub const FEATURE_DIM: usize = 8;

/// Feature indices
pub const FEAT_NODE_TYPE: usize = 0;
pub const FEAT_ARITY: usize = 1;
pub const FEAT_ARG_POSITION: usize = 2;
pub const FEAT_DEPTH: usize = 3;
pub const FEAT_AGE: usize = 4;
pub const FEAT_ROLE: usize = 5;
pub const FEAT_POLARITY: usize = 6;
pub const FEAT_IS_EQUALITY: usize = 7;


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
    pub fn build_from_clause(clause: &Clause, interner: &Interner) -> ClauseGraph {
        Self::build_from_clause_with_context(clause, 1000, interner)
    }

    /// Build graph from a clause with context information
    ///
    /// # Arguments
    /// * `clause` - The clause to build a graph for
    /// * `max_age` - Maximum age for normalization (age will be divided by this)
    /// * `interner` - Interner for name resolution
    pub fn build_from_clause_with_context(clause: &Clause, max_age: usize, interner: &Interner) -> ClauseGraph {
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
            builder.add_literal(literal, clause_node, 1, interner);
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

        // Initialize features with raw values (encoding done in model)
        let mut features = [0.0f32; FEATURE_DIM];
        features[FEAT_NODE_TYPE] = node_type as f32;
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
    fn add_literal(&mut self, literal: &Literal, parent: usize, depth: usize, interner: &Interner) -> usize {
        let lit_node = self.add_node(NODE_TYPE_LITERAL, "literal", depth);
        self.add_edge(parent, lit_node);

        // Update literal features
        {
            let features = &mut self.features[lit_node];
            features[FEAT_POLARITY] = if literal.polarity { 1.0 } else { 0.0 };
        }

        // Add predicate node
        let pred_name = literal.predicate.name(interner);
        let pred_node = self.add_predicate(pred_name, lit_node, depth + 1);

        // Update predicate features
        {
            let features = &mut self.features[pred_node];
            features[FEAT_ARITY] = literal.args.len() as f32;
            features[FEAT_IS_EQUALITY] = if literal.is_equality(interner) { 1.0 } else { 0.0 };
        }

        // Process arguments with their positions
        for (pos, term) in literal.args.iter().enumerate() {
            self.add_term(term, pred_node, depth + 2, pos, interner);
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
    ///
    /// # Arguments
    /// * `term` - The term to add
    /// * `parent` - Parent node ID
    /// * `depth` - Depth in the clause tree
    /// * `arg_position` - 0-indexed position as argument to parent
    /// * `interner` - Interner for name resolution
    fn add_term(&mut self, term: &Term, parent: usize, depth: usize, arg_position: usize, interner: &Interner) -> usize {
        match term {
            Term::Variable(var) => {
                let name = var.name(interner);
                let node = self.add_node(NODE_TYPE_VARIABLE, name, depth);
                self.add_edge(parent, node);
                self.features[node][FEAT_ARG_POSITION] = arg_position as f32;
                node
            }

            Term::Constant(c) => {
                let name = c.name(interner);
                let node = self.add_node(NODE_TYPE_CONSTANT, name, depth);
                self.add_edge(parent, node);
                self.features[node][FEAT_ARG_POSITION] = arg_position as f32;
                node
            }

            Term::Function(func, args) => {
                let name = func.name(interner);
                let func_node = self.add_node(NODE_TYPE_FUNCTION, name, depth);
                self.add_edge(parent, func_node);

                self.features[func_node][FEAT_ARITY] = args.len() as f32;
                self.features[func_node][FEAT_ARG_POSITION] = arg_position as f32;

                // Process arguments recursively with their positions
                for (pos, arg) in args.iter().enumerate() {
                    self.add_term(arg, func_node, depth + 1, pos, interner);
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

/// Multi-clause graph representation for batch inference
#[derive(Debug, Clone)]
pub struct BatchClauseGraph {
    /// Total number of nodes across all clauses
    pub num_nodes: usize,

    /// Edge list: (source_idx, target_idx) pairs
    pub edge_indices: Vec<(usize, usize)>,

    /// Node feature matrix: (num_nodes, 3) - only type, arity, arg_pos
    /// Clause features (age, role, size) are provided separately
    pub node_features: Vec<[f32; 3]>,

    /// Clause boundaries: (start_node, end_node) for each clause
    /// Used to build the pool matrix
    pub clause_boundaries: Vec<(usize, usize)>,

    /// Edge boundaries: (start_edge, end_edge) for each clause
    /// Used to split per-clause edges in trace extraction
    pub edge_boundaries: Vec<(usize, usize)>,
}

impl GraphBuilder {
    /// Collect node names for all clauses in the same traversal order as `build_from_clauses`.
    ///
    /// Returns a flat list of symbol names: structural sentinels ("CLAUSE", "LIT", "VAR")
    /// for clause/literal/variable nodes, and the actual symbol name for
    /// predicate/function/constant nodes. Used by the `node_info="names"` GCN mode.
    pub fn collect_node_names(clauses: &[&Clause], interner: &Interner) -> Vec<String> {
        let mut names = Vec::new();
        for clause in clauses {
            names.push("CLAUSE".to_string());
            for literal in &clause.literals {
                names.push("LIT".to_string());
                names.push(literal.predicate.name(interner).to_string());
                for term in &literal.args {
                    Self::collect_term_names(term, interner, &mut names);
                }
            }
        }
        names
    }

    fn collect_term_names(term: &Term, interner: &Interner, names: &mut Vec<String>) {
        match term {
            Term::Variable(_) => names.push("VAR".to_string()),
            Term::Constant(c) => names.push(c.name(interner).to_string()),
            Term::Function(func, args) => {
                names.push(func.name(interner).to_string());
                for arg in args {
                    Self::collect_term_names(arg, interner, names);
                }
            }
        }
    }

    /// Build a combined graph from multiple clauses for batch inference
    ///
    /// This builds directly into batch arrays without intermediate ClauseGraph
    /// objects. Only computes the 3 features needed for inference: [type, arity, arg_pos].
    /// Skips node_names, node_types, node_depths, and features 3-7.
    pub fn build_from_clauses(clauses: &[&Clause]) -> BatchClauseGraph {
        let mut features: Vec<[f32; 3]> = Vec::new();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut clause_boundaries: Vec<(usize, usize)> = Vec::new();
        let mut edge_boundaries: Vec<(usize, usize)> = Vec::new();
        let mut next_id: usize = 0;

        for clause in clauses {
            let start = next_id;
            let edge_start = edges.len();

            // Clause root node: [type=0, arity=0, arg_pos=0]
            let clause_node = next_id;
            features.push([NODE_TYPE_CLAUSE as f32, 0.0, 0.0]);
            next_id += 1;

            for literal in &clause.literals {
                // Literal node
                let lit_node = next_id;
                features.push([NODE_TYPE_LITERAL as f32, 0.0, 0.0]);
                edges.push((clause_node, lit_node));
                next_id += 1;

                // Predicate node
                let pred_node = next_id;
                features.push([
                    NODE_TYPE_PREDICATE as f32,
                    literal.args.len() as f32,
                    0.0,
                ]);
                edges.push((lit_node, pred_node));
                next_id += 1;

                // Term nodes
                for (pos, term) in literal.args.iter().enumerate() {
                    next_id = Self::add_term_batch(
                        term, pred_node, pos, next_id, &mut features, &mut edges,
                    );
                }
            }

            clause_boundaries.push((start, next_id));
            edge_boundaries.push((edge_start, edges.len()));
        }

        BatchClauseGraph {
            num_nodes: next_id,
            edge_indices: edges,
            node_features: features,
            clause_boundaries,
            edge_boundaries,
        }
    }

    /// Add a term node directly to batch arrays (no intermediate objects)
    fn add_term_batch(
        term: &Term,
        parent: usize,
        arg_position: usize,
        next_id: usize,
        features: &mut Vec<[f32; 3]>,
        edges: &mut Vec<(usize, usize)>,
    ) -> usize {
        match term {
            Term::Variable(_) => {
                features.push([NODE_TYPE_VARIABLE as f32, 0.0, arg_position as f32]);
                edges.push((parent, next_id));
                next_id + 1
            }
            Term::Constant(_) => {
                features.push([NODE_TYPE_CONSTANT as f32, 0.0, arg_position as f32]);
                edges.push((parent, next_id));
                next_id + 1
            }
            Term::Function(_, args) => {
                let func_node = next_id;
                features.push([
                    NODE_TYPE_FUNCTION as f32,
                    args.len() as f32,
                    arg_position as f32,
                ]);
                edges.push((parent, func_node));
                let mut id = next_id + 1;
                for (pos, arg) in args.iter().enumerate() {
                    id = Self::add_term_batch(arg, func_node, pos, id, features, edges);
                }
                id
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, Literal, PredicateSymbol, Variable};

    #[test]
    fn test_simple_variable() {
        let mut interner = Interner::new();
        let mut builder = GraphBuilder::new();
        let root = builder.add_node(NODE_TYPE_CLAUSE, "root", 0);

        let x_id = interner.intern_variable("x");
        let var = Term::Variable(Variable { id: x_id });
        let var_node = builder.add_term(&var, root, 1, 0, &interner);

        assert_eq!(builder.node_types[var_node], NODE_TYPE_VARIABLE);
        assert_eq!(builder.node_names[var_node], "x");
        assert_eq!(builder.node_depths[var_node], 1);

        // Check edge
        assert_eq!(builder.edges.len(), 1);
        assert_eq!(builder.edges[0], (root, var_node));

        // Check arg position
        assert_eq!(builder.features[var_node][FEAT_ARG_POSITION], 0.0);
    }

    #[test]
    fn test_simple_function() {
        let mut interner = Interner::new();
        let mut builder = GraphBuilder::new();
        let root = builder.add_node(NODE_TYPE_CLAUSE, "root", 0);

        // f(x, a)
        let x_id = interner.intern_variable("x");
        let a_id = interner.intern_constant("a");
        let f_id = interner.intern_function("f");

        let x = Term::Variable(Variable { id: x_id });
        let a = Term::Constant(Constant { id: a_id });
        let f = Term::Function(
            FunctionSymbol { id: f_id, arity: 2 },
            vec![x, a],
        );

        let func_node = builder.add_term(&f, root, 1, 0, &interner);

        assert_eq!(builder.node_types[func_node], NODE_TYPE_FUNCTION);
        assert_eq!(builder.node_names[func_node], "f");

        // Check arity feature
        assert_eq!(builder.features[func_node][FEAT_ARITY], 2.0);

        // Check children (x and a)
        assert_eq!(builder.node_id, 4); // root + f + x + a

        // Check arg positions: x is at position 0, a is at position 1
        let x_node = 2;
        let a_node = 3;
        assert_eq!(builder.features[x_node][FEAT_ARG_POSITION], 0.0);
        assert_eq!(builder.features[a_node][FEAT_ARG_POSITION], 1.0);
    }

    #[test]
    fn test_nested_function() {
        let mut interner = Interner::new();
        let mut builder = GraphBuilder::new();
        let root = builder.add_node(NODE_TYPE_CLAUSE, "root", 0);

        // f(g(x))
        let x_id = interner.intern_variable("x");
        let g_id = interner.intern_function("g");
        let f_id = interner.intern_function("f");

        let x = Term::Variable(Variable { id: x_id });
        let g = Term::Function(
            FunctionSymbol { id: g_id, arity: 1 },
            vec![x],
        );
        let f = Term::Function(
            FunctionSymbol { id: f_id, arity: 1 },
            vec![g],
        );

        builder.add_term(&f, root, 1, 0, &interner);

        // Nodes: root, f, g, x
        assert_eq!(builder.node_id, 4);

        // Check depths
        assert_eq!(builder.node_depths[0], 0); // root
        assert_eq!(builder.node_depths[1], 1); // f
        assert_eq!(builder.node_depths[2], 2); // g
        assert_eq!(builder.node_depths[3], 3); // x

        // Check arg positions (all are first arguments)
        assert_eq!(builder.features[1][FEAT_ARG_POSITION], 0.0); // f
        assert_eq!(builder.features[2][FEAT_ARG_POSITION], 0.0); // g
        assert_eq!(builder.features[3][FEAT_ARG_POSITION], 0.0); // x
    }

    #[test]
    fn test_simple_clause() {
        let mut interner = Interner::new();

        // P(x)
        let x_id = interner.intern_variable("x");
        let p_id = interner.intern_predicate("P");

        let x = Term::Variable(Variable { id: x_id });
        let literal = Literal::positive(
            PredicateSymbol { id: p_id, arity: 1 },
            vec![x],
        );
        let clause = Clause::new(vec![literal]);

        let graph = GraphBuilder::build_from_clause(&clause, &interner);

        // Nodes: clause, literal, predicate, variable
        assert_eq!(graph.num_nodes, 4);

        // Edges: clause->literal, literal->predicate, predicate->variable
        assert_eq!(graph.edge_indices.len(), 3);

        // Check node types
        assert_eq!(graph.node_types[0], NODE_TYPE_CLAUSE);
        assert_eq!(graph.node_types[1], NODE_TYPE_LITERAL);
        assert_eq!(graph.node_types[2], NODE_TYPE_PREDICATE);
        assert_eq!(graph.node_types[3], NODE_TYPE_VARIABLE);
    }

    #[test]
    fn test_clause_with_two_literals() {
        let mut interner = Interner::new();

        // P(x) | Q(a)
        let x_id = interner.intern_variable("x");
        let a_id = interner.intern_constant("a");
        let p_id = interner.intern_predicate("P");
        let q_id = interner.intern_predicate("Q");

        let x = Term::Variable(Variable { id: x_id });
        let a = Term::Constant(Constant { id: a_id });

        let lit1 = Literal::positive(
            PredicateSymbol { id: p_id, arity: 1 },
            vec![x],
        );

        let lit2 = Literal::negative(
            PredicateSymbol { id: q_id, arity: 1 },
            vec![a],
        );

        let clause = Clause::new(vec![lit1, lit2]);

        let graph = GraphBuilder::build_from_clause(&clause, &interner);

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
        use crate::logic::ClauseRole;

        let mut interner = Interner::new();

        // Create a simple clause P(x)
        let x_id = interner.intern_variable("x");
        let p_id = interner.intern_predicate("P");

        let x = Term::Variable(Variable { id: x_id });
        let literal = Literal::positive(
            PredicateSymbol { id: p_id, arity: 1 },
            vec![x],
        );

        // Create clause with specific age and role
        let mut clause = Clause::new(vec![literal]);
        clause.age = 500;
        clause.role = ClauseRole::NegatedConjecture;

        // Build graph with max_age = 1000
        let graph = GraphBuilder::build_from_clause_with_context(&clause, 1000, &interner);

        // Check age feature (should be 0.5 = 500/1000)
        assert!((graph.node_features[0][FEAT_AGE] - 0.5).abs() < 0.001);

        // Check role feature (NegatedConjecture = 3.0)
        assert_eq!(graph.node_features[0][FEAT_ROLE], 3.0);

        // Test with Axiom role
        let q_id = interner.intern_predicate("Q");
        let mut axiom_clause = Clause::new(vec![Literal::positive(
            PredicateSymbol { id: q_id, arity: 0 },
            vec![],
        )]);
        axiom_clause.age = 0;
        axiom_clause.role = ClauseRole::Axiom;

        let axiom_graph = GraphBuilder::build_from_clause_with_context(&axiom_clause, 1000, &interner);

        // Check age feature (should be 0.0)
        assert_eq!(axiom_graph.node_features[0][FEAT_AGE], 0.0);

        // Check role feature (Axiom = 0.0)
        assert_eq!(axiom_graph.node_features[0][FEAT_ROLE], 0.0);
    }

    #[test]
    fn test_collect_node_names_matches_batch() {
        let mut interner = Interner::new();

        // Build two clauses: P(x) and Q(f(a, b))
        let x_id = interner.intern_variable("x");
        let a_id = interner.intern_constant("a");
        let b_id = interner.intern_constant("b");
        let f_id = interner.intern_function("f");
        let p_id = interner.intern_predicate("P");
        let q_id = interner.intern_predicate("Q");

        let x = Term::Variable(Variable { id: x_id });
        let a = Term::Constant(Constant { id: a_id });
        let b = Term::Constant(Constant { id: b_id });
        let f_ab = Term::Function(
            FunctionSymbol { id: f_id, arity: 2 },
            vec![a, b],
        );

        let clause1 = Clause::new(vec![Literal::positive(
            PredicateSymbol { id: p_id, arity: 1 },
            vec![x],
        )]);
        let clause2 = Clause::new(vec![Literal::positive(
            PredicateSymbol { id: q_id, arity: 1 },
            vec![f_ab],
        )]);

        let clauses: Vec<&Clause> = vec![&clause1, &clause2];
        let batch = GraphBuilder::build_from_clauses(&clauses);
        let names = GraphBuilder::collect_node_names(&clauses, &interner);

        assert_eq!(names.len(), batch.num_nodes);

        // Clause 1: CLAUSE, LIT, P, VAR
        assert_eq!(names[0], "CLAUSE");
        assert_eq!(names[1], "LIT");
        assert_eq!(names[2], "P");
        assert_eq!(names[3], "VAR");

        // Clause 2: CLAUSE, LIT, Q, f, a, b
        assert_eq!(names[4], "CLAUSE");
        assert_eq!(names[5], "LIT");
        assert_eq!(names[6], "Q");
        assert_eq!(names[7], "f");
        assert_eq!(names[8], "a");
        assert_eq!(names[9], "b");
    }

    #[test]
    fn test_batch_direct_matches_individual() {
        let mut interner = Interner::new();

        // Build two clauses: P(x) and Q(f(a, b))
        let x_id = interner.intern_variable("x");
        let a_id = interner.intern_constant("a");
        let b_id = interner.intern_constant("b");
        let f_id = interner.intern_function("f");
        let p_id = interner.intern_predicate("P");
        let q_id = interner.intern_predicate("Q");

        let x = Term::Variable(Variable { id: x_id });
        let a = Term::Constant(Constant { id: a_id });
        let b = Term::Constant(Constant { id: b_id });
        let f_ab = Term::Function(
            FunctionSymbol { id: f_id, arity: 2 },
            vec![a, b],
        );

        let clause1 = Clause::new(vec![Literal::positive(
            PredicateSymbol { id: p_id, arity: 1 },
            vec![x],
        )]);
        let clause2 = Clause::new(vec![Literal::positive(
            PredicateSymbol { id: q_id, arity: 1 },
            vec![f_ab],
        )]);

        let clauses: Vec<&Clause> = vec![&clause1, &clause2];
        let batch = GraphBuilder::build_from_clauses(&clauses);

        // Verify against individual graphs
        let g1 = GraphBuilder::build_from_clause(&clause1, &interner);
        let g2 = GraphBuilder::build_from_clause(&clause2, &interner);

        assert_eq!(batch.num_nodes, g1.num_nodes + g2.num_nodes);
        assert_eq!(batch.edge_indices.len(), g1.edge_indices.len() + g2.edge_indices.len());
        assert_eq!(batch.clause_boundaries.len(), 2);
        assert_eq!(batch.clause_boundaries[0], (0, g1.num_nodes));
        assert_eq!(batch.clause_boundaries[1], (g1.num_nodes, g1.num_nodes + g2.num_nodes));

        // Check features match (first 3 dims only)
        for i in 0..g1.num_nodes {
            let expected = [g1.node_features[i][0], g1.node_features[i][1], g1.node_features[i][2]];
            assert_eq!(batch.node_features[i], expected, "mismatch at node {i}");
        }
        for i in 0..g2.num_nodes {
            let expected = [g2.node_features[i][0], g2.node_features[i][1], g2.node_features[i][2]];
            assert_eq!(batch.node_features[g1.num_nodes + i], expected, "mismatch at node {}", g1.num_nodes + i);
        }

        // Check edges match with offset
        for (j, &(src, dst)) in g1.edge_indices.iter().enumerate() {
            assert_eq!(batch.edge_indices[j], (src, dst));
        }
        let offset = g1.edge_indices.len();
        for (j, &(src, dst)) in g2.edge_indices.iter().enumerate() {
            assert_eq!(batch.edge_indices[offset + j], (src + g1.num_nodes, dst + g1.num_nodes));
        }
    }
}
