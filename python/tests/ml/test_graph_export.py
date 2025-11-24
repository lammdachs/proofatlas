#!/usr/bin/env python3
"""Test graph export functionality for GNN training"""

import pytest
import numpy as np
from proofatlas import ProofState


class TestBasicGraphExport:
    """Test basic graph export functionality"""

    def test_simple_clause_export(self):
        """Test graph export for a simple clause: P(x)"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])

        # Check graph has correct structure
        # Expected: clause -> literal -> predicate -> variable
        assert graph.num_nodes() == 4
        assert graph.num_edges() == 3
        assert graph.feature_dim() == 20

    def test_two_literal_clause(self):
        """Test graph export for clause with two literals: P(x) | Q(a)"""
        state = ProofState()
        tptp = "cnf(test, axiom, (p(X) | q(a)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])

        # Expected: clause -> 2*(literal -> predicate -> term) = 1 + 2*3 = 7 nodes
        assert graph.num_nodes() == 7
        assert graph.num_edges() == 6

    def test_negated_literal(self):
        """Test graph export with negated literal: ~P(x)"""
        state = ProofState()
        tptp = "cnf(test, axiom, ~p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Find literal node (should be node 1)
        # Feature index 12 is polarity
        literal_polarity = node_features[1, 12]
        assert literal_polarity == 0.0  # negative literal

    def test_invalid_clause_id(self):
        """Test that invalid clause ID raises error"""
        state = ProofState()
        with pytest.raises(Exception):  # Should raise PyValueError
            state.clause_to_graph(999)


class TestNodeTypes:
    """Test correct node type assignment"""

    def test_node_type_constants(self):
        """Test that node types match expected constants"""
        # NODE_TYPE_CLAUSE = 0, LITERAL = 1, PREDICATE = 2,
        # FUNCTION = 3, VARIABLE = 4, CONSTANT = 5
        state = ProofState()
        tptp = "cnf(test, axiom, p(X, a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()

        # Check each node has valid type
        assert all(0 <= t <= 5 for t in node_types)

        # Node 0 should be CLAUSE
        assert node_types[0] == 0

        # Node 1 should be LITERAL
        assert node_types[1] == 1

        # Node 2 should be PREDICATE
        assert node_types[2] == 2

    def test_variable_and_constant_types(self):
        """Test distinction between variables and constants"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X, a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()
        node_names = graph.node_names()

        # Find variable and constant nodes
        var_idx = None
        const_idx = None
        for i, name in enumerate(node_names):
            if name == "X":
                var_idx = i
            elif name == "a":
                const_idx = i

        assert var_idx is not None, "Variable X not found"
        assert const_idx is not None, "Constant a not found"

        assert node_types[var_idx] == 4  # VARIABLE
        assert node_types[const_idx] == 5  # CONSTANT


class TestFunctionHandling:
    """Test graph export for clauses with functions"""

    def test_simple_function(self):
        """Test function node: f(x)"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(f(X)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()

        # Should have: clause, literal, predicate, function, variable
        assert graph.num_nodes() == 5

        # Count node types
        type_counts = {t: np.sum(node_types == t) for t in range(6)}
        assert type_counts[0] == 1  # 1 clause
        assert type_counts[1] == 1  # 1 literal
        assert type_counts[2] == 1  # 1 predicate
        assert type_counts[3] == 1  # 1 function
        assert type_counts[4] == 1  # 1 variable

    def test_nested_functions(self):
        """Test nested functions: f(g(x))"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(f(g(X))))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()

        # Should have: clause, literal, predicate, f, g, variable
        assert graph.num_nodes() == 6

        # Should have 2 functions
        assert np.sum(node_types == 3) == 2

    def test_function_arity(self):
        """Test that function arity is correctly recorded"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(f(X, Y)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()
        node_types = graph.node_types()
        node_names = graph.node_names()

        # Find function node
        func_idx = None
        for i, (t, name) in enumerate(zip(node_types, node_names)):
            if t == 3 and name == "f":  # FUNCTION type
                func_idx = i
                break

        assert func_idx is not None, "Function f not found"

        # Feature index 6 is arity
        arity = node_features[func_idx, 6]
        assert arity == 2.0


class TestEqualityHandling:
    """Test graph export for equality predicates"""

    def test_equality_detection(self):
        """Test that equality is correctly detected"""
        state = ProofState()
        tptp = "cnf(test, axiom, X = Y)."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()
        node_types = graph.node_types()

        # Find predicate node (type 2)
        pred_idx = None
        for i, t in enumerate(node_types):
            if t == 2:
                pred_idx = i
                break

        assert pred_idx is not None

        # Feature index 13 is is_equality
        is_equality = node_features[pred_idx, 13]
        assert is_equality == 1.0

    def test_complex_equality(self):
        """Test equality with function terms: f(X) = g(Y)"""
        state = ProofState()
        tptp = "cnf(test, axiom, f(X) = g(Y))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()

        # Should have: clause, literal, predicate(=), f, X, g, Y
        assert graph.num_nodes() == 7

        # Should have 2 functions
        assert np.sum(node_types == 3) == 2


class TestClauseFeatures:
    """Test clause-level features"""

    def test_unit_clause_feature(self):
        """Test that unit clauses are correctly identified"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Node 0 is clause root
        # Feature index 14 is is_unit
        is_unit = node_features[0, 14]
        assert is_unit == 1.0

    def test_non_unit_clause_feature(self):
        """Test that non-unit clauses are correctly identified"""
        state = ProofState()
        tptp = "cnf(test, axiom, (p(X) | q(Y)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Feature index 14 is is_unit
        is_unit = node_features[0, 14]
        assert is_unit == 0.0

    def test_horn_clause_feature(self):
        """Test that Horn clauses are correctly identified"""
        state = ProofState()
        # Horn clause: at most one positive literal
        tptp = "cnf(test, axiom, (~p(X) | q(X)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Feature index 15 is is_horn
        is_horn = node_features[0, 15]
        assert is_horn == 1.0

    def test_ground_clause_feature(self):
        """Test that ground clauses are correctly identified"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Feature index 16 is is_ground
        is_ground = node_features[0, 16]
        assert is_ground == 1.0

    def test_non_ground_clause_feature(self):
        """Test that non-ground clauses are correctly identified"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Feature index 16 is is_ground
        is_ground = node_features[0, 16]
        assert is_ground == 0.0


class TestEdgeStructure:
    """Test graph edge structure"""

    def test_edge_indices_shape(self):
        """Test that edge indices have correct shape (2, num_edges)"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        edge_indices = graph.edge_indices()

        assert edge_indices.shape == (2, graph.num_edges())
        assert edge_indices.dtype == np.int64

    def test_edge_connectivity(self):
        """Test that edges form a valid tree from clause root"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        edge_indices = graph.edge_indices()

        # All edges should be valid node indices
        assert np.all(edge_indices >= 0)
        assert np.all(edge_indices < graph.num_nodes())

        # Build adjacency for verification
        sources = edge_indices[0, :]
        targets = edge_indices[1, :]

        # Root (node 0) should be a source
        assert 0 in sources

        # Root should not be a target (it's the root)
        assert 0 not in targets

    def test_tree_structure(self):
        """Test that graph forms a tree (no cycles, connected)"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(f(X)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        edge_indices = graph.edge_indices()

        # Tree property: num_edges = num_nodes - 1
        assert graph.num_edges() == graph.num_nodes() - 1

        # Each node (except root) should have exactly one parent
        targets = edge_indices[1, :]
        unique_targets = np.unique(targets)

        # Number of unique targets should be num_nodes - 1 (all except root)
        assert len(unique_targets) == graph.num_nodes() - 1


class TestFeatureArrays:
    """Test feature array properties"""

    def test_node_features_shape(self):
        """Test that node features have correct shape"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        assert node_features.shape == (graph.num_nodes(), 20)
        assert node_features.dtype == np.float32

    def test_node_types_shape(self):
        """Test that node types have correct shape"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()

        assert node_types.shape == (graph.num_nodes(),)
        assert node_types.dtype == np.uint8

    def test_feature_type_encoding(self):
        """Test that node type is one-hot encoded in features"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()
        node_types = graph.node_types()

        # For each node, check that features[0:6] is one-hot encoding of type
        for i in range(graph.num_nodes()):
            type_idx = node_types[i]
            type_features = node_features[i, :6]

            # Should be one-hot: all zeros except at type_idx
            expected = np.zeros(6, dtype=np.float32)
            expected[type_idx] = 1.0

            np.testing.assert_array_equal(type_features, expected)

    def test_depth_feature(self):
        """Test that depth feature is correctly set"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Feature index 7 is depth
        # Clause root should have depth 0
        assert node_features[0, 7] == 0.0

        # Literal should have depth 1
        assert node_features[1, 7] == 1.0

        # Predicate should have depth 2
        assert node_features[2, 7] == 2.0


class TestBatchConversion:
    """Test batch conversion of multiple clauses"""

    def test_single_clause_batch(self):
        """Test batch conversion with single clause"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = state.clauses_to_graphs(clause_ids)

        assert len(graphs) == 1
        assert graphs[0].num_nodes() == 4

    def test_multiple_clause_batch(self):
        """Test batch conversion with multiple clauses"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(X)).
        cnf(c2, axiom, q(Y)).
        cnf(c3, axiom, r(a)).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = state.clauses_to_graphs(clause_ids)

        assert len(graphs) == 3
        assert all(g.num_nodes() > 0 for g in graphs)

    def test_batch_with_different_sizes(self):
        """Test batch conversion with clauses of different sizes"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(X)).
        cnf(c2, axiom, (p(X) | q(Y))).
        cnf(c3, axiom, (p(X) | q(Y) | r(Z))).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = state.clauses_to_graphs(clause_ids)

        assert len(graphs) == 3

        # Different number of literals should yield different graph sizes
        assert graphs[0].num_nodes() < graphs[1].num_nodes()
        assert graphs[1].num_nodes() < graphs[2].num_nodes()

    def test_empty_batch(self):
        """Test batch conversion with empty list"""
        state = ProofState()
        graphs = state.clauses_to_graphs([])
        assert len(graphs) == 0


class TestNodeNames:
    """Test node name tracking for debugging"""

    def test_node_names_present(self):
        """Test that node names are provided"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_names = graph.node_names()

        assert len(node_names) == graph.num_nodes()

    def test_node_names_content(self):
        """Test that node names have meaningful content"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X, a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_names = graph.node_names()

        # Should contain predicate name
        assert "p" in node_names

        # Should contain variable name
        assert "X" in node_names

        # Should contain constant name
        assert "a" in node_names

        # Should contain structural names
        assert "clause_root" in node_names
        assert "literal" in node_names


class TestRealWorldExamples:
    """Test with real-world theorem proving examples"""

    def test_group_theory_axiom(self):
        """Test with group theory left inverse axiom"""
        state = ProofState()
        tptp = "cnf(left_inverse, axiom, mult(inv(X), mult(X, Y)) = Y)."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])

        # Should have multiple function nodes for mult and inv
        node_types = graph.node_types()
        num_functions = np.sum(node_types == 3)
        assert num_functions == 3  # mult, inv, mult

        # Should detect equality
        node_features = graph.node_features()
        pred_nodes = np.where(node_types == 2)[0]
        assert len(pred_nodes) > 0
        is_equality = node_features[pred_nodes[0], 13]
        assert is_equality == 1.0

    def test_resolution_example(self):
        """Test with clauses suitable for resolution"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, (~p(X) | q(X))).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = state.clauses_to_graphs(clause_ids)

        assert len(graphs) == 2

        # First clause should be ground
        assert graphs[0].node_features()[0, 16] == 1.0

        # Second clause should not be ground
        assert graphs[1].node_features()[0, 16] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
