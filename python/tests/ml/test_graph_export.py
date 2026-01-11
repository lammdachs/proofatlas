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
        assert graph.feature_dim() == 8

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
        # Feature index 6 is polarity
        literal_polarity = node_features[1, 6]
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

        # Feature index 1 is arity
        arity = node_features[func_idx, 1]
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

        # Feature index 7 is is_equality
        is_equality = node_features[pred_idx, 7]
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

    def test_unit_clause_structure(self):
        """Test that unit clauses have correct structure"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        # Unit clause has 1 literal
        # Structure: clause -> literal -> predicate -> variable
        assert graph.num_nodes() == 4

    def test_non_unit_clause_structure(self):
        """Test that non-unit clauses have correct structure"""
        state = ProofState()
        tptp = "cnf(test, axiom, (p(X) | q(Y)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        # Two literals: clause -> 2 * (literal -> predicate -> variable)
        assert graph.num_nodes() == 7

    def test_polarity_feature(self):
        """Test that literal polarity is correctly encoded"""
        state = ProofState()
        # Horn clause: at most one positive literal
        tptp = "cnf(test, axiom, (~p(X) | q(X)))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()
        node_types = graph.node_types()

        # Find literal nodes (type 1)
        literal_indices = [i for i, t in enumerate(node_types) if t == 1]
        assert len(literal_indices) == 2

        # Check polarities (index 6)
        polarities = [node_features[i, 6] for i in literal_indices]
        # One negative, one positive
        assert 0.0 in polarities and 1.0 in polarities

    def test_ground_clause_structure(self):
        """Test that ground clauses have correct structure"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()

        # Should have constant (type 5), no variable (type 4)
        assert 5 in node_types  # constant
        assert 4 not in node_types  # no variables

    def test_non_ground_clause_structure(self):
        """Test that non-ground clauses have variables"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_types = graph.node_types()

        # Should have variable (type 4)
        assert 4 in node_types


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

        assert node_features.shape == (graph.num_nodes(), 8)
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
        """Test that node type is encoded as raw value in feature 0"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()
        node_types = graph.node_types()

        # For each node, check that feature 0 matches the node type
        for i in range(graph.num_nodes()):
            type_idx = node_types[i]
            # Feature 0 is the raw node type value
            assert node_features[i, 0] == float(type_idx)

    def test_depth_feature(self):
        """Test that depth feature is correctly set"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()

        # Feature index 3 is depth
        # Clause root should have depth 0
        assert node_features[0, 3] == 0.0

        # Literal should have depth 1
        assert node_features[1, 3] == 1.0

        # Predicate should have depth 2
        assert node_features[2, 3] == 2.0

    def test_arg_position_feature(self):
        """Test that arg_position feature is correctly set for function arguments"""
        state = ProofState()
        # p(a, b, c) - arguments at positions 0, 1, 2
        tptp = "cnf(test, axiom, p(a, b, c))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        node_features = graph.node_features()
        node_names = graph.node_names()

        # Find constant nodes and check their arg positions
        # Feature index 2 is arg_position
        for i, name in enumerate(node_names):
            if name == 'a':
                assert node_features[i, 2] == 0.0  # first arg
            elif name == 'b':
                assert node_features[i, 2] == 1.0  # second arg
            elif name == 'c':
                assert node_features[i, 2] == 2.0  # third arg

    def test_role_feature(self):
        """Test that role feature distinguishes axiom from negated_conjecture"""
        state = ProofState()
        tptp = """
        cnf(ax1, axiom, p(a)).
        cnf(nc1, negated_conjecture, q(b)).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        # Feature index 5 is role
        # Role encoding: 0=axiom, 1=hypothesis, 2=definition, 3=negated_conjecture, 4=derived
        graph1 = state.clause_to_graph(clause_ids[0])
        graph2 = state.clause_to_graph(clause_ids[1])

        # Clause root is node 0, check its role feature
        assert graph1.node_features()[0, 5] == 0.0  # axiom
        assert graph2.node_features()[0, 5] == 3.0  # negated_conjecture

    def test_age_feature(self):
        """Test that age feature is correctly computed"""
        state = ProofState()
        # Add initial clauses - they all have raw age 0
        state.add_clauses_from_tptp("""
        cnf(c1, axiom, p(a)).
        cnf(c2, axiom, q(b)).
        cnf(c3, axiom, r(c)).
        """)

        # Feature index 4 is age
        # Initial clauses have raw age 0, so normalized age is 0/max_age = 0
        graph0 = state.clause_to_graph(0)
        age0 = graph0.node_features()[0, 4]

        # Age feature should be non-negative
        assert age0 >= 0.0
        # For initial clauses, age is 0 (not derived during saturation)
        assert age0 == 0.0


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

        # Should detect equality (feature index 7)
        node_features = graph.node_features()
        pred_nodes = np.where(node_types == 2)[0]
        assert len(pred_nodes) > 0
        is_equality = node_features[pred_nodes[0], 7]
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

        # First clause should be ground (has constant, no variable)
        node_types_0 = graphs[0].node_types()
        assert 5 in node_types_0  # has constant
        assert 4 not in node_types_0  # no variable

        # Second clause should not be ground (has variable)
        node_types_1 = graphs[1].node_types()
        assert 4 in node_types_1  # has variable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
