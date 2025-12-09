#!/usr/bin/env python3
"""Tests for PyTorch graph utilities

Tests are organized by user workflow:
1. Basic Conversion - Converting a single graph to tensors
2. Graph Batching - Preparing data for training
3. Advanced Formats - Specialized tensor formats
4. Post-Processing - Operations on GNN outputs
5. Utilities - Helper and analysis functions
"""

import pytest
from proofatlas import ProofState

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import our utilities
from proofatlas.ml import graph_utils


# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)


# ============================================================================
# Level 1: Getting Started - Basic Conversion
# ============================================================================

class TestBasicConversion:
    """Converting a single graph to tensors (most common use case)

    This covers the fundamental operations users need to get started:
    - Converting graphs to PyTorch tensors
    - Device placement (CPU/CUDA)
    """

    def test_to_torch_tensors_basic(self):
        """Test basic conversion to PyTorch tensors"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph)

        # Check all required keys present
        assert 'edge_index' in tensors
        assert 'x' in tensors
        assert 'node_types' in tensors
        assert 'num_nodes' in tensors
        assert 'num_edges' in tensors

    def test_tensor_shapes_and_dtypes(self):
        """Test that tensor shapes and dtypes are correct"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph)

        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()

        # Shapes
        assert tensors['edge_index'].shape == (2, num_edges)
        assert tensors['x'].shape == (num_nodes, 13)
        assert tensors['node_types'].shape == (num_nodes,)

        # Data types
        assert tensors['edge_index'].dtype == torch.int64
        assert tensors['x'].dtype == torch.float32
        assert tensors['node_types'].dtype == torch.uint8

    def test_device_placement_cpu(self):
        """Test tensor device placement on CPU"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph, device='cpu')

        assert str(tensors['edge_index'].device) == 'cpu'
        assert str(tensors['x'].device) == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self):
        """Test CUDA device placement"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph, device='cuda')

        assert tensors['edge_index'].is_cuda
        assert tensors['x'].is_cuda


# ============================================================================
# Level 2: Training Setup - Graph Batching
# ============================================================================

class TestGraphBatching:
    """Preparing data for training

    Covers operations needed to prepare graphs for batch training:
    - Combining multiple graphs into batches
    - Handling labels for supervised learning
    """

    def test_batch_single_graph(self):
        """Test batching a single graph"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = [state.clause_to_graph(clause_ids[0])]
        batched = graph_utils.batch_graphs(graphs)

        assert 'edge_index' in batched
        assert 'x' in batched
        assert 'batch' in batched
        assert batched['num_graphs'] == 1

    def test_batch_multiple_graphs(self):
        """Test batching multiple graphs together"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(X)).
        cnf(c2, axiom, q(Y)).
        cnf(c3, axiom, r(Z)).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = [state.clause_to_graph(id) for id in clause_ids]
        batched = graph_utils.batch_graphs(graphs)

        assert batched['num_graphs'] == 3

        # Total nodes should be sum of individual graphs
        total_nodes = sum(g.num_nodes() for g in graphs)
        assert batched['x'].shape[0] == total_nodes

    def test_batch_with_labels(self):
        """Test batching with classification labels"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(X)).
        cnf(c2, axiom, q(Y)).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = [state.clause_to_graph(id) for id in clause_ids]
        labels = [0, 1]
        batched = graph_utils.batch_graphs(graphs, labels=labels)

        assert 'y' in batched
        assert batched['y'].shape[0] == 2
        assert batched['y'][0].item() == 0
        assert batched['y'][1].item() == 1

    def test_batch_assignment_correct(self):
        """Test that batch indices correctly track graph membership"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(X)).
        cnf(c2, axiom, q(Y, a)).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = [state.clause_to_graph(id) for id in clause_ids]
        batched = graph_utils.batch_graphs(graphs)

        batch = batched['batch']

        # First graph's nodes should have index 0
        num_nodes_0 = graphs[0].num_nodes()
        assert torch.all(batch[:num_nodes_0] == 0)

        # Second graph's nodes should have index 1
        assert torch.all(batch[num_nodes_0:] == 1)

    def test_batch_different_sizes(self):
        """Test batching graphs of different sizes"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(X)).
        cnf(c2, axiom, (p(X) | q(Y))).
        cnf(c3, axiom, (p(X) | q(Y) | r(Z))).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = [state.clause_to_graph(id) for id in clause_ids]
        batched = graph_utils.batch_graphs(graphs)

        # Should handle different sizes correctly
        assert batched['num_graphs'] == 3
        assert batched['x'].shape[0] > 0

    def test_batch_empty_raises_error(self):
        """Test that batching empty list raises error"""
        with pytest.raises(ValueError):
            graph_utils.batch_graphs([])

    def test_batch_labels_mismatch_raises_error(self):
        """Test that label count mismatch raises error"""
        state = ProofState()
        tptp = """
        cnf(c1, axiom, p(X)).
        cnf(c2, axiom, q(Y)).
        """
        clause_ids = state.add_clauses_from_tptp(tptp)

        graphs = [state.clause_to_graph(id) for id in clause_ids]

        with pytest.raises(ValueError):
            graph_utils.batch_graphs(graphs, labels=[0])  # Only 1 label for 2 graphs


# ============================================================================
# Level 3: Advanced Formats - Specialized Tensor Formats
# ============================================================================

class TestAdvancedFormats:
    """Specialized tensor formats for specific use cases

    Covers advanced sparse matrix formats:
    - COO format (coordinate format)
    - CSR format (compressed sparse row)
    - Format validation
    """

    def test_sparse_coo_format(self):
        """Test COO (coordinate) format sparse adjacency"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        adj = graph_utils.to_sparse_adjacency(graph, format='coo')

        num_nodes = graph.num_nodes()
        assert adj.shape == (num_nodes, num_nodes)
        assert adj.is_sparse

    def test_sparse_csr_format(self):
        """Test CSR (compressed sparse row) format for fast computation"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        adj = graph_utils.to_sparse_adjacency(graph, format='csr')

        num_nodes = graph.num_nodes()
        assert adj.shape == (num_nodes, num_nodes)
        assert adj.layout == torch.sparse_csr

    def test_sparse_format_validation(self):
        """Test that invalid sparse format raises error"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])

        with pytest.raises(ValueError):
            graph_utils.to_sparse_adjacency(graph, format='invalid')


# ============================================================================
# Level 4: Post-Processing - Operations on GNN Outputs
# ============================================================================

class TestPostProcessing:
    """Operations on GNN outputs

    Covers graph-level aggregation after GNN forward pass:
    - Mean pooling (average node embeddings)
    - Sum pooling (sum node embeddings)
    - Max pooling (max over node embeddings)
    - Root pooling (use clause root embedding)
    """

    def test_mean_pooling(self):
        """Test mean pooling for graph-level embeddings"""
        # Create mock node embeddings
        node_emb = torch.randn(10, 16)  # 10 nodes, 16-dim embeddings
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])  # 3 graphs

        graph_emb = graph_utils.extract_graph_embeddings(
            node_emb, batch, method='mean'
        )

        assert graph_emb.shape == (3, 16)

        # Manually verify first graph
        expected_0 = node_emb[:4].mean(dim=0)
        torch.testing.assert_close(graph_emb[0], expected_0)

    def test_sum_pooling(self):
        """Test sum pooling for graph-level embeddings"""
        node_emb = torch.randn(6, 8)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])

        graph_emb = graph_utils.extract_graph_embeddings(
            node_emb, batch, method='sum'
        )

        assert graph_emb.shape == (2, 8)

        # Verify first graph
        expected_0 = node_emb[:3].sum(dim=0)
        torch.testing.assert_close(graph_emb[0], expected_0)

    def test_max_pooling(self):
        """Test max pooling for graph-level embeddings"""
        node_emb = torch.randn(6, 8)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])

        graph_emb = graph_utils.extract_graph_embeddings(
            node_emb, batch, method='max'
        )

        assert graph_emb.shape == (2, 8)

    def test_root_pooling(self):
        """Test root node pooling (use clause root)"""
        node_emb = torch.randn(6, 8)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])

        graph_emb = graph_utils.extract_graph_embeddings(
            node_emb, batch, method='root'
        )

        assert graph_emb.shape == (2, 8)

        # First graph's root should be node 0
        torch.testing.assert_close(graph_emb[0], node_emb[0])

        # Second graph's root should be node 3
        torch.testing.assert_close(graph_emb[1], node_emb[3])

    def test_pooling_method_validation(self):
        """Test that invalid pooling method raises error"""
        node_emb = torch.randn(4, 8)
        batch = torch.tensor([0, 0, 1, 1])

        with pytest.raises(ValueError):
            graph_utils.extract_graph_embeddings(
                node_emb, batch, method='invalid'
            )


# ============================================================================
# Level 5: Utilities - Helper and Analysis Functions
# ============================================================================

class TestUtilityFunctions:
    """Helper and analysis functions

    Covers utility functions for:
    - Node type filtering and masking
    - Graph statistics and analysis
    - Debugging and inspection
    """

    def test_node_type_masks_creation(self):
        """Test creation of boolean masks for each node type"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X, a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph)

        masks = graph_utils.get_node_type_masks(tensors['node_types'])

        # Check all mask types present
        assert 'clause' in masks
        assert 'literal' in masks
        assert 'predicate' in masks
        assert 'variable' in masks
        assert 'constant' in masks

    def test_node_type_masks_are_boolean(self):
        """Test that masks are boolean tensors"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph)

        masks = graph_utils.get_node_type_masks(tensors['node_types'])

        for mask in masks.values():
            assert mask.dtype == torch.bool

    def test_node_type_masks_select_correctly(self):
        """Test that masks select correct node types"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X, a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph)
        node_types = tensors['node_types']

        masks = graph_utils.get_node_type_masks(node_types)

        # Verify masks select correct type
        for type_idx, type_name in enumerate(['clause', 'literal', 'predicate',
                                               'function', 'variable', 'constant']):
            expected_mask = node_types == type_idx
            torch.testing.assert_close(masks[type_name], expected_mask)

    def test_node_type_mask_usage(self):
        """Test using masks to filter node features"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X, a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        tensors = graph_utils.to_torch_tensors(graph)

        masks = graph_utils.get_node_type_masks(tensors['node_types'])

        # Extract only variable features
        variable_features = tensors['x'][masks['variable']]
        assert variable_features.shape[1] == 13  # Feature dimension

        # Should have at least one variable
        assert variable_features.shape[0] > 0

    def test_graph_statistics_basic(self):
        """Test basic statistics computation"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(X))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        stats = graph_utils.compute_graph_statistics(graph)

        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'feature_dim' in stats
        assert stats['feature_dim'] == 13

    def test_graph_statistics_node_type_counts(self):
        """Test node type counts in statistics"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(f(X), a))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        stats = graph_utils.compute_graph_statistics(graph)

        # Should have counts for each type
        assert 'num_clauses' in stats
        assert 'num_literals' in stats
        assert 'num_predicates' in stats
        assert 'num_functions' in stats
        assert 'num_variables' in stats
        assert 'num_constants' in stats

        # Verify specific counts for this example
        assert stats['num_clauses'] == 1
        assert stats['num_literals'] == 1
        assert stats['num_predicates'] == 1
        assert stats['num_functions'] == 1
        assert stats['num_variables'] == 1
        assert stats['num_constants'] == 1

    def test_graph_statistics_depth(self):
        """Test max depth computation in statistics"""
        state = ProofState()
        tptp = "cnf(test, axiom, p(f(g(X))))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        stats = graph_utils.compute_graph_statistics(graph)

        assert 'max_depth' in stats
        assert stats['max_depth'] > 0

    def test_graph_statistics_usage_for_debugging(self):
        """Test using statistics for debugging and analysis"""
        state = ProofState()
        tptp = "cnf(test, axiom, (p(X) | q(Y) | r(f(Z))))."
        clause_ids = state.add_clauses_from_tptp(tptp)

        graph = state.clause_to_graph(clause_ids[0])
        stats = graph_utils.compute_graph_statistics(graph)

        # Should give useful debugging information
        assert stats['num_literals'] == 3  # Three literals in clause
        assert stats['num_variables'] == 3  # X, Y, Z
        assert stats['num_functions'] == 1  # f()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
