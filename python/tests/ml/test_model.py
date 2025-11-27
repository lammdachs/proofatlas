"""Tests for GNN model and training"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from proofatlas.ml import (
    ClauseGNN,
    ClauseGNNWithAttention,
    create_model,
    TrainingConfig,
    create_pyg_dataset,
    split_dataset,
    TrainingDataset,
)


class TestClauseGNN:
    """Test basic GNN model"""

    def test_forward_single_graph(self):
        """Forward pass on single graph"""
        model = ClauseGNN(node_feature_dim=20, hidden_dim=32, num_layers=2)

        # Create a simple graph
        x = torch.randn(5, 20)  # 5 nodes, 20 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # chain

        output = model(x, edge_index)
        assert output.shape == torch.Size([1])

    def test_forward_batched_graphs(self):
        """Forward pass on batched graphs"""
        model = ClauseGNN(node_feature_dim=20, hidden_dim=32, num_layers=2)

        # Create batch of graphs
        data1 = Data(x=torch.randn(3, 20), edge_index=torch.tensor([[0, 1], [1, 2]]))
        data2 = Data(x=torch.randn(4, 20), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        batch = Batch.from_data_list([data1, data2])

        output = model(batch.x, batch.edge_index, batch.batch)
        assert output.shape == torch.Size([2])  # One score per graph

    def test_predict_proba(self):
        """Test probability prediction"""
        model = ClauseGNN(node_feature_dim=20, hidden_dim=32, num_layers=2)
        model.eval()

        x = torch.randn(5, 20)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        proba = model.predict_proba(x, edge_index)
        assert proba.shape == torch.Size([1])
        assert 0 <= proba.item() <= 1

    def test_different_pooling(self):
        """Test different pooling methods"""
        for pooling in ["mean", "sum", "max"]:
            model = ClauseGNN(node_feature_dim=20, hidden_dim=32, pooling=pooling)

            x = torch.randn(5, 20)
            edge_index = torch.tensor([[0, 1], [1, 2]])

            output = model(x, edge_index)
            assert output.shape == torch.Size([1])

    def test_training_mode(self):
        """Test that dropout is applied in training mode"""
        model = ClauseGNN(node_feature_dim=20, hidden_dim=32, dropout=0.5)

        x = torch.randn(5, 20)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        # Train mode - dropout active
        model.train()
        outputs_train = [model(x, edge_index).item() for _ in range(10)]

        # Eval mode - dropout inactive
        model.eval()
        outputs_eval = [model(x, edge_index).item() for _ in range(10)]

        # Eval outputs should be identical
        assert all(o == outputs_eval[0] for o in outputs_eval)


class TestClauseGNNWithAttention:
    """Test attention-based GNN model"""

    def test_forward_single_graph(self):
        """Forward pass on single graph"""
        model = ClauseGNNWithAttention(node_feature_dim=20, hidden_dim=32, num_layers=2)

        x = torch.randn(5, 20)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

        output = model(x, edge_index)
        assert output.shape == torch.Size([1])

    def test_forward_batched(self):
        """Forward pass on batched graphs"""
        model = ClauseGNNWithAttention(node_feature_dim=20, hidden_dim=32, num_layers=2)

        data1 = Data(x=torch.randn(3, 20), edge_index=torch.tensor([[0, 1], [1, 2]]))
        data2 = Data(x=torch.randn(4, 20), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        batch = Batch.from_data_list([data1, data2])

        output = model(batch.x, batch.edge_index, batch.batch)
        assert output.shape == torch.Size([2])


class TestCreateModel:
    """Test model factory function"""

    def test_create_gcn(self):
        """Create GCN model"""
        model = create_model("gcn", hidden_dim=64, num_layers=3)
        assert isinstance(model, ClauseGNN)

    def test_create_gcn_attention(self):
        """Create attention GCN model"""
        model = create_model("gcn_attention", hidden_dim=64, num_layers=2)
        assert isinstance(model, ClauseGNNWithAttention)

    def test_unknown_model_type_raises(self):
        """Unknown model type raises error"""
        with pytest.raises(ValueError):
            create_model("unknown_model")


class TestDatasetPreparation:
    """Test dataset preparation for training"""

    def test_create_pyg_dataset(self):
        """Convert TrainingDataset to PyG format"""
        # Create mock training dataset
        graphs = [
            {"x": torch.randn(3, 20), "edge_index": torch.randint(0, 3, (2, 4))},
            {"x": torch.randn(4, 20), "edge_index": torch.randint(0, 4, (2, 5))},
        ]
        labels = torch.tensor([1.0, 0.0])

        dataset = TrainingDataset(
            graphs=graphs,
            labels=labels,
            problem_names=["prob1", "prob2"],
            clause_ids=[0, 1],
        )

        pyg_data = create_pyg_dataset(dataset)

        assert len(pyg_data) == 2
        assert pyg_data[0].y.item() == 1.0
        assert pyg_data[1].y.item() == 0.0

    def test_split_dataset(self):
        """Test dataset splitting"""
        # Create dummy data
        data = [Data(x=torch.randn(3, 20), edge_index=torch.randint(0, 3, (2, 2)))
                for _ in range(100)]

        train, val, test = split_dataset(data, train_split=0.8, val_split=0.1)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


class TestTrainingConfig:
    """Test training configuration"""

    def test_default_config(self):
        """Default configuration values"""
        config = TrainingConfig()

        assert config.model_type == "gcn"
        assert config.hidden_dim == 64
        assert config.num_layers == 2
        assert config.epochs == 100

    def test_custom_config(self):
        """Custom configuration"""
        config = TrainingConfig(
            hidden_dim=128,
            num_layers=4,
            epochs=50,
        )

        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.epochs == 50


class TestGradientFlow:
    """Test that gradients flow correctly through the model"""

    def test_gradient_flow_gcn(self):
        """Gradients flow through GCN"""
        model = ClauseGNN(node_feature_dim=20, hidden_dim=32, num_layers=2)

        x = torch.randn(5, 20, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        output = model(x, edge_index)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_gradient_flow_attention(self):
        """Gradients flow through attention GCN"""
        model = ClauseGNNWithAttention(node_feature_dim=20, hidden_dim=32, num_layers=2)

        x = torch.randn(5, 20, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [1, 2]])

        output = model(x, edge_index)
        loss = output.sum()
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
