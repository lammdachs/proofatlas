"""Tests for data problemset module."""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from proofatlas.data.problemset import Problemset
from proofatlas.data.config import DatasetConfig, DatasetSplit
from proofatlas.core.logic import Problem, Clause, Literal, Predicate, Constant
from proofatlas.core.state import ProofState


class TestProblemset:
    """Test Problemset class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock dataset configuration."""
        return DatasetConfig(
            name="test_dataset",
            file_format="tptp",
            data_format="graph",
            base_path=Path("/tmp/test"),
            splits=[
                DatasetSplit(
                    name="train",
                    files=["problem1.p", "problem2.p"],
                    patterns=["*.p"]
                ),
                DatasetSplit(
                    name="val",
                    files=["val1.p"]
                )
            ]
        )
    
    @pytest.fixture
    def mock_problem(self):
        """Create a mock problem."""
        a = Constant("a")
        P = Predicate("p", 1)
        Q = Predicate("q", 1)
        
        return Problem(
            Clause(Literal(P(a), True)),
            Clause(Literal(P(a), False), Literal(Q(a), True)),
            Clause(Literal(Q(a), False))
        )
    
    def test_init(self, mock_config):
        """Test initialization of Problemset."""
        with patch('proofatlas.data.problemset.get_format_handler') as mock_handler, \
             patch('proofatlas.data.problemset.get_data_format') as mock_format:
            
            mock_handler.return_value = Mock()
            mock_format.return_value = Mock()
            
            dataset = Problemset(mock_config, 'train')
            
            assert dataset.config == mock_config
            assert dataset.split_name == 'train'
            assert dataset.split.name == 'train'
            
            mock_handler.assert_called_once_with('tptp')
            mock_format.assert_called_once_with('graph')
    
    def test_init_invalid_split(self, mock_config):
        """Test initialization with invalid split name."""
        with patch('proofatlas.data.problemset.get_format_handler'), \
             patch('proofatlas.data.problemset.get_data_format'):
            
            with pytest.raises(ValueError, match="Split 'invalid' not found"):
                Problemset(mock_config, 'invalid')
    
    def test_collect_problem_files(self, mock_config):
        """Test collecting problem files."""
        with patch('proofatlas.data.problemset.get_format_handler'), \
             patch('proofatlas.data.problemset.get_data_format'), \
             patch('proofatlas.data.problemset.glob.glob') as mock_glob, \
             patch('pathlib.Path.exists') as mock_exists:
            
            # Mock file existence
            mock_exists.return_value = True
            mock_glob.return_value = ['/tmp/test/extra1.p', '/tmp/test/extra2.p']
            
            dataset = Problemset(mock_config, 'train')
            
            # Should have files from both explicit list and pattern
            assert len(dataset.problem_files) == 4
            assert Path('/tmp/test/problem1.p') in dataset.problem_files
            assert Path('/tmp/test/problem2.p') in dataset.problem_files
            assert Path('/tmp/test/extra1.p') in dataset.problem_files
            assert Path('/tmp/test/extra2.p') in dataset.problem_files
    
    def test_len(self, mock_config):
        """Test dataset length."""
        with patch('proofatlas.data.problemset.get_format_handler'), \
             patch('proofatlas.data.problemset.get_data_format'), \
             patch.object(Problemset, '_collect_problem_files') as mock_collect:
            
            mock_collect.return_value = [Path('1.p'), Path('2.p'), Path('3.p')]
            
            dataset = Problemset(mock_config, 'train')
            assert len(dataset) == 3
    
    def test_get_problem(self, mock_config, mock_problem):
        """Test getting a problem by index."""
        with patch('proofatlas.data.problemset.get_format_handler') as mock_handler, \
             patch('proofatlas.data.problemset.get_data_format'), \
             patch.object(Problemset, '_collect_problem_files') as mock_collect:
            
            mock_file_handler = Mock()
            mock_file_handler.parse_file.return_value = mock_problem
            mock_handler.return_value = mock_file_handler
            
            mock_collect.return_value = [Path('1.p'), Path('2.p')]
            
            dataset = Problemset(mock_config, 'train')
            
            # First call should parse the file
            problem1 = dataset.get_problem(0)
            assert problem1 == mock_problem
            mock_file_handler.parse_file.assert_called_once_with(Path('1.p'))
            
            # Second call should use cache
            problem1_cached = dataset.get_problem(0)
            assert problem1_cached == mock_problem
            assert mock_file_handler.parse_file.call_count == 1  # Not called again
    
    def test_get_proof_state(self, mock_config, mock_problem):
        """Test getting initial proof state."""
        with patch('proofatlas.data.problemset.get_format_handler'), \
             patch('proofatlas.data.problemset.get_data_format'), \
             patch.object(Problemset, '_collect_problem_files') as mock_collect, \
             patch.object(Problemset, 'get_problem') as mock_get_problem:
            
            mock_collect.return_value = [Path('1.p')]
            mock_get_problem.return_value = mock_problem
            
            dataset = Problemset(mock_config, 'train')
            
            state = dataset.get_proof_state(0)
            
            assert isinstance(state, ProofState)
            assert len(state.processed) == 0
            assert len(state.unprocessed) == 3  # Three clauses in mock_problem
            assert state.unprocessed == list(mock_problem.clauses)
    
    def test_get_clauses(self, mock_config, mock_problem):
        """Test getting clauses for a problem."""
        with patch('proofatlas.data.problemset.get_format_handler'), \
             patch('proofatlas.data.problemset.get_data_format'), \
             patch.object(Problemset, '_collect_problem_files') as mock_collect, \
             patch.object(Problemset, 'get_problem') as mock_get_problem:
            
            mock_collect.return_value = [Path('1.p')]
            mock_get_problem.return_value = mock_problem
            
            dataset = Problemset(mock_config, 'train')
            
            clauses = dataset.get_clauses(0)
            
            assert isinstance(clauses, list)
            assert len(clauses) == 3
            assert all(isinstance(c, Clause) for c in clauses)
            assert clauses == list(mock_problem.clauses)
    
    def test_getitem(self, mock_config, mock_problem):
        """Test getting an item for training."""
        with patch('proofatlas.data.problemset.get_format_handler'), \
             patch('proofatlas.data.problemset.get_data_format') as mock_format_getter, \
             patch.object(Problemset, '_collect_problem_files') as mock_collect, \
             patch.object(Problemset, 'get_problem') as mock_get_problem, \
             patch.object(Problemset, 'get_proof_state') as mock_get_state:
            
            # Setup mocks
            mock_data_format = Mock()
            mock_data_format.encode_state.return_value = {'encoded': 'data'}
            mock_format_getter.return_value = mock_data_format
            
            mock_collect.return_value = [Path('/tmp/test/problem1.p')]
            mock_get_problem.return_value = mock_problem
            
            mock_state = ProofState([], list(mock_problem.clauses))
            mock_get_state.return_value = mock_state
            
            dataset = Problemset(mock_config, 'train')
            
            encoded, metadata = dataset[0]
            
            assert encoded == {'encoded': 'data'}
            assert metadata['problem_file'] == '/tmp/test/problem1.p'
            assert metadata['problem_name'] == 'problem1'
            assert metadata['num_clauses'] == 3
            assert metadata['num_processed'] == 0
            assert metadata['num_unprocessed'] == 3
            
            mock_data_format.encode_state.assert_called_once_with(mock_state)
    
    def test_clear_cache(self, mock_config):
        """Test clearing the cache."""
        with patch('proofatlas.data.problemset.get_format_handler'), \
             patch('proofatlas.data.problemset.get_data_format'), \
             patch.object(Problemset, '_collect_problem_files') as mock_collect:
            
            mock_collect.return_value = [Path('1.p')]
            
            dataset = Problemset(mock_config, 'train')
            
            # Add some items to cache
            dataset._problem_cache[0] = Mock()
            dataset._state_cache[0] = Mock()
            
            # Clear cache
            dataset.clear_cache()
            
            assert len(dataset._problem_cache) == 0
            assert len(dataset._state_cache) == 0