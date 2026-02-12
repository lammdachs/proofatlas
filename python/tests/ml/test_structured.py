"""Tests for structured clause format conversion."""

import pytest
import json
from pathlib import Path

# Skip if proofatlas not available
pytest.importorskip("proofatlas")


@pytest.fixture
def sample_clause():
    """A sample structured clause."""
    return {
        "literals": [
            {
                "polarity": True,
                "atom": {
                    "predicate": "=",
                    "args": [
                        {
                            "type": "Function",
                            "name": "mult",
                            "args": [
                                {"type": "Variable", "name": "X"},
                                {"type": "Variable", "name": "Y"},
                            ],
                        },
                        {"type": "Variable", "name": "Z"},
                    ],
                },
            }
        ],
        "label": 1,
        "age": 0,
        "role": "axiom",
    }


@pytest.fixture
def sample_multi_literal_clause():
    """A clause with multiple literals."""
    return {
        "literals": [
            {
                "polarity": True,
                "atom": {
                    "predicate": "p",
                    "args": [{"type": "Variable", "name": "X"}],
                },
            },
            {
                "polarity": False,
                "atom": {
                    "predicate": "q",
                    "args": [
                        {"type": "Variable", "name": "X"},
                        {"type": "Constant", "name": "c"},
                    ],
                },
            },
        ],
        "label": 0,
        "age": 5,
        "role": "derived",
    }


class TestClauseToString:
    """Test structured clause to string conversion."""

    def test_equality_clause(self, sample_clause):
        from proofatlas.ml.structured import clause_to_string

        result = clause_to_string(sample_clause)
        assert result == "mult(X, Y) = Z"

    def test_multi_literal_clause(self, sample_multi_literal_clause):
        from proofatlas.ml.structured import clause_to_string

        result = clause_to_string(sample_multi_literal_clause)
        assert result == "p(X) | ~q(X, c)"

    def test_empty_clause(self):
        from proofatlas.ml.structured import clause_to_string

        result = clause_to_string({"literals": []})
        assert result == "[]"


class TestClausesConversion:
    """Test batch conversion functions."""

    def test_clauses_to_strings(self, sample_clause, sample_multi_literal_clause):
        from proofatlas.ml.structured import clauses_to_strings

        clauses = [sample_clause, sample_multi_literal_clause]
        results = clauses_to_strings(clauses)

        assert len(results) == 2
        assert results[0] == "mult(X, Y) = Z"
        assert results[1] == "p(X) | ~q(X, c)"
