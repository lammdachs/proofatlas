# Proofs Module Documentation

The `proofs` module provides data structures for representing proof states and complete proofs in theorem proving.

## Overview

The proofs module consists of three main components:

1. **state.py** - Proof state tracking (processed and unprocessed clauses)
2. **proof.py** - Complete proof representation with history
3. **serialization.py** - JSON serialization for proof objects

## Module Structure

### state.py

This module provides the `ProofState` class for tracking the state of a proof search.

#### ProofState

Represents the state of the given clause algorithm with two sets of clauses:
- **processed**: Clauses that have been selected and used for inference
- **unprocessed**: Clauses available for selection

**Key Methods:**
- `add_unprocessed(clause)` - Add a new clause to unprocessed
- `add_processed(clause)` - Add a clause to processed  
- `move_to_processed(clause)` - Move clause from unprocessed to processed
- `all_clauses` - Property returning all clauses (processed + unprocessed)

#### Usage Example

```python
from proofatlas.proofs import ProofState
from proofatlas.core import Clause, Literal, Predicate, Constant

# Create initial state with axioms
initial_state = ProofState([], [clause1, clause2, clause3])

# Move a clause to processed
initial_state.move_to_processed(clause1)

# Add a derived clause
initial_state.add_unprocessed(derived_clause)

# Access all clauses
all_clauses = initial_state.all_clauses
```

### proof.py

This module provides classes for representing complete proofs.

#### ProofStep

A single step in a proof, storing:
- `state`: The ProofState at this step
- `selected_clause`: Index of the selected clause (optional)
- `applied_rules`: List of RuleApplication objects
- `metadata`: Dictionary for additional information

#### Proof

Represents a complete proof as a list of ProofStep objects where:
- Each step contains a state, optional selected_clause, applied rules, and metadata
- The last step always has `selected_clause = None` (representing the final state)
- `final_state` property returns the state from the last step

**Key Methods:**
- `add_step(state, selected_clause, applied_rules, **metadata)` - Add a new proof step
  - If last step has no selection, it's replaced
  - Automatically maintains the invariant that last step has no selection
- `finalize(final_state)` - Ensure proof ends with a step with no selection
- `get_selected_clauses()` - Get list of all selected clause indices
- `get_metadata_history(key)` - Get history of a metadata value
- `is_complete` - Check if proof found empty clause
- `is_saturated` - Check if no unprocessed clauses remain
- `length` - Number of inference steps (excluding final step if it has no selection)

#### Usage Example

```python
from proofatlas.proofs import Proof, ProofState
from proofatlas.rules import ResolutionRule

# Create proof with initial state
initial_state = ProofState([], [clause1, clause2])
proof = Proof(initial_state)

# Apply resolution
rule = ResolutionRule()
new_state = ProofState([clause1], [clause2, clause3])
result = rule.apply(new_state, [0, 1])

# Add step with rule application
proof.add_step(
    new_state, 
    selected_clause=0,
    applied_rules=[result] if result else [],
    rule="given_clause"
)

# Check if proof is complete
if proof.is_complete:
    print("Found contradiction!")
```

### serialization.py

Provides JSON serialization for proof objects:

- **ProofJSONEncoder** - Encodes ProofState, Proof, ProofStep, and RuleApplication
- **ProofJSONDecoder** - Decodes JSON back to proof objects
- **proof_to_json(proof)** - Convert proof to JSON string
- **proof_from_json(json_str)** - Create proof from JSON string
- **save_proof(proof, file_path)** - Save proof to file
- **load_proof(file_path)** - Load proof from file

## Design Principles

1. **Separation of State and History**: ProofState represents a snapshot, Proof represents the full history
2. **Immutability of States**: Each ProofStep has its own ProofState instance
3. **Flexible Metadata**: Both ProofStep and RuleApplication support arbitrary metadata
4. **Clean Invariants**: The last step in a Proof always has no selection

## Integration

The proofs module integrates with:
- **Rules**: Rules operate on ProofState and return RuleApplication objects
- **Loops**: Loops orchestrate the proof search, creating ProofSteps
- **Navigator**: Visualizes proof steps and state transitions
- **Serialization**: Full JSON support for saving/loading proofs