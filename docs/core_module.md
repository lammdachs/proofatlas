# Core Module Documentation

The `core` module provides the fundamental data structures for theorem proving in first-order logic.

## Overview

The core module consists of three main components:

1. **logic.py** - First-order logic representation (terms, literals, clauses, problems)
2. **state.py** - Proof state tracking (processed and unprocessed clauses)
3. **proof.py** - Proof representation (sequence of states and clause selections)

## Module Structure

### logic.py

This module implements the basic building blocks of first-order logic in CNF (Conjunctive Normal Form).

#### Classes

- **Symbol Classes**
  - `Function(name, arity)` - Function symbols
  - `Predicate(name, arity)` - Predicate symbols
  - `Variable(name)` - Variables (arity is always 0)
  - `Constant(name)` - Constants (special case of Function with arity 0)

- **Term** - Represents a term in first-order logic
  - Created by applying a function/constant to arguments: `f(x, a)`
  - Variables and constants are also terms

- **Literal** - An atomic formula with polarity
  - `Literal(predicate_term, polarity)`
  - Example: `Literal(P(x), True)` for P(x), `Literal(P(x), False)` for ~P(x)

- **Clause** - A disjunction of literals (CNF clause)
  - `Clause(*literals)`
  - Empty clause represents contradiction
  - Example: `Clause(Literal(P(x), False), Literal(Q(x), True))` for ~P(x) | Q(x)

- **Problem** - A collection of clauses
  - `Problem(*clauses)`
  - Represents a theorem proving problem in CNF

#### Usage Example

```python
from proofatlas.core.logic import Variable, Constant, Predicate, Literal, Clause, Problem

# Create symbols
x = Variable("X")
a = Constant("a")
P = Predicate("P", 1)
Q = Predicate("Q", 1)

# Create clauses
c1 = Clause(Literal(P(a), True))  # P(a)
c2 = Clause(Literal(P(x), False), Literal(Q(x), True))  # ~P(X) | Q(X)
c3 = Clause(Literal(Q(a), False))  # ~Q(a)

# Create problem
problem = Problem(c1, c2, c3)
```

### state.py

This module provides the `ProofState` class for tracking the state of a proof search.

#### ProofState

Represents the state of the given clause algorithm with two sets of clauses:
- **processed**: Clauses that have been selected and used for inference
- **unprocessed**: Clauses available for selection

**Key Methods:**
- `add_unprocessed(clause)` - Add a new clause to unprocessed
- `move_to_processed(clause)` - Move clause from unprocessed to processed
- `all_clauses` - Property returning all clauses (processed + unprocessed)

#### Usage Example

```python
from proofatlas.core.state import ProofState

# Create initial state with axioms
initial_state = ProofState([], [clause1, clause2, clause3])

# Move a clause to processed
initial_state.move_to_processed(clause1)

# Add a derived clause
initial_state.add_unprocessed(derived_clause)
```

### proof.py

This module provides classes for representing complete proofs.

#### ProofStep

A single step in a proof, storing:
- `state`: The ProofState at this step
- `selected_clause`: Index of the selected clause (optional)
- `metadata`: Dictionary for additional information (rules, scores, etc.)

#### Proof

Represents a complete proof as a list of ProofStep objects where:
- Each step contains a state, optional selected_clause, and metadata
- The last step always has `selected_clause = None` (representing the final state)
- `final_state` property returns the state from the last step

**Key Methods:**
- `add_step(state, selected_clause, **metadata)` - Add a new proof step
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
from proofatlas.core.proof import Proof, ProofStep

# Create proof with initial state
proof = Proof(initial_state)

# Add steps as proof progresses
new_state = ProofState([clause1], [clause2, clause3])
proof.add_step(
    new_state, 
    selected_clause=0,
    rule="given_clause"
)

# Add resolution step with metadata
proof.add_step(
    next_state,
    selected_clause=1, 
    rule="resolution",
    parent_clauses=[0, 1],
    generated_clause=resolvent
)

# Check if proof is complete
if proof.is_complete:
    print("Found contradiction!")
```

## Design Principles

1. **Immutability**: Logic objects (Terms, Literals, Clauses) are immutable
2. **Separation of Concerns**: Logic representation is separate from proof search
3. **Flexibility**: The metadata dictionary in ProofStep allows storing any additional information without changing the core structure
4. **Simplicity**: Only essential fields are part of the core data structures

## Integration

The core module serves as the foundation for:
- **Loops**: Implement the given clause algorithm using ProofState
- **Selectors**: Choose which clause to process next from ProofState.unprocessed
- **Rules**: Generate new clauses and update the proof state
- **Data formats**: Encode ProofState for machine learning models