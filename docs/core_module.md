# Core Module Documentation

The `core` module provides the fundamental data structures for theorem proving in first-order logic.

## Overview

The core module focuses exclusively on first-order logic representation:

1. **logic.py** - First-order logic representation (terms, literals, clauses, problems)
2. **serialization.py** - JSON serialization for logic objects

Note: The `ProofState` class has been moved to the `proofs` module along with `Proof` and `ProofStep` to better organize proof-related functionality.

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
  - `Clause(*literals)` - Note: takes unpacked literals, not a list
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

## Related Modules

### proofs module

The proof-related classes have been moved to a separate `proofs` module:
- **ProofState** - Tracks processed/unprocessed clauses during proof search
- **Proof** and **ProofStep** - Represent complete proofs with history
See the proofs module documentation for details.

### rules

The `rules` module provides modular inference rules that operate on proof states:

- **Rule** - Abstract base class for inference rules
- **RuleApplication** - Dataclass containing rule application results
- **ResolutionRule** - Binary resolution
- **FactoringRule** - Factoring (unifying literals within a clause)
- **SubsumptionRule** - Subsumption elimination

Rules operate on ProofState objects and return RuleApplication objects containing:
- `rule_name`: Name of the rule applied
- `parents`: Indices of parent clauses used
- `generated_clauses`: New clauses created
- `deleted_clause_indices`: Clauses to be removed
- `metadata`: Additional information about the application

Example:
```python
from proofatlas.rules import ResolutionRule
from proofatlas.core.state import ProofState

# Create a state with some clauses
state = ProofState(processed=[clause1, clause2], unprocessed=[])

# Apply resolution
rule = ResolutionRule()
result = rule.apply(state, [0, 1])  # Resolve clauses at indices 0 and 1

if result:
    print(f"Generated {len(result.generated_clauses)} new clauses")
```

## Design Principles

1. **Immutability**: Logic objects (Terms, Literals, Clauses) are immutable
2. **Separation of Concerns**: Logic representation is separate from proof search
3. **Flexibility**: The metadata dictionary in ProofStep allows storing any additional information without changing the core structure
4. **Simplicity**: Only essential fields are part of the core data structures

## JSON Serialization

The core module supports JSON serialization for all major objects:

```python
from proofatlas.core import (
    save_problem, load_problem,
    save_proof, load_proof,
    problem_to_json, problem_from_json,
    proof_to_json, proof_from_json
)

# Save/load problems
save_problem(problem, "problem.json")
loaded_problem = load_problem("problem.json")

# Save/load proofs
save_proof(proof, "proof.json")
loaded_proof = load_proof("proof.json")

# Convert to/from JSON strings
json_str = problem_to_json(problem)
problem = problem_from_json(json_str)
```

JSON files preserve:
- Complete problem/proof structure
- Variable and constant names
- Metadata in proof steps
- All clause and literal information

## Integration

The core module serves as the foundation for:
- **Rules**: Apply inference rules to ProofState objects
- **Proofs**: Track sequences of ProofSteps with rule applications
- **Loops**: Implement the given clause algorithm using ProofState and Rules
- **Selectors**: Choose which clause to process next from ProofState.unprocessed
- **Navigator**: Visualize proof steps in terminal with clean UI
- **Data formats**: Encode ProofState for machine learning models
- **File formats**: Import/export problems in various formats
- **Data storage**: Serialize problems and proofs for datasets