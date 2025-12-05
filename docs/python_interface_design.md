# Python Interface Design for ProofAtlas

## Overview

This document outlines the design for a Python interface to the Rust-based ProofAtlas theorem prover. The interface will allow Python users to:

1. Create and manage proof states
2. Apply inference rules step-by-step
3. Inspect clauses, literals, and terms
4. Control the saturation process
5. Analyze proof search progress

## Core Components

### 1. ProofState Management

```python
from proofatlas import ProofState, Clause, Literal, Term

# Create a new proof state
state = ProofState()

# Add clauses to the state
state.add_clause(Clause.from_string("P(a) | Q(b)"))
state.add_clause(Clause.from_string("~P(a)"))

# Get processed and unprocessed clauses
processed = state.get_processed_clauses()
unprocessed = state.get_unprocessed_clauses()

# Select next given clause
given_clause = state.select_given_clause(strategy="age_weight_ratio")
```

### 2. Clause and Term Inspection

```python
# Inspect a clause
clause = state.get_clause(0)
print(f"Clause: {clause}")
print(f"Literals: {clause.literals}")
print(f"Variables: {clause.variables}")
print(f"Is unit clause: {clause.is_unit}")
print(f"Is horn clause: {clause.is_horn}")

# Inspect literals
for lit in clause.literals:
    print(f"  Literal: {lit}")
    print(f"  Polarity: {lit.polarity}")
    print(f"  Predicate: {lit.predicate}")
    print(f"  Arguments: {lit.arguments}")

# Inspect terms
for term in lit.arguments:
    print(f"    Term: {term}")
    print(f"    Type: {term.term_type}")  # Variable, Constant, Function
    if term.is_function():
        print(f"    Function: {term.function_symbol}")
        print(f"    Args: {term.arguments}")
```

### 3. Inference Rule Application

```python
from proofatlas import Resolution, Factoring, Superposition
from proofatlas import LiteralSelector, ClauseSelector

# Configure literal selection (0=all, 20=maximal, 21=unique, 22=neg max-weight)
literal_selector = LiteralSelector.sel20()  # or .sel0() for no selection

# Apply resolution between two clauses
resolution = Resolution(literal_selector)
results = resolution.apply(state, clause_idx1=0, clause_idx2=1)

for result in results:
    print(f"Generated clause: {result.clause}")
    print(f"From parents: {result.parents}")
    
# Apply factoring to a single clause
factoring = Factoring(literal_selector)
results = factoring.apply(state, clause_idx=0)

# Apply superposition (if equality present)
superposition = Superposition(literal_selector)
results = superposition.apply(state, clause_idx1=0, clause_idx2=1)
```

### 4. Saturation Control

```python
from proofatlas import SaturationConfig, saturate_step

# Configure saturation
config = SaturationConfig(
    literal_selection="20",  # 0=all, 20=maximal, 21=unique, 22=neg_max_weight
    clause_selection="age_weight_ratio",
    age_weight_ratio=(1, 5),
    simplify_generated=True,
    forward_subsumption=True,
    backward_subsumption=False  # Not yet implemented
)

# Single-step saturation
result = saturate_step(state, config)
if result.proof_found:
    print(f"Proof found! Empty clause derived")
    print(f"Proof: {result.proof}")
elif result.saturated:
    print("Saturated without proof")
else:
    print(f"Generated {len(result.new_clauses)} new clauses")
    
# Full saturation with step limit
result = saturate(state, config, max_steps=1000)
```

### 5. Proof Analysis

```python
from proofatlas import ProofAnalyzer

analyzer = ProofAnalyzer(state)

# Statistics
stats = analyzer.get_statistics()
print(f"Total clauses: {stats.total_clauses}")
print(f"Processed: {stats.processed_count}")
print(f"Unprocessed: {stats.unprocessed_count}")
print(f"Generated: {stats.generated_count}")
print(f"Deleted by subsumption: {stats.subsumed_count}")

# Proof search tree visualization
tree = analyzer.get_proof_tree()
tree.display()

# Clause derivation history
history = analyzer.get_clause_history(clause_idx=10)
for step in history:
    print(f"Step {step.number}: {step.rule} from {step.parents}")
```

### 6. TPTP Integration

```python
from proofatlas import parse_tptp_file, parse_tptp_string

# Parse TPTP file
problem = parse_tptp_file("problem.p")
state = ProofState.from_problem(problem)

# Parse TPTP string
tptp_content = """
cnf(c1, axiom, p(a) | q(b)).
cnf(c2, axiom, ~p(a)).
cnf(goal, negated_conjecture, ~q(b)).
"""
problem = parse_tptp_string(tptp_content)
```

### 7. Advanced Features

```python
# Custom term ordering
from proofatlas import KBO, LPO

kbo = KBO(variable_weight=1, symbol_weights={"f": 2, "g": 3})
state.set_term_ordering(kbo)

# Literal selection functions
def custom_selector(clause):
    # Select negative literals with maximum depth
    return [i for i, lit in enumerate(clause.literals) 
            if not lit.polarity and lit.max_depth() == clause.max_depth()]

state.set_literal_selector(custom_selector)

# Clause evaluation for machine learning
from proofatlas import ClauseFeatures

features = ClauseFeatures(clause)
print(f"Symbol count: {features.symbol_count}")
print(f"Variable count: {features.variable_count}")
print(f"Depth: {features.depth}")
print(f"Weight: {features.weight}")
```

## Implementation Strategy

### Phase 1: Core Bindings
- Expose ProofState, Clause, Literal, Term as Python classes
- Implement basic inspection methods
- Add TPTP parsing functions

### Phase 2: Inference Rules
- Wrap inference rules as Python classes
- Support literal selection configuration
- Return inference results with parent tracking

### Phase 3: Saturation Control  
- Expose single-step saturation
- Add configuration options
- Implement statistics collection

### Phase 4: Advanced Features
- Custom selectors and orderings
- Proof tree visualization
- Feature extraction for ML

## Example Usage

```python
from proofatlas import *

# Load a problem
problem = parse_tptp_file("GRP001-1.p")
state = ProofState.from_problem(problem)

# Configure saturation
config = SaturationConfig(
    literal_selection="20",  # 0=all, 20=maximal, 21=unique, 22=neg_max_weight
    clause_selection="age_weight_ratio",
    age_weight_ratio=(1, 5)
)

# Run saturation with inspection
for step in range(1000):
    # Get current statistics
    print(f"\nStep {step}: {len(state.processed)} processed, {len(state.unprocessed)} unprocessed")
    
    # Perform one saturation step
    result = saturate_step(state, config)
    
    if result.proof_found:
        print("Proof found!")
        print(result.proof)
        break
    elif result.saturated:
        print("Saturated without proof")
        break
    
    # Inspect generated clauses
    for clause in result.new_clauses:
        print(f"  New: {clause}")
```

## Benefits

1. **Interactive Development**: Experiment with proof strategies in Jupyter notebooks
2. **Machine Learning**: Extract features and train clause selection models
3. **Debugging**: Step through proof search to understand behavior
4. **Education**: Teach automated theorem proving concepts
5. **Research**: Prototype new inference rules and strategies

## Technical Considerations

1. **Zero-Copy**: Where possible, avoid copying data between Rust and Python
2. **Safety**: Ensure Rust's ownership rules are respected in Python API
3. **Performance**: Keep hot paths in Rust, expose high-level operations
4. **Pythonic**: Follow Python naming conventions and idioms
5. **Type Hints**: Provide comprehensive type annotations for IDE support