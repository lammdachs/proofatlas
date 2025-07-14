# ProofState Python API Design

## Overview

The ProofState is the central data structure that tracks the state of a saturation-based proof search. This document details the Python API for creating, manipulating, and inspecting ProofState objects.

## ProofState Class

### Construction

```python
from proofatlas import ProofState, Clause

# Create empty proof state
state = ProofState()

# Create from list of clauses
clauses = [
    Clause.from_string("P(a) | Q(b)"),
    Clause.from_string("~P(a)"),
    Clause.from_string("~Q(b)")
]
state = ProofState(clauses)

# Create from TPTP problem
problem = parse_tptp_file("problem.p")
state = ProofState.from_problem(problem)
```

### Basic Properties

```python
# Clause counts
len(state)                    # Total number of clauses
state.num_processed          # Number of processed clauses  
state.num_unprocessed        # Number of unprocessed clauses
state.num_clauses            # Total clauses ever created

# Check state
state.is_saturated()         # True if no unprocessed clauses
state.contains_empty_clause() # True if contradiction found
state.proof_found()          # Alias for contains_empty_clause()
```

### Clause Access

```python
# Get all clauses
all_clauses = state.clauses()          # List of all clauses
processed = state.processed_clauses()   # List of processed clauses
unprocessed = state.unprocessed_clauses() # List of unprocessed clauses

# Get specific clause by ID
clause = state.get_clause(42)          # Get clause with ID 42
clause = state[42]                     # Alternative syntax

# Check if clause exists
if 42 in state:
    clause = state[42]

# Iterate over clauses
for clause in state:
    print(clause)

# Get clause metadata
meta = state.get_metadata(42)
print(f"Parents: {meta.parents}")
print(f"Rule: {meta.inference_rule}")
print(f"Selected literals: {meta.selected_literals}")
```

### Clause Management

```python
# Add new clauses
clause_id = state.add_clause(clause)
clause_ids = state.add_clauses([clause1, clause2, clause3])

# Add with metadata
clause_id = state.add_clause(
    clause, 
    parents=[0, 1],
    rule="resolution",
    selected_literals={0: [0], 1: [1]}
)

# Move clause from unprocessed to processed
state.process_clause(clause_id)

# Get next unprocessed clause (FIFO)
next_id = state.next_unprocessed()

# Delete clause (mark as deleted)
state.delete_clause(clause_id)

# Check if clause is deleted
if state.is_deleted(clause_id):
    print(f"Clause {clause_id} was deleted")
```

### Selection Strategies

```python
# Select given clause using different strategies
given_id = state.select_given_clause()  # Default FIFO
given_id = state.select_given_clause(strategy="smallest")
given_id = state.select_given_clause(strategy="oldest")
given_id = state.select_given_clause(strategy="age_weight_ratio", ratio=(1, 5))

# Custom selection function
def my_selector(state, unprocessed_ids):
    # Return ID of clause with fewest literals
    return min(unprocessed_ids, key=lambda id: len(state[id]))

given_id = state.select_given_clause(strategy=my_selector)
```

### Inference Application

```python
# Apply inference rules and update state
from proofatlas import apply_resolution, apply_factoring

# Resolution between two clauses
new_clauses = apply_resolution(state, clause1_id, clause2_id)
for clause, metadata in new_clauses:
    state.add_clause(clause, **metadata)

# Factoring on single clause  
new_clauses = apply_factoring(state, clause_id)
for clause, metadata in new_clauses:
    state.add_clause(clause, **metadata)

# Batch inference generation
from proofatlas import generate_inferences

# Generate all inferences with given clause
inferences = generate_inferences(
    state, 
    given_clause_id,
    rules=["resolution", "factoring", "superposition"],
    literal_selection="max_weight"
)

# Add generated clauses to state
for inference in inferences:
    state.add_clause(
        inference.clause,
        parents=inference.parents,
        rule=inference.rule_name
    )
```

### Simplification

```python
# Forward simplification of new clauses
simplified = state.simplify_forward(clause)
if simplified is None:
    print("Clause is tautology or subsumed")
else:
    state.add_clause(simplified)

# Backward simplification (when implemented)
deleted_ids = state.simplify_backward(new_clause_id)
print(f"New clause subsumed {len(deleted_ids)} existing clauses")

# Check subsumption manually
if state.is_subsumed_by(clause, clause_id):
    print(f"Clause is subsumed by clause {clause_id}")

# Check if clause is tautology
if state.is_tautology(clause):
    print("Clause is a tautology")
```

### State Inspection

```python
# Get statistics
stats = state.statistics()
print(f"Total clauses: {stats['total']}")
print(f"Processed: {stats['processed']}")
print(f"Unprocessed: {stats['unprocessed']}")
print(f"Deleted: {stats['deleted']}")
print(f"Empty clauses: {stats['empty_clauses']}")
print(f"Unit clauses: {stats['unit_clauses']}")

# Clause size distribution
size_dist = state.clause_size_distribution()
for size, count in size_dist.items():
    print(f"Clauses with {size} literals: {count}")

# Symbol frequency
symbol_freq = state.symbol_frequency()
for symbol, count in symbol_freq.items():
    print(f"Symbol '{symbol}': {count} occurrences")
```

### Proof Extraction

```python
# Get proof if found
if state.proof_found():
    proof = state.extract_proof()
    
    # Proof is a list of steps
    for step in proof:
        print(f"[{step.id}] {step.clause}")
        print(f"    From: {step.parents} by {step.rule}")
    
    # Get proof DAG
    proof_dag = state.proof_dag()
    
    # Get minimal proof (remove unnecessary steps)
    minimal_proof = state.minimal_proof()
```

### Serialization

```python
# Save state to file
state.save("proof_state.json")
state.save_binary("proof_state.bin")  # More efficient

# Load state from file
loaded_state = ProofState.load("proof_state.json")
loaded_state = ProofState.load_binary("proof_state.bin")

# Convert to/from dict
state_dict = state.to_dict()
restored_state = ProofState.from_dict(state_dict)
```

### Advanced Features

```python
# Set term ordering
from proofatlas import KBO
state.set_term_ordering(KBO(variable_weight=1))

# Configure literal selection
state.set_literal_selector("max_weight")

# Enable/disable simplification
state.enable_tautology_deletion = True
state.enable_forward_subsumption = True
state.enable_backward_subsumption = False  # Not yet implemented

# Memory limits
state.max_clauses = 1_000_000
state.max_clause_size = 100

# Time tracking
state.start_timer()
# ... perform operations ...
elapsed = state.elapsed_time()
print(f"Proof search took {elapsed:.3f} seconds")
```

## Example: Step-by-Step Proof Search

```python
from proofatlas import ProofState, parse_tptp_file
from proofatlas import generate_inferences

# Load problem
problem = parse_tptp_file("problem.p")
state = ProofState.from_problem(problem)

# Configure
state.set_literal_selector("max_weight")

# Main loop
steps = 0
while not state.is_saturated() and not state.proof_found():
    steps += 1
    
    # Select given clause
    given_id = state.select_given_clause(strategy="age_weight_ratio", ratio=(1, 5))
    given = state[given_id]
    
    print(f"\nStep {steps}: Given clause [{given_id}] {given}")
    
    # Generate inferences
    inferences = generate_inferences(state, given_id)
    print(f"Generated {len(inferences)} inferences")
    
    # Process each inference
    for inf in inferences:
        # Simplify forward
        simplified = state.simplify_forward(inf.clause)
        if simplified is not None:
            # Add to unprocessed
            new_id = state.add_clause(
                simplified,
                parents=inf.parents,
                rule=inf.rule_name
            )
            print(f"  [{new_id}] {simplified} from {inf.parents} by {inf.rule_name}")
    
    # Move given clause to processed
    state.process_clause(given_id)
    
    # Check limits
    if steps >= 1000:
        print("Step limit reached")
        break

# Results
if state.proof_found():
    print("\nProof found!")
    proof = state.extract_proof()
    for step in proof:
        print(f"[{step.id}] {step.clause} from {step.parents} by {step.rule}")
else:
    print(f"\nSaturated without proof after {steps} steps")
    
print(f"\nFinal statistics: {state.statistics()}")
```

## Implementation Notes

1. **Rust Backend**: ProofState is implemented in Rust for performance
2. **Python Wrapper**: Thin Python wrapper using PyO3 bindings
3. **Memory Safety**: Rust ensures no data races or memory issues
4. **Lazy Evaluation**: Some operations (like statistics) computed on demand
5. **Efficient Storage**: Clauses stored in contiguous arrays when possible