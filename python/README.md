# ProofAtlas Python Interface

ProofAtlas provides high-performance automated theorem proving for first-order logic through Python bindings to its Rust implementation.

## Features

- **High Performance**: Core algorithms implemented in Rust for speed
- **Easy to Use**: Pythonic API for theorem proving
- **Complete**: Implements superposition calculus with equality
- **Flexible**: Multiple clause and literal selection strategies
- **Inspectable**: Step-by-step proof exploration

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/proofatlas.git
cd proofatlas/python

# Build the Rust extension
./build.sh

# Install in development mode
pip install -e .
```

### Prerequisites

- Python 3.7 or later
- Rust toolchain (install from https://rustup.rs/)
- C compiler (for Python extensions)

### Using pip (coming soon)

```bash
pip install proofatlas
```

## Quick Start

```python
from proofatlas import ProofState, saturate_step

# Create a proof state
state = ProofState()

# Add a simple problem
state.add_clauses_from_tptp("""
cnf(axiom1, axiom, p(a)).
cnf(axiom2, axiom, ~p(X) | q(X)).
cnf(goal, negated_conjecture, ~q(a)).
""")

# Run automated proof search
while True:
    result = saturate_step(state)
    if result['proof_found']:
        print("Theorem proven!")
        break
    if result['saturated']:
        print("Cannot prove theorem")
        break
```

## Examples

### Basic Propositional Logic

```python
from proofatlas import ProofState

state = ProofState()

# Modus ponens: From P and P→Q, derive Q
state.add_clauses_from_tptp("""
cnf(premise1, axiom, p).
cnf(premise2, axiom, ~p | q).
cnf(goal, negated_conjecture, ~q).
""")

# Find proof
while state.num_unprocessed() > 0:
    given_id = state.select_given_clause()
    if given_id is None:
        break
    
    for inf in state.generate_inferences(given_id):
        state.add_inference(inf)
    
    state.process_clause(given_id)
    
    if state.contains_empty_clause():
        print("Proof found!")
        break
```

### Equality Reasoning

```python
# Enable superposition for equality
state = ProofState()
state.set_use_superposition(True)

# Prove transitivity of equality
state.add_clauses_from_tptp("""
cnf(eq1, axiom, a = b).
cnf(eq2, axiom, b = c).
cnf(goal, negated_conjecture, a != c).
""")

# The prover will automatically handle equality
```

### Interactive Proof Exploration

```python
# Step through proof manually
state = ProofState()
state.add_clauses_from_tptp("...")

while state.num_unprocessed() > 0:
    # Select next clause
    given_id = state.select_given_clause()
    
    # Inspect it
    info = state.get_clause_info(given_id)
    print(f"Processing: {info.clause_string}")
    print(f"Weight: {info.weight}, Variables: {info.variables}")
    
    # Generate and examine inferences
    for inf in state.generate_inferences(given_id):
        print(f"  Can infer: {inf.clause_string} by {inf.rule_name}")
        state.add_inference(inf)
    
    state.process_clause(given_id)
```

## API Reference

### Core Classes

#### ProofState

The main class for managing proof search.

**Methods:**
- `add_clauses_from_tptp(content: str) -> List[int]` - Parse and add TPTP clauses
- `num_clauses() -> int` - Total number of clauses
- `num_processed() -> int` - Number of processed clauses
- `num_unprocessed() -> int` - Number of unprocessed clauses
- `contains_empty_clause() -> bool` - Check if proof found
- `select_given_clause(strategy="age") -> Optional[int]` - Select next clause
- `generate_inferences(clause_id: int) -> List[InferenceResult]` - Generate inferences
- `add_inference(inference: InferenceResult) -> Optional[int]` - Add new clause
- `process_clause(clause_id: int)` - Mark clause as processed
- `get_clause_info(clause_id: int) -> ClauseInfo` - Get clause details
- `get_statistics() -> Dict[str, int]` - Get search statistics
- `get_proof_trace() -> List[ProofStep]` - Get proof derivation

**Configuration:**
- `set_literal_selection(strategy: str)` - Set literal selection ("all" or "max_weight")
- `set_use_superposition(enabled: bool)` - Enable/disable equality reasoning

#### ClauseInfo

Information about a clause.

**Attributes:**
- `clause_id: int` - Unique clause identifier
- `clause_string: str` - Formatted clause
- `num_literals: int` - Number of literals
- `literal_strings: List[str]` - Individual literals as strings
- `is_unit: bool` - True if unit clause
- `is_horn: bool` - True if Horn clause
- `is_equality: bool` - True if contains equality
- `weight: int` - Total symbol count
- `variables: List[str]` - Variable names

### Helper Functions

#### saturate_step

Perform one step of proof search.

```python
result = saturate_step(state, clause_selection="age")
# Returns dict with:
#   given_id: Selected clause ID
#   new_clauses: List of new clause IDs  
#   saturated: True if no more to process
#   proof_found: True if proof found
```

## Advanced Usage

### Custom Clause Selection

```python
# Available strategies
state.select_given_clause(strategy="age")      # Age-based (FIFO)
state.select_given_clause(strategy="smallest") # Smallest clause first
```

### Proof Analysis

```python
# Get detailed proof trace
if state.contains_empty_clause():
    for step in state.get_proof_trace():
        print(f"[{step.clause_id}] {step.clause_string}")
        if step.parent_ids:
            print(f"  Derived from {step.parent_ids} by {step.rule_name}")
```

### Statistics Monitoring

```python
# Monitor proof search progress
stats = state.get_statistics()
print(f"Total clauses: {stats['total']}")
print(f"Processed: {stats['processed']}")
print(f"In queue: {stats['unprocessed']}")
print(f"Unit clauses: {stats['unit_clauses']}")
```

## Examples Directory

See the `examples/` directory for complete examples:

- `basic_usage.py` - Simple theorem proving examples
- `interactive_demo.py` - Step-by-step proof exploration
- `group_theory.py` - Advanced equality reasoning in algebra

## Performance Tips

1. **Literal Selection**: Use "max_weight" for problems with many literals
2. **Clause Selection**: Use "smallest" for problems generating many clauses
3. **Superposition**: Only enable for problems with equality
4. **Early Termination**: Check `contains_empty_clause()` frequently

## Limitations

- Input must be in CNF (Conjunctive Normal Form)
- No built-in CNF conversion (use TPTP FOF format with external tools)
- Limited to first-order logic without arithmetic
- No parallel proof search (single-threaded)

## Contributing

Contributions are welcome! See the main ProofAtlas repository for guidelines.

## License

See LICENSE file in the main repository.