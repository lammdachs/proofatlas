# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup and Commands

### Initial Setup
```bash
# Create and activate conda environment
./setup.sh  # Creates 'proofatlas' conda environment with all dependencies

# Manual activation
conda activate proofatlas

# Install in development mode
pip install -e .
```

### Running Tests
```bash
# Run all tests from src directory (required for imports to work)
cd src && python -m pytest ../tests/ -v

# Run specific test category
cd src && python -m pytest ../tests/core/ -v
cd src && python -m pytest ../tests/rules/ -v

# Run with coverage
cd src && python -m pytest ../tests/ --cov=proofatlas
```

### Test Structure
Tests are organized to mirror the source structure:
```
tests/
├── core/           # Tests for core.logic, core.state
├── proofs/         # Tests for proofs module
├── rules/          # Tests for rules module
├── loops/          # Tests for saturation loops
├── data/           # Tests for data module
├── fileformats/    # Tests for file format parsers
├── navigator/      # Tests for proof navigator
├── test_data/      # Test data and generated proofs for inspection
└── test_serialized_data.py  # Integration tests
```

### Development Commands
```bash
# Type checking (if mypy is configured)
mypy src/proofatlas

# Linting
ruff check src/proofatlas
black src/proofatlas --check
```

## Codebase Architecture

### Module Dependency Order
The codebase follows a strict dependency hierarchy. When modifying code, respect this order:
1. `core.logic` (standalone - pure FOL)
2. `proofs.state` depends on `core.logic`
3. `rules` depends on `core`, `proofs.state`
4. `proofs.proof` depends on `core`, `rules`, `proofs.state`
5. `fileformats` depends on `core`
6. `dataformats` depends on `core`
7. `loops` depends on `core`, `rules`, `proofs`
8. `selectors` depends on `core`, `dataformats`
9. `data` depends on all above

### Core Module (`src/proofatlas/core/`)
The foundation of the system, implementing pure first-order logic:
- **logic.py**: Immutable FOL objects (Variable, Constant, Function, Predicate, Term, Literal, Clause, Problem)
  - Variables and Constants don't take arity parameter in constructor
  - Use `Problem(*clauses)` not `Problem([clauses])`
  - Clause constructor takes `*literals` (unpacked), not a list

### Rules Module (`src/proofatlas/rules/`)
Modular inference rules that operate on proof states:
- **base.py**: Abstract `Rule` class and `RuleApplication` dataclass
- **resolution.py**: Binary resolution rule
- **factoring.py**: Factoring rule
- **subsumption.py**: Subsumption elimination rule

Rules return `RuleApplication` objects containing:
- `rule_name`: Name of the rule
- `parents`: Indices of parent clauses
- `generated_clauses`: New clauses produced
- `deleted_clause_indices`: Clauses to remove
- `metadata`: Additional information

### Proofs Module (`src/proofatlas/proofs/`)
- **state.py**: `ProofState` tracks processed/unprocessed clauses during proof search
- **proof.py**: `Proof` stores sequence of `ProofStep` objects
  - ProofStep contains: state, selected_clause, applied_rules, metadata
  - Proof tracks both steps list and final_state
  - Applied rules are stored as RuleApplication objects
- **serialization.py**: JSON serialization for proof objects including ProofState

### Current Structure (Post-Refactoring)
```
src/proofatlas/
├── core/           # FOL logic and proof state
├── rules/          # Modular inference rules (resolution, factoring, etc.)
├── proofs/         # Proof tracking and management
├── fileformats/    # File format parsers (TPTP)
├── dataformats/    # ML data representation
├── data/           # Problem and proof set management
├── loops/          # Given clause algorithm implementations
├── selectors/      # Clause selection strategies (base, random, gnn)
├── navigator/      # Terminal-based proof visualization
└── utils/

scripts/
├── print_proof.py  # Non-interactive proof printer (CLI starts at step 0)
└── inspect_proof.py # Interactive proof navigator

docs/
└── saturation_loop_design.md  # Design and implementation details for BasicLoop
```

### Navigator Module (`src/proofatlas/navigator/`)
Interactive terminal-based proof visualization:
- **proof_navigator.py**: Navigate through proof steps with keyboard controls
  - Uses box-drawing characters for clean terminal UI
  - Two-column layout showing PROCESSED and UNPROCESSED clauses
  - Highlights the given clause with an arrow (→)
  - Shows rule applications and metadata
  - Keyboard controls: n/p (next/prev), q (quit), h (help)

### Loops Module (`src/proofatlas/loops/`)
Implements the given clause algorithm:
- **base.py**: Abstract `Loop` class with helper methods (tautology/subsumption checking)
- **basic.py**: `BasicLoop` - complete implementation with:
  - Resolution and factoring inference rules
  - Forward simplification (tautology deletion, subsumption checking)
  - Redundancy filtering (duplicate removal)
  - Clause size limits
  - Full proof tracking with rule applications
  - **Note**: Backward simplification is TODO

### Key Implementation Notes

1. **Import Structure**: Tests run from `src/` directory, so imports use `proofatlas.module.submodule`

2. **Rule Architecture**:
   - All rules inherit from `Rule` abstract base class
   - Rules implement `apply(state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]`
   - Rules don't modify state - they return RuleApplication objects
   - Loops are responsible for applying RuleApplications to create new states
   - **Important**: Import ProofState from `proofatlas.proofs.state`, not from `proofatlas.proofs` to avoid circular imports

3. **Loop Architecture**:
   - BasicLoop records the state BEFORE processing each given clause
   - Rule applications are filtered to only show those producing non-redundant clauses
   - Parent indices in rule applications show only the processed clause (given clause is implicit)
   - Steps are 0-indexed in both JSON and CLI display

4. **Selector Architecture**: 
   - Base class in `selectors/base.py`
   - GNN selector includes the model implementation directly
   - Selectors have `select()`, `run()`, and `train()` methods

5. **Parser Integration**: TPTP parser implemented in `fileformats/tptp.py`

6. **No Models Directory**: GNN and other models are integrated into their usage points (e.g., GNN in selectors/gnn.py)

7. **Testing Pattern**: Tests mirror source structure under `tests/` directory

8. **Proof Inspection**: Use scripts/print_proof.py or scripts/inspect_proof.py to examine generated proofs

### Working with the Refactored Codebase

When implementing new features:
1. Check if it belongs in existing modules before creating new ones
2. Follow the established patterns (e.g., metadata dict for extensibility)
3. Maintain the dependency hierarchy
4. Add tests for any new functionality in the appropriate test file

The refactoring prioritized:
- Flatter directory structure
- Clear module boundaries  
- Flexible metadata-based extensibility
- Testability and maintainability