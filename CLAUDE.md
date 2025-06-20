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

# Run specific test module
cd src && python -m pytest ../tests/test_logic.py -v

# Run with coverage
cd src && python -m pytest ../tests/ --cov=proofatlas
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
1. `core.logic` → `core.state` → `core.proof`
2. `fileformats` depends on `core`
3. `dataformats` depends on `core`
4. `loops` depends on `core`
5. `selectors` depends on `core`, `dataformats`
6. `data` depends on all above

### Core Module (`src/proofatlas/core/`)
The foundation of the system, implementing first-order logic:
- **logic.py**: Immutable FOL objects (Variable, Constant, Function, Predicate, Term, Literal, Clause, Problem)
  - Variables and Constants don't take arity parameter in constructor
  - Use `Problem(*clauses)` not `Problem([clauses])`
- **state.py**: `ProofState` tracks processed/unprocessed clauses during proof search
- **proof.py**: `Proof` stores sequence of `ProofStep` objects with states and selected clauses
  - ProofStep only stores state, selected_clause, and metadata dict
  - Proof has both steps list and final_state

### Current Structure (Post-Refactoring)
```
src/proofatlas/
├── core/           # Flattened - no subdirectories
├── fileformats/    # Contains tptp_parser/ and vampire_parser/ subdirs
├── dataformats/    # ML data representation
├── data/           # Problem and proof set management
├── loops/          # Given clause algorithm with rules/ subdir
├── selectors/      # Clause selection strategies (base, random, gnn)
└── utils/
```

### Key Implementation Notes

1. **Import Structure**: Tests run from `src/` directory, so imports use `proofatlas.module.submodule`

2. **Selector Architecture**: 
   - Base class in `selectors/base.py`
   - GNN selector includes the model implementation directly
   - Selectors have `select()`, `run()`, and `train()` methods

3. **Parser Integration**: Parsers moved to `fileformats/tptp_parser/` and `fileformats/vampire_parser/`

4. **No Models Directory**: GNN and other models are integrated into their usage points (e.g., GNN in selectors/gnn.py)

5. **Testing Pattern**: Each core module has corresponding test file with comprehensive unit tests

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