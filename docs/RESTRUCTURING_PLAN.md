# ProofAtlas Restructuring Plan

## Key Changes from API Design

1. **Rename**: `datasets` → `problemsets`
2. **Rename**: `environments` → `loops`
3. **Move**: `ProofState` from `dataformats.base` → `core.proof`
4. **Add**: `Proof` class to `core.proof`
5. **Add**: `Rule` interface to `loops`
6. **Simplify**: Remove redundant components (training, inference directories)
7. **Change**: Selectors now have `run()` and `train()` methods

## Proposed Directory Structure

```
src/proofatlas/
├── __init__.py
├── core/                    # Core data structures
│   ├── __init__.py
│   ├── fol/                # First-order logic in CNF
│   │   ├── __init__.py
│   │   ├── logic.py       # Term, Literal, Clause, Problem
│   │   └── unification.py # Unification and substitution
│   └── proof/             # Proof representation
│       ├── __init__.py
│       ├── state.py       # ProofState class
│       └── proof.py       # Proof class
├── fileformats/           # File format handlers
│   ├── __init__.py
│   ├── base.py
│   └── tptp.py           # TPTPFormat implementation
├── dataformats/           # Data representation formats
│   ├── __init__.py
│   ├── base.py           # DataFormat interface
│   └── graph.py          # GraphFormat for GNNs
├── data/           # Problem and proof set management
│   ├── __init__.py
│   ├── problemset.py     # ProblemSet class
│   ├── proofset.py       # ProofSet class
│   ├── config.py         # Configuration classes
│   └── splits.py         # DatasetSplit logic
├── loops/                 # Given clause loops
│   ├── __init__.py
│   ├── base.py           # Loop and Rule interfaces
│   ├── basic.py          # BasicLoop implementation
│   └── rules/            # Individual inference rules
│       ├── __init__.py
│       ├── resolution.py # Binary resolution rule
│       ├── factoring.py  # Factoring rule
│       └── subsumption.py # Subsumption rule
├── selectors/            # Clause selectors
│   ├── __init__.py
│   ├── base.py          # Selector interface with run() and train()
│   ├── random.py        # RandomSelector
│   └── gnn.py           # GNNSelector
├── parsers/             # Low-level parsers
│   ├── __init__.py
│   └── tptp/
│       ├── __init__.py
│       ├── lexer.py
│       ├── parser.py
│       └── tptp_fof_cnf.lark
└── utils/               # Utilities
    ├── __init__.py
    └── config.py
```

## Migration Steps

### Phase 1: Core Restructuring

1. **Create `core/proof/`**:
   - Move `ProofState` from `dataformats/base.py` → `core/proof/state.py`
   - Create `core/proof/proof.py` with `Proof` class
   - Update all imports

2. **Rename `datasets/` → `problemsets/`**:
   - Rename directory
   - Rename `dataset.py` → `problemset.py`
   - Create `proofset.py`
   - Update class names and imports

3. **Rename `environments/` → `loops/`**:
   - Rename directory
   - Rename classes: `ProvingEnvironment` → `Loop`
   - Extract rules into `loops/rules/`
   - Update imports

### Phase 2: Simplify Components

4. **Update Selectors**:
   - Add `run()` method to base selector
   - Add `train()` method to base selector
   - Rename existing selectors to match API design
   - Move neural selector logic to `gnn.py`

5. **Clean up File/Data Formats**:
   - Simplify FileFormat interface (remove `to_cnf`, `write_cnf`)
   - Update DataFormat to use `ProofState` from core

6. **Remove Unnecessary Components**:
   - Delete empty `training/` structure
   - Delete empty `inference/` structure
   - Move any useful code to appropriate locations

### Phase 3: Implementation Details

7. **Implement Missing Classes**:
   - `core/proof/proof.py`: Proof class with state history
   - `loops/base.py`: Rule interface
   - `loops/rules/*.py`: Individual rule implementations
   - `problemsets/proofset.py`: ProofSet for training data

8. **Update Existing Code**:
   - Update `BasicLoop` (was `GivenClauseEnvironment`) to use Rule interface
   - Update selectors to include `run()` and `train()` methods
   - Fix all imports throughout codebase

## Files to Update/Create

### New Files:
1. `core/proof/state.py` - Move ProofState here
2. `core/proof/proof.py` - New Proof class
3. `loops/base.py` - Loop and Rule interfaces
4. `loops/rules/resolution.py` - Resolution rule
5. `loops/rules/factoring.py` - Factoring rule
6. `problemsets/proofset.py` - ProofSet class
7. `selectors/random.py` - RandomSelector
8. `selectors/gnn.py` - GNNSelector

### Files to Rename:
1. `datasets/` → `problemsets/`
2. `environments/` → `loops/`
3. `environments/given_clause.py` → `loops/basic.py`
4. `selectors/fifo.py` → `selectors/random.py`
5. `selectors/neural.py` → `selectors/gnn.py`

### Files to Update:
1. All import statements
2. Class names to match new API
3. Method signatures to match interfaces

## Benefits of This Structure

1. **Cleaner separation**: Core logic (proof, fol) separate from algorithms (loops, selectors)
2. **Simpler API**: Fewer components, clearer responsibilities
3. **Better naming**: Names match the domain (loops, problemsets)
4. **Extensible**: Easy to add new rules, loops, or selectors
5. **Self-contained**: Each component has everything it needs