# ProofAtlas API Design and Component Architecture

## Overview

ProofAtlas is a modular theorem proving framework that separates concerns into distinct, composable components. Each component has a clear interface and can be extended or replaced independently.

## Core Components

### 1. Core (`proofatlas.core`)

**Purpose**: Core data structures and logic.

**Components**:
- `fol`: First-order logic in CNF
  - `Term`, `Literal`, `Clause`, `Problem`
  - Unification, substitution, matching
- `proof`:
  - `ProofState` encoding processed and unprocessed clauses.
  - `Proof` consisting of a list of ProofStates and all applied rules.

### 2. File Formats (`proofatlas.fileformats`)

**Purpose**: Handle parsing and conversion of different theorem proving file formats to a common internal representation.

**Key Interfaces**:
```python
class FileFormat:
    def parse_file(path: Path) -> Problem
    def parse_string(content: str) -> Problem
```

**Implementations**:
- `TPTPFormat`: TPTP file format (FOF, CNF)

### 3. Data Formats (`proofatlas.dataformats`)

**Purpose**: Convert proof states (processed/unprocessed clauses) into usable representations, e.g. for machine learning.

**Key Interfaces**:
```python
class DataFormat:
    def encode_state(state: ProofState) -> Any
```

**Implementations**:
- `GraphFormat`: Convert to graph representation (for GNNs)

### 4. Datasets (`proofatlas.problemsets`)

**Purpose**: Manage collections of theorem proving problems and their proofs with train/val/test splits.

**Key Interfaces**:
```python
class ProblemSet:
    def __init__(name: str, path: Optional[Path], format: Optional[FileFormat], splits: List[DatasetSplit]) -> ProblemSet

class ProofSet(Dataset):
    name: str
    def create(path: Path, loop: Loop, sel: Selector, steps: int) 
```

**Features**:
- YAML-based configuration

### 5. Loops (`proofatlas.loops`)

**Purpose**: Implement the given clause algorithm for saturation-based theorem proving.

**Key Interfaces**:
```python
class Loop:
    def step(proof: Proof, given_clause: int) -> Proof
    """Execute one step of the saturation loop"""
```

**Implementations**:
- `BasicLoop`: Complete implementation with:
  - Resolution and factoring inference rules
  - Forward simplification (tautology deletion, subsumption checking)
  - Redundancy filtering (duplicate removal)
  - Clause size limits
  - Full proof tracking with rule applications
  - Note: Backward simplification is TODO

### 6. Selectors (`proofatlas.selectors`)

**Purpose**: Implement clause selection strategies for choosing which clause to process next.

**Key Interfaces**:
```python
class Selector:
    def select(ProofState) -> Optional[int]
    def score_clauses(ProofState) -> List[float]
    def run(ProofState, steps: Optional[int]) -> Proof
    def train(ProofSet, **kwargs) -> ()
```

**Implementations**:
- `RandomSelector`: Random clause selection.
- `GNNSelector`: GNN for clause selection.


## API Usage Examples

### 1. Basic Problem Solving
```python
from proofatlas.fileformats.tptp_parser import TPTPParser
from proofatlas.loops.basic import BasicLoop
from proofatlas.selectors import RandomSelector
from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState

# Load problem
parser = TPTPParser()
problem = parser.parse_file('problem.p')

# Create initial proof state
initial_state = ProofState(processed=[], unprocessed=problem.clauses)
proof = Proof(initial_state)

# Create loop and selector
loop = BasicLoop(max_clause_size=50, forward_simplify=True)
selector = RandomSelector()

# Run proof search
while proof.final_state.unprocessed and not proof.is_complete:
    # Select clause
    idx = selector.select(proof.final_state)
    if idx is None:
        break
    
    # Take step
    proof = loop.step(proof, idx)
```

### 2. ProblemSet Creation
```python
from proofatlas.datasets import ProblemSet
from proofatlas.datasets.config import DatasetSplit

# Create dataset
problemset = ProblemSet(
    name='my_dataset',
    path=Path(TPTP_PATH + '/Problems'),
    format='tptp',
    format_args={"include" : "TPTP_PATH + \'/Axioms\'"}
    splits=[
        DatasetSplit('train', patterns=['*.p'], ratio=0.8),
        DatasetSplit('val', patterns=['*.p'], ratio=0.2)
    ]
)
```

### 3. ProofSet Creation

```python
from proofatlas.datasets import ProofSet

proofset = ProofSet(
  'my_dataset',
  loop=loop,
  selector=selector,
  steps=10
)

```


### 3. Model Training
```python
from proofatlas.selectors import GNNSelector
from torch.optim import Adam

# Create selector
selector = GNNSelector(
  format='graph',
  hidden_dim=256,
  num_layers=4,
  optimizer=Adam()
)


selector.train(ProofSet)
```

## Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Extensibility**: New implementations can be added by implementing base interfaces
3. **Composability**: Components can be mixed and matched freely
4. **Configuration**: YAML-based configuration for datasets and experiments
5. **Type Safety**: Use Python type hints throughout
6. **Testing**: Each component should have comprehensive unit tests