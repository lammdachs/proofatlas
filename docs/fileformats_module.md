# FileFormats Module Documentation

The `fileformats` module handles parsing and writing various theorem proving file formats.

## Overview

The fileformats module provides a flexible system for handling different file formats used in theorem proving, with TPTP being the primary supported format.

## Module Structure

### base.py

Defines the abstract base class for all file format handlers.

```python
class FileFormat(ABC):
    @abstractmethod
    def parse_file(self, file_path: Path, **kwargs) -> Problem
    
    @abstractmethod
    def parse_string(self, content: str, **kwargs) -> Problem
    
    @abstractmethod
    def write_file(self, problem: Problem, file_path: Path, **kwargs) -> None
    
    @abstractmethod
    def format_problem(self, problem: Problem, **kwargs) -> str
    
    @property
    @abstractmethod
    def name(self) -> str
    
    @property
    @abstractmethod
    def extensions(self) -> list[str]
```

### registry.py

Provides a registry system for managing file format handlers.

```python
# Get a handler by format name
handler = get_format_handler(format_name='tptp')

# Get a handler by file extension
handler = get_format_handler(file_path=Path('problem.p'))

# Parse a file
problem = handler.parse_file(Path('problem.p'))
```

### tptp.py

TPTP (Thousands of Problems for Theorem Provers) format handler.

**Features:**
- Parses CNF and FOF formulas
- Converts FOF to CNF automatically
- Handles includes with path resolution
- Supports conjectures (automatically negated)
- Writes problems back to TPTP format

**Supported Extensions:** `.p`, `.tptp`, `.ax`

### tptp_parser/

Contains the Lark-based parser for TPTP syntax:
- `lexer.py`: Grammar definition for TPTP
- `parser.py`: Transformation logic from parse trees to logic objects

## Usage Examples

### Reading TPTP Files

```python
from proofatlas.fileformats import get_format_handler

# Get TPTP handler
tptp = get_format_handler('tptp')

# Parse a file
problem = tptp.parse_file(Path('examples/simple.p'))

# Parse a string
problem = tptp.parse_string("""
cnf(axiom1, axiom, p(a)).
cnf(axiom2, axiom, ~p(X) | q(X)).
cnf(goal, negated_conjecture, ~q(a)).
""")
```

### Writing TPTP Files

```python
from proofatlas.core.logic import *
from proofatlas.fileformats import get_format_handler

# Create a problem
a = Constant("a")
P = Predicate("p", 1)
Q = Predicate("q", 1)

problem = Problem(
    Clause(Literal(P(a), True)),
    Clause(Literal(P(Variable("X")), False), Literal(Q(Variable("X")), True)),
    Clause(Literal(Q(a), False))
)

# Write to file
tptp = get_format_handler('tptp')
tptp.write_file(problem, Path('output.p'))
```

### Extending with New Formats

To add support for a new file format:

1. Create a new class inheriting from `FileFormat`
2. Implement all abstract methods
3. Register it in the registry

```python
from proofatlas.fileformats import FileFormat, FileFormatRegistry

class MyFormat(FileFormat):
    def parse_file(self, file_path, **kwargs):
        # Implementation
        pass
    
    def parse_string(self, content, **kwargs):
        # Implementation
        pass
    
    def write_file(self, problem, file_path, **kwargs):
        # Implementation
        pass
    
    def format_problem(self, problem, **kwargs):
        # Implementation
        pass
    
    @property
    def name(self):
        return 'myformat'
    
    @property
    def extensions(self):
        return ['.myf', '.myfmt']

# Register the format
registry = FileFormatRegistry()
registry.register('myformat', MyFormat)
```

## Design Principles

1. **Separation of Concerns**: Each format handler is independent
2. **Extensibility**: Easy to add new formats without modifying existing code
3. **Consistency**: All formats convert to/from the same `Problem` representation
4. **Flexibility**: Handlers can accept format-specific options via `**kwargs`

## Integration

The fileformats module integrates with:
- **Core**: Uses Problem, Clause, Literal, and Term objects
- **Data**: Problem loading for datasets
- **Scripts**: Problem file conversion utilities