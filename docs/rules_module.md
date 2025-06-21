# Rules Module Documentation

The `rules` module provides a modular system for inference rules in theorem proving.

## Overview

The rules module implements a clean separation between inference rules and proof state management. Rules operate on `ProofState` objects and return `RuleApplication` objects describing what changes should be made.

## Architecture

### Base Classes

#### Rule (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from proofatlas.core.state import ProofState

class Rule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the rule."""
        pass
    
    @abstractmethod
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        """
        Apply the rule to the given clauses.
        
        Args:
            state: Current proof state
            clause_indices: Indices of clauses to use (in processed list)
            
        Returns:
            RuleApplication if successful, None otherwise
        """
        pass
```

#### RuleApplication

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any
from proofatlas.core.logic import Clause

@dataclass
class RuleApplication:
    """Result of applying an inference rule."""
    rule_name: str
    parents: List[int]  # Indices of parent clauses
    generated_clauses: List[Clause] = field(default_factory=list)
    deleted_clause_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Implemented Rules

### ResolutionRule

Binary resolution combines two clauses by resolving on complementary literals.

```python
from proofatlas.rules import ResolutionRule

rule = ResolutionRule()
# Resolve clauses at indices 0 and 1 in processed list
result = rule.apply(state, [0, 1])

if result:
    for clause in result.generated_clauses:
        print(f"Generated: {clause}")
```

**Example:**
- Clause 1: `P(a) | Q(b)`
- Clause 2: `~P(a) | R(c)`
- Resolvent: `Q(b) | R(c)`

### FactoringRule

Factoring unifies duplicate literals within a single clause.

```python
from proofatlas.rules import FactoringRule

rule = FactoringRule()
# Factor clause at index 0
result = rule.apply(state, [0])
```

**Example:**
- Clause: `P(X) | P(a) | Q(X)`
- Factor: `P(a) | Q(a)` (with X unified to a)

### SubsumptionRule

Subsumption removes clauses that are logically implied by others.

```python
from proofatlas.rules import SubsumptionRule

rule = SubsumptionRule()
# Check all clauses for subsumption
result = rule.apply(state, [])  # Indices ignored for this rule

if result and result.deleted_clause_indices:
    print(f"Removed {len(result.deleted_clause_indices)} subsumed clauses")
```

**Example:**
- Clause 1: `P(a)`
- Clause 2: `P(a) | Q(b)`
- Result: Clause 2 is subsumed by Clause 1 and removed

## Usage in Loops

The rules are designed to be used by loop implementations:

```python
from proofatlas.loops.basic import BasicLoop
from proofatlas.rules import ResolutionRule, FactoringRule

class BasicLoop:
    def _generate_clauses_with_rules(self, given_clause, processed):
        """Generate new clauses and track rule applications."""
        new_clauses = []
        rule_applications = []
        
        # Create temporary state for rule application
        temp_state = ProofState(processed=processed + [given_clause], unprocessed=[])
        given_idx = len(processed)
        
        # Apply resolution with each processed clause
        resolution = ResolutionRule()
        for i, proc_clause in enumerate(processed):
            result = resolution.apply(temp_state, [i, given_idx])
            if result:
                new_clauses.extend(result.generated_clauses)
                rule_applications.append(result)
        
        # Apply factoring to given clause
        factoring = FactoringRule()
        result = factoring.apply(temp_state, [given_idx])
        if result:
            new_clauses.extend(result.generated_clauses)
            rule_applications.append(result)
            
        return new_clauses, rule_applications
```

## Extending with New Rules

To add a new inference rule:

1. Create a new class inheriting from `Rule`
2. Implement the `name` property
3. Implement the `apply` method
4. Return appropriate `RuleApplication` objects

Example:
```python
from proofatlas.rules.base import Rule, RuleApplication

class ParamodulationRule(Rule):
    @property
    def name(self) -> str:
        return "paramodulation"
    
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        if len(clause_indices) != 2:
            return None
            
        # Implementation of paramodulation...
        # ...
        
        return RuleApplication(
            rule_name=self.name,
            parents=clause_indices,
            generated_clauses=new_clauses
        )
```

## Design Principles

1. **Immutability**: Rules don't modify the input state
2. **Modularity**: Each rule is independent and self-contained
3. **Flexibility**: RuleApplication allows arbitrary metadata
4. **Testability**: Rules can be tested in isolation
5. **Extensibility**: New rules can be added without modifying existing code

## Integration with Proofs

Rule applications are stored in `ProofStep` objects:

```python
from proofatlas.proofs import Proof, ProofStep

# In a loop implementation
proof.add_step(
    new_state,
    selected_clause=given_clause_idx,
    applied_rules=rule_applications,  # List of RuleApplication objects
    **other_metadata
)
```

This allows complete reconstruction of how each clause was derived.