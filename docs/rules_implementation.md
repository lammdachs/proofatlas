# Rules System Implementation Guide

This document provides implementation details for the ProofAtlas rules system in both Python and Rust.

## Architecture Overview

### Rule Interface

Every inference rule implements the following interface:

```python
class Rule(ABC):
    @abstractmethod
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        """Apply the rule to the given clauses."""
        pass
```

```rust
trait Rule {
    fn apply(&self, state: &ProofState, clause_indices: &[usize]) -> Option<RuleApplication>;
}
```

### RuleApplication Structure

The result of applying a rule:

```python
@dataclass
class RuleApplication:
    rule_name: str                    # Name of the applied rule
    parents: List[int]               # Indices of parent clauses
    generated_clauses: List[Clause]  # New clauses produced
    deleted_clause_indices: List[int] # Clauses to remove (for simplification)
    metadata: Dict[str, Any]         # Additional information (unifier, positions, etc.)
```

## Core Rules Implementation

### 1. Resolution

**File**: `rules/resolution.py` (Python), `rules/resolution.rs` (Rust)

```python
class ResolutionRule(Rule):
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        if len(clause_indices) != 2:
            return None
        
        c1 = state.all_clauses[clause_indices[0]]
        c2 = state.all_clauses[clause_indices[1]]
        
        # Try all pairs of complementary literals
        for i, lit1 in enumerate(c1.literals):
            for j, lit2 in enumerate(c2.literals):
                if lit1.predicate.symbol == lit2.predicate.symbol and \
                   lit1.polarity != lit2.polarity:
                    
                    # Attempt unification
                    unifier = unify(lit1.predicate, lit2.predicate)
                    if unifier:
                        # Build resolvent
                        new_lits = []
                        # Add literals from c1 except lit1
                        for k, lit in enumerate(c1.literals):
                            if k != i:
                                new_lits.append(apply_substitution(lit, unifier))
                        # Add literals from c2 except lit2
                        for k, lit in enumerate(c2.literals):
                            if k != j:
                                new_lits.append(apply_substitution(lit, unifier))
                        
                        resolvent = Clause(*new_lits)
                        
                        return RuleApplication(
                            rule_name="resolution",
                            parents=clause_indices,
                            generated_clauses=[resolvent],
                            metadata={
                                "positions": [i, j],
                                "unifier": unifier
                            }
                        )
        
        return None
```

### 2. Factoring

**File**: `rules/factoring.py` (Python), `rules/factoring.rs` (Rust)

```python
class FactoringRule(Rule):
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        if len(clause_indices) != 1:
            return None
        
        clause = state.all_clauses[clause_indices[0]]
        factors = []
        
        # Try all pairs of literals
        for i in range(len(clause.literals)):
            for j in range(i + 1, len(clause.literals)):
                lit1 = clause.literals[i]
                lit2 = clause.literals[j]
                
                # Same polarity required
                if lit1.polarity == lit2.polarity:
                    unifier = unify(lit1.predicate, lit2.predicate)
                    if unifier:
                        # Build factor
                        new_lits = []
                        for k, lit in enumerate(clause.literals):
                            if k != j:  # Skip duplicate
                                new_lits.append(apply_substitution(lit, unifier))
                        
                        factor = Clause(*new_lits)
                        factors.append(factor)
        
        if factors:
            return RuleApplication(
                rule_name="factoring",
                parents=clause_indices,
                generated_clauses=factors
            )
        
        return None
```

### 3. Superposition

**File**: `rules/superposition.py` (Python), `rules/superposition.rs` (Rust)

```python
class SuperpositionRule(Rule):
    def __init__(self, ordering: TermOrdering):
        self.ordering = ordering
    
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        if len(clause_indices) != 2:
            return None
        
        c1 = state.all_clauses[clause_indices[0]]
        c2 = state.all_clauses[clause_indices[1]]
        results = []
        
        # Find positive equality in c1
        for i, lit1 in enumerate(c1.literals):
            if lit1.polarity and is_equality(lit1):
                left, right = get_equality_sides(lit1)
                
                # Check ordering constraint
                if not self.ordering.greater(left, right):
                    continue
                
                # Find positions to paramodulate into in c2
                for j, lit2 in enumerate(c2.literals):
                    positions = find_subterm_positions(lit2.predicate, left)
                    
                    for pos in positions:
                        # Check eligible position
                        if not is_eligible_position(lit2, pos, self.ordering):
                            continue
                        
                        # Unify
                        subterm = get_subterm_at(lit2.predicate, pos)
                        unifier = unify(left, subterm)
                        
                        if unifier:
                            # Build new literal
                            new_term = replace_subterm_at(
                                lit2.predicate, pos, 
                                apply_substitution(right, unifier)
                            )
                            new_lit = Literal(new_term, lit2.polarity)
                            
                            # Build new clause
                            new_lits = []
                            # Add from c1 except equality
                            for k, lit in enumerate(c1.literals):
                                if k != i:
                                    new_lits.append(apply_substitution(lit, unifier))
                            # Add from c2 with replacement
                            for k, lit in enumerate(c2.literals):
                                if k == j:
                                    new_lits.append(new_lit)
                                else:
                                    new_lits.append(apply_substitution(lit, unifier))
                            
                            results.append(Clause(*new_lits))
        
        if results:
            return RuleApplication(
                rule_name="superposition",
                parents=clause_indices,
                generated_clauses=results
            )
        
        return None
```

### 4. Forward Subsumption

**File**: `rules/subsumption.py` (Python), `rules/subsumption.rs` (Rust)

```python
class ForwardSubsumptionRule(Rule):
    def apply(self, state: ProofState, clause_indices: List[int]) -> Optional[RuleApplication]:
        if len(clause_indices) != 1:
            return None
        
        new_clause_idx = clause_indices[0]
        new_clause = state.all_clauses[new_clause_idx]
        
        # Check if any existing clause subsumes the new one
        for i, existing in enumerate(state.all_clauses):
            if i != new_clause_idx and subsumes(existing, new_clause):
                # New clause is redundant
                return RuleApplication(
                    rule_name="forward_subsumption",
                    parents=[i, new_clause_idx],
                    generated_clauses=[],  # No new clauses
                    deleted_clause_indices=[new_clause_idx],
                    metadata={"subsuming_clause": i}
                )
        
        return None

def subsumes(c1: Clause, c2: Clause) -> bool:
    """Check if c1 subsumes c2."""
    if len(c1.literals) > len(c2.literals):
        return False
    
    # Try to find a substitution
    return find_subsumption_substitution(c1.literals, c2.literals) is not None
```

## Literal Selection Implementation

```python
class LiteralSelector(ABC):
    @abstractmethod
    def select(self, clause: Clause) -> List[int]:
        """Return indices of selected literals."""
        pass

class SelectNegativeSelector(LiteralSelector):
    def select(self, clause: Clause) -> List[int]:
        return [i for i, lit in enumerate(clause.literals) if not lit.polarity]

class SelectSmallestNegativeSelector(LiteralSelector):
    def select(self, clause: Clause) -> List[int]:
        negative_lits = [(i, lit) for i, lit in enumerate(clause.literals) 
                        if not lit.polarity]
        if not negative_lits:
            return []
        
        min_size = min(size(lit) for _, lit in negative_lits)
        return [i for i, lit in negative_lits if size(lit) == min_size]
```

## Indexing Implementation

### Discrimination Tree (for unification)

```python
class DiscriminationTree:
    def __init__(self):
        self.root = DTNode()
    
    def insert(self, term: Term, data: Any):
        """Insert term with associated data."""
        path = term_to_path(term)
        node = self.root
        for symbol in path:
            if symbol not in node.children:
                node.children[symbol] = DTNode()
            node = node.children[symbol]
        node.data.append(data)
    
    def retrieve_unifiable(self, term: Term) -> List[Any]:
        """Retrieve all terms that unify with the query."""
        results = []
        self._retrieve_unifiable(self.root, term_to_path(term), 0, {}, results)
        return results
```

### Feature Vector Index (for filtering)

```python
class FeatureVector:
    def __init__(self, term: Term):
        self.features = {}
        # Count symbols
        for symbol in get_symbols(term):
            self.features[symbol] = self.features.get(symbol, 0) + 1
        # Add depth
        self.features['_depth'] = depth(term)
        # Add variable count
        self.features['_vars'] = count_variables(term)
    
    def compatible_with(self, other: 'FeatureVector') -> bool:
        """Quick check if terms might unify."""
        # Check symbol compatibility
        for symbol in self.features:
            if symbol.startswith('_'):
                continue
            if symbol in other.features:
                # Constants must match exactly
                if is_constant(symbol) and \
                   self.features[symbol] != other.features[symbol]:
                    return False
        return True
```

## Integration with Saturation Loop

```python
class BasicLoop:
    def __init__(self, rules: List[Rule], selector: ClauseSelector):
        self.rules = rules
        self.selector = selector
        self.literal_selector = SelectNegativeSelector()
        self.term_index = DiscriminationTree()
        self.subsumption_index = SubsumptionIndex()
    
    def process_given_clause(self, given: Clause, processed: List[Clause]):
        new_clauses = []
        
        # Binary inferences
        for rule in self.binary_rules:
            eligible_literals = self.literal_selector.select(given)
            candidates = self.term_index.get_candidates(given, eligible_literals)
            
            for partner in candidates:
                result = rule.apply(ProofState(processed, []), [given_idx, partner_idx])
                if result:
                    new_clauses.extend(result.generated_clauses)
        
        # Unary inferences
        for rule in self.unary_rules:
            result = rule.apply(ProofState(processed, []), [given_idx])
            if result:
                new_clauses.extend(result.generated_clauses)
        
        # Forward simplification
        simplified = []
        for clause in new_clauses:
            if not self.is_redundant(clause, processed):
                simplified.append(clause)
        
        return simplified
```

## Rust Implementation Structure

```
src/rules/
├── mod.rs           # Module exports and traits
├── rule.rs          # Rule trait and RuleApplication
├── resolution.rs    # Resolution implementation
├── factoring.rs     # Factoring implementation
├── superposition.rs # Superposition implementation
├── subsumption.rs   # Subsumption checking
├── ordering.rs      # Term ordering (KBO, LPO)
├── selection.rs     # Literal selection strategies
└── indexing/
    ├── mod.rs
    ├── discrimination_tree.rs
    ├── feature_vector.rs
    └── substitution_tree.rs
```

## Testing Strategy

1. **Unit tests** for each rule with specific examples
2. **Property tests** for invariants (soundness)
3. **Integration tests** with complete problems
4. **Performance benchmarks** for indexing structures
5. **Proof verification** to ensure correctness