# Saturation Loop Design

## Overview

The saturation loop implements the core of a saturation-based theorem prover using the given clause algorithm. It takes a single clause (the "given clause") and applies inference rules to generate new clauses, maintaining a separation between processed and unprocessed clauses.

## Current Implementation: BasicLoop

The `BasicLoop` class in `loops/basic.py` provides a complete implementation of the given clause algorithm with the following features:

### Inference Rules

1. **Resolution Rule** - Binary resolution between the given clause and each previously processed clause
2. **Factoring Rule** - Factoring applied to the given clause to eliminate duplicate literals

### Simplification and Redundancy Elimination

The implementation includes several mechanisms to prevent redundant clauses:

1. **Forward Simplification** (enabled by default):
   - **Tautology deletion**: Clauses containing complementary literals (e.g., P(a) ∨ ¬P(a)) are discarded
   - **Forward subsumption**: New clauses subsumed by existing clauses are discarded
   - **Duplicate checking**: Exact duplicate clauses are filtered out

2. **Backward Simplification** (not yet implemented):
   - Would remove existing clauses that become redundant after new clauses are generated
   - Currently a TODO in the codebase

3. **Size Limits**:
   - Clauses exceeding `max_clause_size` (default: 100 literals) are discarded
   - Prevents memory explosion from very large clauses

### Proof Tracking

The loop maintains complete proof history:
- Each step records the state BEFORE processing the given clause
- Shows which clause was selected as the given clause
- Records all inference rules applied and their results
- **Important**: Only rule applications that produce non-redundant clauses are recorded

### Single Step Algorithm

Given a proof and a clause index, `step()` performs:

1. **Factor the Selected Clause**: If the selected clause can be factored to a simpler form (fewer literals), the factored version is used instead
2. **Move to Processed**: The (possibly factored) clause is added to the processed set
3. **Generate Inferences**: 
   - Resolution between the given clause and each previously processed clause
4. **Filter Results**:
   - Apply size limits
   - Check for tautologies (if forward_simplify=True)
   - Check for subsumption (if forward_simplify=True)
   - Remove exact duplicates
5. **Update Rule Applications**: Only keep rules that produced non-redundant clauses
   - Note: Factoring applications are always kept if factoring occurred
6. **Create New State**: Add kept clauses from resolution to unprocessed
7. **Record Step**: Add step to proof with selected clause and applied rules

### Constructor Parameters

```python
BasicLoop(max_clause_size=100, forward_simplify=True, backward_simplify=True, use_indexing=True)
```

- `max_clause_size`: Maximum number of literals in a kept clause
- `forward_simplify`: Enable tautology and subsumption checking for new clauses
- `backward_simplify`: Currently unused (placeholder for future implementation)
- `use_indexing`: Enable clause indexing for efficient resolution partner finding (default: True)

### Design Decisions

1. **Parent Tracking**: Rule applications only show the index of the processed clause, not the given clause (which is implicit)
2. **Redundancy Filtering**: If all clauses from a rule application are redundant, the entire rule application is omitted from the proof
3. **State Immutability**: Each step creates new state objects rather than modifying existing ones
4. **No Clause Selection**: The loop is completely independent of clause selection strategy
5. **Immediate Factoring**: When a clause with duplicate literals is selected, it is immediately factored and the simpler version is added to processed instead of the original

### Usage Example

```python
loop = BasicLoop(max_clause_size=50, forward_simplify=True)
proof = Proof(initial_state)
proof = loop.step(proof, given_clause=0)  # Process first unprocessed clause
```

The loop integrates seamlessly with any clause selection strategy, making it a flexible foundation for saturation-based theorem proving.

### Performance Optimizations

#### Clause Indexing (Optional)

The BasicLoop includes an optional efficient clause indexing system to reduce the number of resolution attempts. This is controlled by the `use_indexing` parameter (default: True).

1. **Predicate Index**: When enabled, maintains a mapping from `(predicate_symbol, polarity)` to sets of clause indices
   - Built for all processed clauses before resolution
   - Allows O(1) lookup of clauses containing specific predicates

2. **Resolution Candidate Finding**: Before attempting resolution, the loop:
   - Identifies clauses with complementary literals (same predicate, opposite polarity)
   - Only attempts resolution with these candidate clauses
   - Dramatically reduces failed resolution attempts

**Example Impact**: With 1000 processed clauses where only 10 have complementary literals with the given clause:
- **Without indexing** (`use_indexing=False`): 1000 resolution attempts (990 will fail immediately)
- **With indexing** (`use_indexing=True`): ~10 resolution attempts (only with viable candidates)

This optimization is particularly effective when:
- The number of processed clauses is large
- Predicates are diverse across clauses
- Many clauses share no common predicates with the given clause

You might disable indexing (`use_indexing=False`) for:
- Small problems where the indexing overhead isn't worth it
- Debugging to ensure the indexing doesn't affect correctness
- Comparing performance with and without indexing

### Known Limitations

1. **No Backward Simplification**: The current implementation does not remove existing clauses that are subsumed by newly generated clauses. This could lead to redundant clauses remaining in the processed/unprocessed sets.

2. **Index Rebuilding**: The predicate index is rebuilt for each step rather than maintained incrementally. This could be optimized further for very large clause sets.

3. **Basic Clause Selection**: The loop requires external clause selection strategy. More sophisticated selection heuristics could improve performance.

### Implementation Notes

- **Subsumption with Duplicate Literals**: The subsumption check properly handles clauses with duplicate literals by tracking which literals have been matched, ensuring that P(a) ∨ P(a) does not subsume P(a).

- **Index Structure**: The predicate index uses a `defaultdict(set)` for efficient addition and lookup operations.