# Demodulation in ProofAtlas

## Overview

Demodulation is a crucial simplifying inference rule in equational theorem proving. It uses unit equalities (clauses with exactly one positive equality literal) to rewrite and simplify other clauses.

## The Demodulation Rule

```
l ≈ r   P[l'] ∨ C
-------------------  where σ = match(l, l'), lσ ≻ rσ
   P[rσ] ∨ C
```

Key aspects:
- `l ≈ r` must be a unit equality (single positive equality literal)
- `σ` is computed by one-way matching: only variables in `l` can be substituted
- The ordering constraint `lσ ≻ rσ` must be satisfied (using KBO)
- The rewrite replaces all occurrences of `l'` with `rσ` in the clause

## Implementation Details

### One-Way Matching

ProofAtlas uses a dedicated `match_term` function that only allows substitution of variables from the pattern (left-hand side of the equality). This is crucial for soundness:

```rust
// Correct: mult(inv(X),X) matches mult(inv(a),a) with X → a
// Incorrect: mult(inv(X),X) should NOT match mult(inv(Y),mult(Y,Z))
```

### Ordering Constraints

The Knuth-Bendix Ordering (KBO) is used to ensure:
1. **Termination**: Rewrites always decrease term size according to the ordering
2. **Confluence**: Different rewrite sequences lead to the same normal form

## Application Strategies

### Forward Demodulation

When a new clause is generated:
1. Orient equalities in the clause
2. Apply all available unit equalities from the processed set
3. Repeat until no more rewrites are possible
4. Only then check subsumption and add to clause set

This ensures clauses are in simplified form before entering the search space.

### Backward Demodulation

When a unit equality is selected as the given clause:
1. Check all existing clauses (both processed and unprocessed)
2. Attempt to demodulate each clause with the new unit equality
3. Replace simplified clauses (removing the old version)
4. Add replacements through the normal clause addition process

This is particularly powerful when discovering general simplifying equalities like `mult(e,X) = X`.

## Performance Impact

Backward demodulation can dramatically reduce the search space:
- **Example**: In group theory, discovering `mult(e,X) = X` can simplify hundreds of existing clauses
- **Measurement**: The `uniqueness_of_inverse` test improved from timeout (>10s) to 0.013s

## Common Patterns

### Identity Simplification
```
mult(e,X) = X     mult(e,mult(a,b)) = c
----------------------------------------
         mult(a,b) = c
```

### Inverse Simplification
```
mult(inv(X),X) = e     mult(inv(a),a) = mult(b,c)
--------------------------------------------------
              e = mult(b,c)
```

## Debugging Demodulation

To trace demodulation:
1. Check which unit equalities are available when a clause is processed
2. Verify the matching succeeds (variables in pattern match terms in target)
3. Confirm the ordering constraint is satisfied after substitution
4. Ensure the rewritten clause is actually simpler

Common issues:
- Using full unification instead of one-way matching (unsound)
- Not checking ordering constraints (non-terminating)
- Missing backward demodulation opportunities (inefficient)