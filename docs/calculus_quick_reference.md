# ProofAtlas Calculus Quick Reference

## Simplifying Inferences

### Demodulation
```
l ≈ r   P[l'] ∨ C
-------------------  where σ = match(l, l'), lσ ≻ rσ, l ≈ r is a unit equality
   P[rσ] ∨ C
```
- Applied to unit equalities (clauses with exactly one positive equality literal)
- Uses one-way matching: only variables in l can be substituted (not in l')
- The ordering constraint lσ ≻ rσ ensures termination and confluence
- Applied in two contexts:
  1. **Forward demodulation**: When a new clause is generated, it's immediately demodulated by all existing unit equalities before being added to the clause set
  2. **Backward demodulation**: When a unit equality is selected as the given clause, all existing clauses are demodulated by it

## Generating Inferences

### Resolution
```
P(x) ∨ C₁    ¬P(t) ∨ C₂
------------------------  where σ = mgu(x, t), P(x) and ¬P(t) selected.
      (C₁ ∨ C₂)σ
```

### Factoring
```
P(s) ∨ P(t) ∨ C
----------------  where σ = mgu(s, t), P(s) selected.
   (P(s) ∨ C)σ
```

### Superposition 1
```
l ≈ r ∨ C₁    P[l'] ∨ C₂
------------------------  where σ = mgu(l, l'), lσ ⪯̸ rσ, l' is not a variable, P[l'] is not an equality literal, l ≈ r and P[l'] are selected.
   (P[r] ∨ C₁ ∨ C₂)σ
```

### Superposition 2
```
l ≈ r ∨ C₁    s[l'] ⊕ t ∨ C₂
-----------------------------  where σ = mgu(l, l'), lσ ⪯̸ rσ, l' is not a variable, s[l']σ ⪯̸ tσ, ⊕ stands either for ≈ or ≉, l ≈ r and s[l'] ⊕ t are selected.
   (s[r] ⊕ t ∨ C₁ ∨ C₂)σ
```

### Equality Resolution
```
s ≉  t ∨ C
----------- where σ = mgu(s, t),  s ≉  t is selected.
    Cσ
```

### Equality Factoring
```
l ≈ r ∨ s ≈ t ∨ C
--------------------  where σ = mgu(l, s), l ≈ r is selected, lσ ⪯̸ rσ, lσ ⪯̸ tσ, rσ ⪯̸ tσ
(l ≈ r ∨ r ≉ t ∨ C)σ
```

## Simplification Rules

### Forward Subsumption
- Remove new clauses subsumed by existing ones
- `C₁` subsumes `C₂` if `∃σ: C₁σ ⊆ C₂`
- Applied when adding new clauses to prevent redundant clauses from entering the clause set

### Backward Subsumption
- Remove existing clauses subsumed by a newly added clause
- When adding clause `C₁`, remove all existing clauses `C₂` where `C₁` subsumes `C₂`
- Helps reduce the search space by eliminating redundant clauses retroactively
- Particularly effective when deriving general clauses (e.g., unit clauses)

### Tautology Deletion
- Remove clauses containing `P ∨ ¬P`
- Remove clauses containing `t ≈ t`

## Selection Strategies

1. **No Selection**: All literals are selected
2. **Select Negative**: All negative literals eligible
3. **Select First Negative**: First negative literal only
4. **Select Smallest**: Negative literals with minimal size

## Term Orderings

### Knuth-Bendix Ordering (KBO)
- Weight-based ordering
- Variable condition must hold
- Total on ground terms
