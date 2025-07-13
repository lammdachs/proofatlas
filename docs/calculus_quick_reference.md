# ProofAtlas Calculus Quick Reference

## Inference Rules

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
l ≈ r ∨ C₁    P[s] ∨ C₂
------------------------  where σ = mgu(l, s), lσ ≻ rσ, P[s] is not and equality literal, l ≈ r and P[s] are selected.
   (P[r] ∨ C₁ ∨ C₂)σ
```

### Superposition 2
```
l ≈ r ∨ C₁    t[s] ⊕ t' ∨ C₂
-----------------------------  where σ = mgu(l, s), lσ ≻ rσ, t[s]σ ≻ t'σ, ⊕ stands either for ≈ or ≉ , l ≈ r and t[s] ⊕ t' are selected.
   (t[r] ⊕ t' ∨ C₁ ∨ C₂)σ
```

### Equality Resolution

s ≉  t ∨ C
----------- where σ = mgu(s, t),  s ≉  t is selected.
    Cσ

### Equality Factoring
```
s ≈ t ∨ s' ≈ t' ∨ C
--------------------  where σ = mgu(s, s'), s ≈ t is selected, sσ ≻ tσ, s'σ ≻ t'σ
(t ≉ t' ∨ s ≈ t ∨ C)σ
```

## Simplification Rules

### Forward Subsumption
- Remove new clauses subsumed by existing ones
- `C₁` subsumes `C₂` if `∃σ: C₁σ ⊆ C₂`

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

### Lexicographic Path Ordering (LPO)
- Precedence-based ordering
- Recursive comparison
- Simplification ordering

## Implementation Priorities

1. **Core**: Resolution + Factoring + Forward Subsumption
2. **Equality**: Superposition + Equality Factoring
3. **Optimization**: Indexing via Discrimination Trees
4. **Advanced**: AC handling, Theory integration
