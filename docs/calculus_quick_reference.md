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
l ≈ r ∨ C₁    P[l'] ∨ C₂
------------------------  where σ = mgu(l, l'), l ⪯̸ r, l' is not a variable, P[l'] is not an equality literal, l ≈ r and P[l'] are selected.
   (P[r] ∨ C₁ ∨ C₂)σ
```

### Superposition 2
```
l ≈ r ∨ C₁    s[l'] ⊕ t ∨ C₂
-----------------------------  where σ = mgu(l, l'), l ⪯̸ r, l' is not a variable, s[l'] ⪯̸ t, ⊕ stands either for ≈ or ≉, l ≈ r and s[l'] ⊕ t are selected.
   (s[r] ⊕ t ∨ C₁ ∨ C₂)σ
```

### Equality Resolution

s ≉  t ∨ C
----------- where σ = mgu(s, t),  s ≉  t is selected.
    Cσ

### Equality Factoring
```
s ≈ t ∨ s' ≈ t' ∨ C
--------------------  where σ = mgu(s, s'), s ≈ t is selected, sσ ⪯̸ tσ, s'σ ⪯̸ t'σ
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
