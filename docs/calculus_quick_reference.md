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
P ∨ C₁    ¬Q ∨ C₂
-----------------  where σ = mgu(P, Q), P and ¬Q selected.
    (C₁ ∨ C₂)σ
```

### Factoring
```
P ∨ Q ∨ C
-----------  where σ = mgu(P, Q), P selected.
 (P ∨ C)σ
```

### Superposition
```
l ≈ r ∨ C₁    L[l'] ∨ C₂
------------------------  where σ = mgu(l, l'), lσ ⪯̸ rσ, l' not a variable, l ≈ r and L[l'] selected.
   (L[r] ∨ C₁ ∨ C₂)σ        If L[l'] is s[l'] ⊕ t (equality literal), also require s[l']σ ⪯̸ tσ.
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

## Literal Selection Strategies

Based on Hoder et al. "Selecting the selection" (2016), matching Vampire's numbering:

| # | Strategy | Description |
|---|----------|-------------|
| **0** | Select All | All literals are selected (no restriction) |
| **20** | Select Maximal | Select all maximal literals (using KBO) |
| **21** | Unique Maximal | Select unique maximal if exists, else max-weight negative, else all maximal |
| **22** | Neg Max-Weight | Select max-weight negative if exists, else all maximal |

## Term Orderings

### Knuth-Bendix Ordering (KBO)

Let `#(x, s)` be the number of occurrences of variable `x` in term `s`.

**Definition:** `s > t` if:

1. `#(x, s) ≥ #(x, t)` for all variables `x` **AND** `|s| > |t|`

2. `#(x, s) ≥ #(x, t)` for all variables `x` **AND** `|s| = |t|` **AND** one of the following holds:
   - **2.1** `s = g(...)`, `t = h(...)` and `g ≫ h` by precedence (alphabetic)
   - **2.2** `s = g(s₁,...,sₘ)`, `t = g(t₁,...,tₘ)` and for some `1 ≤ i ≤ m`: `s₁ = t₁, ..., sᵢ₋₁ = tᵢ₋₁` and `sᵢ > tᵢ`

**Properties:**
- Total on ground terms (no variables)
- Partial on non-ground terms (variable condition may fail)
- Well-founded and stable under substitution

**Extension to Atoms/Literals:**
- Predicates are compared by alphabetic precedence (like function symbols)
- Same predicate: lexicographic comparison of arguments
- Variable condition applies to the entire atom
