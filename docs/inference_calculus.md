# ProofAtlas Inference Calculus

This document describes the complete inference calculus implemented in ProofAtlas, including resolution, paramodulation (superposition), literal selection, and indexing strategies.

## Table of Contents

1. [Core Inference Rules](#core-inference-rules)
2. [Equality Reasoning](#equality-reasoning)
3. [Simplification Rules](#simplification-rules)
4. [Literal Selection](#literal-selection)
5. [Term Indexing](#term-indexing)
6. [Implementation Notes](#implementation-notes)

## Core Inference Rules

### 1. Binary Resolution

The fundamental inference rule for propositional and first-order logic.

```
C₁ = L ∨ C₁'    C₂ = ¬L' ∨ C₂'
-------------------------------- (Resolution)
      (C₁' ∨ C₂')σ
```

Where:
- `L` and `L'` are literals that unify with MGU (most general unifier) `σ`
- `C₁'` and `C₂'` are the remaining literals in their respective clauses
- The resolvent is the disjunction of the remaining literals with the unifier applied

**Example:**
```
P(x) ∨ Q(x)    ¬P(a) ∨ R(a)
---------------------------
      Q(a) ∨ R(a)
```

### 2. Factoring

Eliminates duplicate literals within a clause.

```
C = L₁ ∨ L₂ ∨ C'
----------------- (Factoring)
  (L₁ ∨ C')σ
```

Where:
- `L₁` and `L₂` unify with MGU `σ`
- The factor contains only one copy of the unified literal

**Example:**
```
P(x) ∨ P(f(a)) ∨ Q(x)
----------------------
P(f(a)) ∨ Q(f(a))
```

## Equality Reasoning

### 3. Superposition (Paramodulation)

The primary inference rule for equality reasoning, replacing the older paramodulation rule.

#### Superposition Right (into positive literal)
```
C₁ = l ≈ r ∨ C₁'    C₂ = P[s] ∨ C₂'
------------------------------------ (Sup-R)
        (P[r] ∨ C₁' ∨ C₂')σ
```

#### Superposition Left (into negative literal)
```
C₁ = l ≈ r ∨ C₁'    C₂ = ¬P[s] ∨ C₂'
------------------------------------- (Sup-L)
        (¬P[r] ∨ C₁' ∨ C₂')σ
```

Where:
- `s` and `l` unify with MGU `σ`
- `P[s]` denotes a literal containing term `s` at some position
- `P[r]` denotes the same literal with `s` replaced by `r`
- Ordering constraints: `lσ ≻ rσ` and `P[s]σ ≻ P[r]σ`

**Example:**
```
f(a) ≈ b    P(f(a), x)
----------------------
      P(b, x)
```

### 4. Equality Factoring

Specialized factoring for equality literals.

```
C = s ≈ t ∨ s' ≈ t' ∨ C'
------------------------- (Eq-Fact)
(s ≈ t ∨ t ≉ t' ∨ C')σ
```

Where:
- `s` and `s'` unify with MGU `σ`
- The result includes an inequality literal

**Example:**
```
f(x) ≈ a ∨ f(b) ≈ c ∨ P(x)
---------------------------
f(b) ≈ a ∨ a ≉ c ∨ P(b)
```

## Simplification Rules

### 5. Forward Subsumption

Removes clauses that are logically implied by other clauses.

```
C₁ subsumes C₂ if ∃σ: C₁σ ⊆ C₂
```

A clause `C₁` subsumes `C₂` if there exists a substitution `σ` such that every literal in `C₁σ` appears in `C₂`.

**Example:**
```
C₁: P(x)
C₂: P(a) ∨ Q(b)
→ C₁ subsumes C₂ with σ = {x ↦ a}
```

### 6. Backward Subsumption (TODO)

Removes existing clauses that are subsumed by newly derived clauses.

### 7. Tautology Deletion

Removes clauses that are always true.

A clause is a tautology if:
- It contains complementary literals (e.g., `P(a) ∨ ¬P(a)`)
- It contains a reflexive equality (e.g., `t ≈ t`)

## Literal Selection

Literal selection strategies determine which literals in a clause are eligible for inference. This is crucial for completeness and efficiency.

### Selection Function

A selection function `sel` maps each clause to a subset of its negative literals:
- `sel(C) ⊆ {L ∈ C | L is negative}`
- If `sel(C) = ∅`, all maximal literals are eligible

### Common Selection Strategies

1. **Select All Negative**: Select all negative literals
2. **Select First Negative**: Select the first negative literal
3. **Select Smallest Negative**: Select negative literals with smallest size
4. **No Selection**: Use ordering-based selection only

### Ordering Constraints

When no literals are selected, only maximal literals (w.r.t. a term ordering) participate in inferences.

## Term Indexing

Efficient term indexing is essential for finding unifiable literals and checking subsumption.

### 1. Discrimination Trees

Used for finding terms that unify with a query term.
- Path from root to leaf represents term structure
- Variables are treated specially to handle unification

### 2. Feature Vector Indexing

Used for quickly filtering candidates for unification.
- Each term mapped to a vector of features
- Features include: symbol counts, depth, variables

### 3. Substitution Trees

Used for forward subsumption checking.
- Stores generalizations of terms
- Efficient for finding more general terms

### 4. Fingerprint Indexing

Used for backward subsumption and equality matching.
- Compact representation of term structure
- Fast filtering before full unification check

## Implementation Notes

### Given Clause Algorithm

The saturation loop follows the given clause algorithm:

```python
while unprocessed:
    given = select_clause(unprocessed)
    unprocessed.remove(given)
    
    # Generate new clauses
    new_clauses = []
    for clause in processed:
        new_clauses.extend(resolve(given, clause))
        new_clauses.extend(factor(given))
        if equality_present:
            new_clauses.extend(superposition(given, clause))
    
    # Simplification
    new_clauses = forward_simplify(new_clauses, processed + [given])
    processed, unprocessed = backward_simplify(processed, unprocessed, new_clauses)
    
    processed.add(given)
    unprocessed.extend(new_clauses)
```

### Completeness

The calculus is refutationally complete for first-order logic with equality when:
1. A complete term ordering is used
2. The selection function is "admissible"
3. All inference rules are applied fairly

### Redundancy Elimination

A clause C is redundant if:
- It is a tautology
- It is subsumed by another clause
- It can be simplified by rewriting with smaller equations

### Proof Recording

Each inference records:
- Parent clause indices
- Applied rule and unifier
- Position information (for superposition)
- Selected literals

This enables proof reconstruction and verification.

## Future Extensions

1. **AC Handling**: Special treatment of associative-commutative operators
2. **Theory Integration**: Built-in theories (arithmetic, arrays, etc.)
3. **Model Building**: Constructing models for satisfiable problems
4. **Strategy Scheduling**: Automatic selection of parameters
5. **Parallel Inference**: Distributed proof search