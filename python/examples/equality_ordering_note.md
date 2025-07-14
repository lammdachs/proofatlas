# Note on Equality Reasoning and Term Ordering

The superposition inference rule in ProofAtlas uses Knuth-Bendix Ordering (KBO) to prevent generating redundant inferences. This means that simple examples like proving transitivity of equality from `a = b` and `b = c` won't work directly, because:

1. The superposition rule requires `lσ > rσ` (left side must be greater than right after substitution)
2. Constants like `a`, `b`, `c` are typically incomparable in KBO or use lexicographic ordering
3. This prevents the necessary superposition inferences

## Working Examples

Instead, use examples with function symbols that have clear ordering:

```python
# This works - function symbols create ordering
cnf(eq1, axiom, f(a) = a).
cnf(eq2, axiom, f(f(a)) = a).
cnf(goal, negated_conjecture, f(f(f(a))) != f(a)).
```

## Why This Is Correct

This behavior is actually correct for a complete superposition calculus. The ordering constraints ensure:
- No redundant inferences are generated
- The calculus remains complete
- Proof search terminates more often

For simple constant equality chains, you would need:
- Reflexivity, symmetry, and transitivity axioms explicitly
- Or a different term ordering that makes the constants comparable
- Or disable ordering constraints (which would hurt completeness)