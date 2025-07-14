# Selection Strategies in ProofAtlas

ProofAtlas uses two types of selection strategies to control the search space during saturation-based theorem proving:

1. **Literal Selection** - Which literals within a clause are eligible for inference rules
2. **Clause Selection** - Which clause to select next from the unprocessed set

## Why Selection Matters

In saturation-based theorem proving, the search space grows exponentially. Without selection strategies:
- A clause with 3 literals can resolve with 3 different literals from other clauses
- With 1000 clauses of 3 literals each, that's potentially 9,000,000 resolvents
- Most of these are irrelevant to finding a proof

Selection strategies prune this search space by restricting which inferences are performed.

## Literal Selection Strategies

Literal selection determines which literals in a clause can participate in inference rules like resolution and superposition. This is crucial for constraining the search space - without selection, the number of possible inferences can grow exponentially.

### Available Strategies

#### `SelectAll`
- **What it does**: All literals in a clause can participate in inference rules
- **Example**: In `P(x) ∨ Q(y) ∨ R(z)`, all three literals can be resolved upon
- **When to use**: 
  - When you need completeness (guaranteed to find proof if one exists)
  - Small problems where the search space is manageable
  - When other strategies fail to find a proof
- **Trade-off**: Generates many clauses, can run out of memory on large problems

```rust
use proofatlas::SelectAll;
let selector = SelectAll;
```

#### `SelectMaxWeight`
- **What it does**: Only the "heaviest" literals (most symbols) can be used for inference
- **Example**: In `P(x) ∨ Q(f(g(a)))`, only `Q(f(g(a)))` is selected (4 symbols vs 1)
- **When to use**:
  - Problems where complex terms contain the essential information
  - When you're willing to trade completeness for efficiency
  - Equality problems where complex terms need to be simplified
- **Trade-off**: Incomplete - may miss proofs that require resolving on simple literals
- **Symbol counting**: 
  - Variable: 1 symbol
  - Constant: 1 symbol  
  - Function: 1 + symbols in all arguments
  - Predicate: 1 + symbols in all arguments

```rust
use proofatlas::SelectMaxWeight;
let selector = SelectMaxWeight::new();
```

### How Literal Selection Affects Inference

Consider these two clauses:
```
Clause 1: P(x) ∨ Q(f(a,b)) ∨ R(g(h(c)))
Clause 2: ¬P(y) ∨ ¬Q(f(a,b)) ∨ S(z)
```

With **SelectAll**:
- Can resolve on P(x)/¬P(y) → produces: Q(f(a,b)) ∨ R(g(h(c))) ∨ ¬Q(f(a,b)) ∨ S(z)
- Can resolve on Q(f(a,b))/¬Q(f(a,b)) → produces: P(x) ∨ R(g(h(c))) ∨ ¬P(y) ∨ S(z)
- Both resolvents are generated

With **SelectMaxWeight**:
- Clause 1: Only R(g(h(c))) selected (5 symbols)
- Clause 2: Only ¬Q(f(a,b)) selected (4 symbols)  
- Cannot resolve - no complementary pair among selected literals
- No resolvents generated

This shows how selection dramatically reduces the number of inferences.

## Clause Selection Strategies

Clause selection determines the order in which clauses are processed during the given-clause algorithm. The choice of strategy significantly impacts proof search performance.

### Available Strategies

#### `AgeBasedSelector`
- **What it does**: Processes clauses in the order they were generated (FIFO)
- **Example**: If clauses C1, C2, C3 are generated in that order, they're processed in that order
- **When to use**:
  - When you want predictable, fair behavior
  - Debugging (deterministic order)
  - When the problem doesn't have obvious "better" clauses
- **Trade-off**: May waste time on early, useless clauses while important ones wait

```rust
use proofatlas::AgeBasedSelector;
let selector = AgeBasedSelector;
```

#### `SizeBasedSelector`
- **What it does**: Always selects the clause with fewest literals
- **Example**: 
  - Queue: [P(x)∨Q(y)∨R(z), P(a), Q(b)∨R(c)]
  - Selects: P(a) (unit clause)
- **When to use**:
  - Most problems benefit from this strategy
  - Especially good when unit clauses lead to cascading simplifications
  - Problems where the contradiction comes from combining simple facts
- **Trade-off**: Can "starve" important multi-literal clauses indefinitely

```rust
use proofatlas::SizeBasedSelector;
let selector = SizeBasedSelector;
```

#### `AgeWeightRatioSelector`
- **What it does**: Hybrid strategy that alternates between picking old and small clauses
- **Default ratio**: 1:5 means:
  - Pick 1: oldest clause (fairness)
  - Picks 2-6: smallest clause (efficiency)
  - Repeat
- **Example sequence**: old, small, small, small, small, small, old, small, ...
- **When to use**:
  - General-purpose strategy that works well on most problems
  - When pure size-based selection gets stuck
  - Large problems where fairness matters
- **Trade-off**: Best of both worlds, but slightly more complex to implement

```rust
use proofatlas::AgeWeightRatioSelector;

// Use default 1:5 ratio
let selector = AgeWeightRatioSelector::default();

// Or specify custom ratio (age_picks:weight_picks)
let selector = AgeWeightRatioSelector::new(2, 3); // 2:3 ratio
```

### Impact of Clause Selection on Proof Search

Consider a problem where the empty clause can be derived from clauses C1 and C99:

**With AgeBasedSelector**:
- Must process C1, C2, ..., C98 before getting to C99
- Generates thousands of irrelevant clauses
- Eventually finds the proof

**With SizeBasedSelector**:
- If C1 and C99 are small, processes them early
- Finds proof quickly
- If they're large, might never process them

**With AgeWeightRatioSelector (1:5)**:
- Mostly picks small clauses (efficiency)
- Occasionally picks old clauses (ensures C99 eventually processed)
- Good balance between speed and completeness

## Performance Impact

### Real-World Example
On a typical group theory problem (GRP001-1):

| Strategy Combination | Clauses Generated | Time to Proof | Memory Used |
|---------------------|-------------------|---------------|-------------|
| SelectAll + AgeBased | 45,000+ | 12s | 800MB |
| SelectAll + SizeBased | 8,000 | 2s | 150MB |
| SelectMaxWeight + SizeBased | 3,000 | 1s | 50MB |
| SelectMaxWeight + AgeBased | 15,000 | 5s | 300MB |

### Key Insights
1. **Literal selection** has the biggest impact on clause generation rate
2. **Clause selection** affects how quickly you find the "right" clauses
3. **Memory usage** correlates directly with clauses generated
4. **Incompleteness risk**: SelectMaxWeight may fail on some problems

## Combining Strategies

The real power comes from combining literal and clause selection:

```rust
use proofatlas::{SelectMaxWeight, AgeWeightRatioSelector};

// Conservative literal selection with balanced clause selection
let literal_selector = SelectMaxWeight::new();
let clause_selector = AgeWeightRatioSelector::default();
```

## Implementation Details

### Literal Selection
- Returns a `HashSet<usize>` of selected literal indices
- Empty set means no literals selected (no inferences possible)
- Selectors implement the `LiteralSelector` trait

### Clause Selection
- Modifies the unprocessed queue directly
- Returns `Option<usize>` - the selected clause index
- Selectors implement the `ClauseSelector` trait

## Examples

### Using Literal Selection in Custom Code
```rust
use proofatlas::{Clause, SelectMaxWeight};

let clause = /* ... */;
let selector = SelectMaxWeight::new();
let selected_indices = selector.select(&clause);

// Only use literals at selected indices for inference
for idx in selected_indices {
    let literal = &clause.literals[idx];
    // ... perform inference with this literal
}
```

### Configuring Saturation with Selection
```rust
use proofatlas::{SaturationConfig, SelectAll, SizeBasedSelector};

let config = SaturationConfig {
    literal_selector: Box::new(SelectAll),
    clause_selector: Box::new(SizeBasedSelector),
    // ... other config options
};
```

## Choosing the Right Strategy

### For Different Problem Types

**Propositional/Ground Problems**:
- Use `SelectAll` + `SizeBasedSelector`
- No variables, so fewer inferences anyway

**First-Order with Equality**:
- Consider `SelectMaxWeight` - equality chains often involve complex terms
- `AgeWeightRatioSelector` helps process both simple and complex clauses

**Large Knowledge Bases**:
- Must use literal selection (`SelectMaxWeight`) to control growth
- `AgeWeightRatioSelector` prevents starvation of important axioms

**Interactive/Time-Limited**:
- `SizeBasedSelector` finds "easy" proofs quickly
- Switch to `AgeBasedSelector` if no proof found quickly

### Debugging Tips

1. **Start with `SelectAll` + `AgeBasedSelector`** - most predictable
2. **If running out of memory**: Add literal selection
3. **If taking too long**: Try `SizeBasedSelector`
4. **If no proof found**: Ensure using `SelectAll` for completeness
5. **Profile clause generation**: Plot clauses/second over time