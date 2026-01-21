# Performance Analysis Report

Analysis of ProofAtlas codebase for performance anti-patterns, inefficient algorithms, and optimization opportunities.

## Executive Summary

The ProofAtlas theorem prover has several performance bottlenecks concentrated in:
1. **Saturation loop** - O(n) linear scans over processed clauses
2. **Subsumption checking** - Repeated string formatting and linear scans
3. **Substitution propagation** - O(n²) worst-case behavior
4. **Web frontend** - Eager loading and DOM manipulation patterns

---

## 1. Rust Core: Saturation Loop Issues

### 1.1 Forward Simplification - N+1 Pattern
**Location**: `crates/proofatlas/src/saturation/state.rs:345-373`

```rust
fn forward_simplify_given(&mut self, given_idx: usize) -> usize {
    // ...
    while changed {
        changed = false;
        for &unit_idx in &self.processed {  // O(n) scan for EACH iteration
            let unit_clause = &self.clauses[unit_idx];
            if unit_clause.literals.len() == 1 /* ... */ {
                let results = demodulation::demodulate(unit_clause, &current_clause, ...);
                // ...
            }
        }
    }
}
```

**Issue**: For every given clause selection, this iterates over ALL processed clauses to find unit equalities. With 1000+ processed clauses, this is O(n) per selection, leading to O(n²) total.

**Fix**: Maintain a separate index of unit equalities:
```rust
struct SaturationState {
    processed_unit_equalities: Vec<usize>,  // Pre-filtered index
    // ...
}
```

### 1.2 Inference Generation - Full Scan
**Location**: `crates/proofatlas/src/saturation/state.rs:409-476`

```rust
fn generate_inferences(&self, given_idx: usize) -> Vec<InferenceResult> {
    for &processed_idx in &self.processed {  // O(n) for each given clause
        results.extend(resolution(...));
        results.extend(superposition(...));
    }
}
```

**Issue**: Every given clause must test against every processed clause. This is fundamental to the algorithm but could benefit from indexing.

**Fix**: Use discrimination trees or path indexing to filter incompatible clauses before attempting unification.

### 1.3 Backward Demodulation - Double Iteration
**Location**: `crates/proofatlas/src/saturation/state.rs:531-586`

```rust
fn backward_demodulate_with_unit(&mut self, unit_idx: usize) {
    for &idx in &self.processed { /* ... */ }  // First pass: collect
    for &idx in &self.unprocessed { /* ... */ }  // Second pass
    for clause_idx in clauses_to_demodulate {  // Third pass: apply
        // ...
    }
}
```

**Issue**: Three separate iterations over clause sets when a unit equality is processed.

---

## 2. Subsumption Checking Issues

### 2.1 String Formatting Overhead
**Location**: `crates/proofatlas/src/saturation/subsumption.rs:65-79`

```rust
pub fn add_clause(&mut self, clause: Clause) -> usize {
    let clause_str = format!("{}", clause);  // Allocation on EVERY add
    self.clause_strings.insert(clause_str);
    // ...
}

pub fn is_subsumed(&self, clause: &Clause) -> bool {
    let clause_str = format!("{}", clause);  // ANOTHER allocation
    if self.clause_strings.contains(&clause_str) {
        return true;
    }
    // ...
}
```

**Issue**: `format!` allocates a new String for every clause added and every subsumption check. With thousands of clauses, this creates significant allocation pressure.

**Fix**: Use a hash of the clause structure instead:
```rust
fn clause_hash(clause: &Clause) -> u64 {
    // Compute hash without allocation
}
```

### 2.2 Linear Variant Scan
**Location**: `crates/proofatlas/src/saturation/subsumption.rs:176-197`

```rust
fn has_variant(&self, clause: &Clause) -> bool {
    for existing in &self.clauses {  // O(n) scan
        if existing.literals.len() != clause.literals.len() {
            continue;
        }
        if get_clause_shape(existing) != shape {
            continue;
        }
        if are_variants(existing, clause) { /* ... */ }
    }
}
```

**Issue**: Checks every clause in the set for variants. With N clauses, this is O(N) per check.

**Fix**: Index clauses by their shape (predicate set + arity) for O(1) lookup of candidates.

### 2.3 Repeated Shape Computation
**Location**: `crates/proofatlas/src/saturation/subsumption.rs:201-209`

```rust
fn get_clause_shape(clause: &Clause) -> Vec<(String, bool)> {
    let mut shape: Vec<_> = clause.literals.iter()
        .map(|lit| (lit.atom.predicate.name.clone(), lit.polarity))  // Clone on each call
        .collect();
    shape.sort();
    shape
}
```

**Issue**: Shape is recomputed and sorted on every variant check, with string cloning.

---

## 3. Substitution Propagation

### 3.1 Eager Normalization - O(n²) Worst Case
**Location**: `crates/proofatlas/src/core/substitution.rs:27-55`

```rust
pub fn insert_normalized(&mut self, var: Variable, term: Term) {
    let normalized_term = term.apply_substitution(self);  // Apply to new term
    self.map.insert(var.clone(), normalized_term);

    // Now apply NEW binding to ALL existing mappings
    let mut updated_map = HashMap::new();
    for (existing_var, existing_term) in self.map.iter() {
        if existing_var != &var {
            let single_subst = Substitution { /* ... */ };
            updated_map.insert(
                existing_var.clone(),
                existing_term.apply_substitution(&single_subst),
            );
        }
    }
    self.map = updated_map;
}
```

**Issue**: Every new binding triggers re-processing of ALL existing bindings. With deep derivation chains:
- Binding 1: 0 updates
- Binding 2: 1 update
- Binding n: n-1 updates
- Total: O(n²)

**Fix**: Use lazy substitution or persistent data structures that share structure.

### 3.2 Excessive Cloning in Apply
**Location**: `crates/proofatlas/src/core/substitution.rs:77-92`

```rust
impl Term {
    pub fn apply_substitution(&self, subst: &Substitution) -> Term {
        match self {
            Term::Variable(v) => subst.map.get(v).cloned().unwrap_or_else(|| self.clone()),
            Term::Constant(_) => self.clone(),  // Clone even when unchanged
            Term::Function(f, args) => {
                let new_args = args.iter()
                    .map(|arg| arg.apply_substitution(subst))  // Recursive clone
                    .collect();
                Term::Function(f.clone(), new_args)
            }
        }
    }
}
```

**Issue**: Every substitution application clones the entire term tree, even when no substitutions apply.

**Fix**: Return `Cow<Term>` or use reference counting (`Rc<Term>`).

---

## 4. Superposition Rule

### 4.1 Recursive Position Finding
**Location**: `crates/proofatlas/src/inference/superposition.rs:189-210`

```rust
fn find_positions_in_term(
    term: &Term,
    pattern: &Term,
    path: Vec<usize>,  // Cloned at each level
    positions: &mut Vec<Position>,
) {
    if could_unify(term, pattern) {
        positions.push(Position {
            term: term.clone(),  // Clone on every potential match
            path: path.clone(),
        });
    }
    if let Term::Function(_, args) = term {
        for (i, arg) in args.iter().enumerate() {
            let mut new_path = path.clone();  // Clone path at each recursion
            new_path.push(i);
            find_positions_in_term(arg, pattern, new_path, positions);
        }
    }
}
```

**Issue**:
1. Path vector is cloned at every recursion level
2. Term is cloned even for positions that won't unify
3. No early termination when one match is sufficient

---

## 5. ML Selector Caching

### 5.1 String-Based Cache Keys
**Location**: `crates/proofatlas/src/selectors/cached.rs:105-107`

```rust
fn clause_to_key(clause: &Clause) -> String {
    clause.to_string()  // Full string representation as key
}
```

**Issue**: Using the full string representation as cache key requires:
1. Formatting the entire clause to string
2. Hashing a potentially long string
3. Storing duplicate string data

**Fix**: Use clause ID or structural hash as key.

### 5.2 GCN Embedding Per-Clause Overhead
**Location**: `crates/proofatlas/src/selectors/burn_gcn.rs:385-423`

```rust
fn compute_and_cache_embedding(&mut self, clause_id: usize, clause: &Clause) {
    let graph = GraphBuilder::build_from_clause_with_context(clause, 1);
    // Build tensors for single clause
    let node_features = Tensor::from_floats(...);
    let adj = self.build_single_adjacency(...);  // Dense matrix!
    let pool_matrix = Tensor::from_floats(...);
    let embedding = self.model.encode(node_features, adj, pool_matrix);
}
```

**Issue**: Each clause embedding is computed individually with:
1. Full graph construction
2. Dense adjacency matrix (O(n²) space for n nodes)
3. Separate tensor allocations

**Fix**: Batch multiple clause embeddings together and use sparse adjacency.

---

## 6. Web Frontend Issues

### 6.1 Eager Example Loading
**Location**: `web/app.js:24-41`

```javascript
async function loadExamples() {
    // Load ALL example files upfront
    for (const example of exampleMetadata) {
        const fileResponse = await fetch(`examples/${example.file}`);
        const content = await fileResponse.text();
        exampleContents[example.id] = content;  // All stored in memory
    }
}
```

**Issue**: All examples are loaded at startup, regardless of whether they'll be used.

**Fix**: Load examples on-demand when selected.

### 6.2 Proof Inspector DOM Rebuild
**Location**: `web/app.js:215-262`

```javascript
goToStep(groupNum) {
    // Reset state and replay up to the target group
    this.processedClauses.clear();
    this.unprocessedClauses.clear();

    // Replay ALL steps from beginning
    for (let i = 0; i <= groupNum; i++) {
        // ... process each group
    }
    this.render();  // Full DOM rebuild
}
```

**Issue**: Navigation to any step replays from the beginning and rebuilds the entire DOM.

**Fix**:
1. Cache intermediate states
2. Use incremental DOM updates (e.g., virtual DOM diff)

### 6.3 innerHTML Manipulation in Loops
**Location**: `web/app.js:336-344`

```javascript
renderClauseList(divId, countId, clauseIds) {
    const clauseArray = Array.from(clauseIds).sort((a, b) => a - b);
    div.innerHTML = clauseArray.map(id => {
        const clause = this.getClauseById(id);  // O(n) lookup each time!
        return `<div class="clause-item">[${id}] ${escapeHtml(clause.clause)}</div>`;
    }).join('');
}
```

**Issue**:
1. `getClauseById` does linear scan through all clauses for each ID
2. Full innerHTML replacement triggers layout reflow

---

## 7. Python Training Pipeline

### 7.1 Per-Proof Data Loading
**Location**: `python/proofatlas/ml/training.py:221-278`

```python
def __getitem__(self, idx):
    with open(self.trace_files[idx]) as f:
        trace = json.load(f)  # Full JSON parse for each item

    max_age = len(clauses)
    graphs = [self._clause_to_graph(c, max_age) for c in clauses]  # Convert each clause
```

**Issue**: Each training sample requires:
1. Full JSON file parse from disk
2. Conversion of all clauses to graphs

**Fix**: Pre-process traces to binary format (e.g., HDF5 or memory-mapped arrays).

### 7.2 InfoNCE Loss - All Negatives
**Location**: `python/proofatlas/ml/training.py:70-83`

```python
def info_nce_loss(scores, labels, temperature=1.0):
    neg_logsumexp = torch.logsumexp(neg_scores, dim=0)  # Over ALL negatives

    losses = -pos_scores + torch.logsumexp(
        torch.stack([pos_scores, neg_logsumexp.expand_as(pos_scores)], dim=0),
        dim=0
    )
```

**Issue**: For each positive, computes logsumexp over ALL negatives. With large batches (1000+ negatives), this is expensive.

**Fix**: Use hard negative mining to sample a subset of negatives.

---

## 8. Priority Recommendations

### High Impact
1. **Add unit equality index** in saturation state - avoids O(n) scan on every forward simplification
2. **Replace string-based subsumption** with structural hashing - eliminates allocation overhead
3. **Lazy clause loading in web** - reduces initial load time

### Medium Impact
4. **Path index for superposition** - reduces unproductive unification attempts
5. **Batch GCN embeddings** - amortizes tensor creation overhead
6. **Pre-process training traces** - eliminates per-sample JSON parsing

### Lower Impact (but easy fixes)
7. **Cache clause shapes** in subsumption checker
8. **Use clause ID for cache keys** instead of string representation
9. **Virtual scrolling** in web clause list for large proofs

---

## Appendix: Complexity Summary

| Component | Current | Optimal | Location |
|-----------|---------|---------|----------|
| Forward simplification | O(n) per selection | O(log n) with index | state.rs:355 |
| Subsumption check | O(n) linear scan | O(1) with hash index | subsumption.rs:82 |
| Substitution normalize | O(n²) worst case | O(n) with persistent DS | substitution.rs:29 |
| Variant detection | O(n) per check | O(1) with shape index | subsumption.rs:176 |
| GCN adjacency | O(n²) dense | O(e) sparse | burn_gcn.rs:426 |
| Proof step navigation | O(steps) replay | O(1) with caching | app.js:215 |
