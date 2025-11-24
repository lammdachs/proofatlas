# ML Data Collection for Clause Selection

Design for collecting proof search traces to train a learned clause selection model.

## Objective

Train a GNN to rank clauses by quality using supervised learning on successful proofs found by age/weight heuristic.

## Data Collection Strategy

### What to Record

At each clause selection step during successful proof search:
- **Selected clause**: The clause chosen as "given clause" (positive example)
- **Alternative clauses**: Other clauses in the queue at selection time (negative examples)
- **Context**: Problem name, selection step number, proof outcome

### Ranking Setup

For each selection step:
```
{
  "problem": "GRP001-1.p",
  "step": 42,
  "selected_clause_id": 123,
  "alternative_clause_ids": [124, 125, 126, ...],
  "selected_clause_graph": {...},
  "alternative_clause_graphs": [{...}, {...}, ...]
}
```

The model will be trained so that:
```
score(selected_clause) > score(alternative_clause)  for all alternatives
```

## Implementation in Saturation Loop

### Modifications to `saturation/state.rs`

Add trace recording mode:
```rust
pub struct SaturationConfig {
    // ... existing fields
    pub record_trace: bool,
    pub trace_output_dir: Option<PathBuf>,
}

pub struct SelectionStep {
    pub step_number: usize,
    pub selected_clause: ClauseId,
    pub alternatives: Vec<ClauseId>,
}

pub struct ProofTrace {
    pub problem_name: String,
    pub successful: bool,
    pub steps: Vec<SelectionStep>,
}
```

### Data Collection Trigger

Only record traces for:
- ✅ Successful proofs (result = Proof)
- ✅ Age/weight heuristic (not random selection)
- ❌ Saturated problems (no training signal)
- ❌ Timeout/resource limit (incomplete data)

### Sampling Strategy

**Full trace**: Record every selection step
- Pro: Maximum training data
- Con: Very large datasets

**Sampled trace**: Record every Nth step or last K steps before proof
- Pro: Manageable dataset size
- Con: Less data per problem

**Initial approach**: Record full trace, add sampling later if needed.

## Data Storage Format

### JSON Lines Format

Each line is one selection step:
```json
{"problem": "GRP001-1.p", "step": 0, "selected": 5, "alternatives": [6,7,8], "selected_graph": {...}, "alternative_graphs": [...]}
{"problem": "GRP001-1.p", "step": 1, "selected": 9, "alternatives": [6,7,8,10], "selected_graph": {...}, "alternative_graphs": [...]}
```

Benefits:
- Streamable (don't need to load all data at once)
- Easy to process line-by-line in Python
- Standard format with good tooling

### Graph Serialization

Reuse existing `ClauseGraphData` serialization:
```json
{
  "edge_index": [[0,0,1],[1,2,2]],
  "node_features": [[1,0,0,...], [0,1,0,...], ...],
  "node_types": [0, 1, 2, ...],
  "num_nodes": 3,
  "num_edges": 3
}
```

## Training Data Pipeline

### Collection Phase
1. Run prover on TPTP problem set with trace recording enabled
2. Save traces to `data/traces/{problem_name}.jsonl`
3. Only save successful proof traces

### Processing Phase
1. Load traces from disk
2. Convert graphs to PyTorch tensors
3. Create ranking pairs/triplets for training
4. Batch for efficient training

### Training Phase
1. Load batches of ranking examples
2. Forward pass through GNN
3. Compute ranking loss
4. Update model weights

## Ranking Loss Options

### Pairwise Ranking Loss
For each selected clause and one alternative:
```
loss = max(0, margin - (score_selected - score_alternative))
```

### Listwise Ranking Loss
Softmax over all clauses, maximize probability of selected:
```
loss = -log(exp(score_selected) / sum(exp(score_alternative)))
```

### Triplet Loss
Anchor=selected, positive=selected, negative=alternative:
```
loss = max(0, score_negative - score_positive + margin)
```

**Recommendation**: Start with listwise (softmax) - most similar to actual selection.

## Expected Data Volume

Assumptions:
- 1000 problems
- Average 100 selection steps per proof
- Average 50 alternative clauses per step
- Average 10 nodes per clause graph

**Storage estimate:**
- 1000 × 100 × 50 = 5M clause graphs
- 5M × 10 nodes × 20 features × 4 bytes = 4 GB

**Mitigation**: Sample alternatives (e.g., top-K by age/weight score + random sample)

## Negative Sampling

To reduce data volume, sample alternatives instead of recording all:
- Top-K by age/weight score (e.g., K=10)
- Random sample of M others (e.g., M=5)
- Total: 15 alternatives per step instead of 50+

This maintains training signal while reducing storage 3-5x.

## Python Interface

```python
from proofatlas import ProofState

# Enable trace recording
state = ProofState()
state.enable_trace_recording("data/traces")

# Run prover
result = state.prove_tptp("problems/GRP001-1.p", timeout=60)

# Trace saved to data/traces/GRP001-1.jsonl
```

## Implementation Plan

1. Add `record_trace` flag to `SaturationConfig`
2. Record selection steps in saturation loop
3. Serialize graphs to JSON
4. Add Python interface for trace recording
5. Test on small problem set
6. Collect full dataset from TPTP
