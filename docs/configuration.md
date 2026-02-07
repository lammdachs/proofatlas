# Configuration Guide

ProofAtlas uses JSON configuration files in the `configs/` directory. This guide documents all available options.

## Configuration Files

| File | Purpose |
|------|---------|
| `proofatlas.json` | ProofAtlas prover presets |
| `vampire.json` | Vampire prover presets |
| `spass.json` | SPASS prover presets |
| `tptp.json` | TPTP library paths and problem sets |
| `embeddings.json` | Clause embedding architectures |
| `scorers.json` | Clause scorer architectures |
| `training.json` | ML training hyperparameters |

---

## proofatlas.json

Configures the ProofAtlas prover and its presets.

### Structure

```json
{
  "paths": {
    "binary": "rust/target/release/prove"
  },
  "defaults": {
    "preset": "time"
  },
  "presets": {
    "preset_name": { ... }
  }
}
```

### Preset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `description` | string | - | Human-readable description |
| `timeout` | int | 10 | Time limit in seconds |
| `max_clauses` | int | 0 | Clause limit (0 = unlimited) |
| `literal_selection` | int | 0 | Literal selection strategy (0, 20, 21) |
| `age_weight_ratio` | float | 0.167 | Probability of selecting oldest clause |
| `model` | string | - | ML model name (gcn, mlp, gat) |
| `training` | string | - | Training config name |
| `traces` | string | - | Trace preset for training data |

### Example Presets

```json
{
  "time": {
    "description": "10s timeout",
    "timeout": 10,
    "literal_selection": 21,
    "age_weight_ratio": 0.167
  },
  "gcn_mlp": {
    "description": "GCN + MLP",
    "timeout": 10,
    "literal_selection": 21,
    "encoder": "gcn",
    "scorer": "mlp",
    "traces": "time"
  }
}
```

---

## vampire.json

Configures the Vampire prover.

### Preset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `description` | string | - | Human-readable description |
| `time_limit` | int | 10 | Time limit in seconds |
| `selection` | int | 21 | Literal selection (0, 20, 21) |
| `avatar` | string | "off" | AVATAR splitting ("on"/"off") |
| `memory_limit` | int | - | Memory limit in MB |
| `activation_limit` | int | - | Maximum clause activations |

### Example

```json
{
  "time_sel21": {
    "description": "10s with selection 21",
    "time_limit": 10,
    "selection": 21,
    "avatar": "off"
  },
  "age_weight_sel21": {
    "description": "Age-weight baseline (512 activations) with selection 21",
    "time_limit": 600,
    "activation_limit": 512,
    "selection": 21,
    "avatar": "off"
  }
}
```

---

## spass.json

Configures the SPASS prover.

### Preset Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `description` | string | - | Human-readable description |
| `TimeLimit` | int | 10 | Time limit in seconds |
| `Select` | int | 1 | Selection strategy (0, 1, 2) |
| `Memory` | int | - | Memory limit |
| `Loops` | int | - | Maximum loop iterations |

### Selection Mapping

SPASS uses different selection numbers than Vampire:

| SPASS | Vampire | Description |
|-------|---------|-------------|
| 0 | 20 | All maximal literals |
| 1 | 21 | Prefer negative literals |
| 2 | 22 | Negative max-weight |

Note: SPASS does not support Vampire's sel0 (select all).

---

## tptp.json

Configures TPTP library paths and problem sets.

### Structure

```json
{
  "version": "9.0.0",
  "source": "https://tptp.org/TPTP/Archive/TPTP-v9.0.0.tgz",
  "paths": {
    "root": ".tptp/TPTP-v9.0.0",
    "problems": ".tptp/TPTP-v9.0.0/Problems",
    "axioms": ".tptp/TPTP-v9.0.0/Axioms"
  },
  "defaults": {
    "problem_set": "unsat100"
  },
  "problem_sets": {
    "set_name": { ... }
  }
}
```

### Problem Set Filters

| Filter | Type | Description |
|--------|------|-------------|
| `description` | string | Human-readable description |
| `status` | list | Problem status: `["unsatisfiable"]`, `["satisfiable"]` |
| `format` | list | Input format: `["cnf"]`, `["fof"]`, `["cnf", "fof"]` |
| `domains` | list | Include only these domains: `["GRP", "PUZ"]` |
| `exclude_domains` | list | Exclude these domains: `["CSR", "HWV", "SWV"]` |
| `max_rating` | float | Maximum TPTP difficulty rating (0.0-1.0) |
| `max_clauses` | int | Maximum number of clauses |
| `max_term_depth` | int | Maximum term nesting depth |
| `max_clause_size` | int | Maximum literals per clause |
| `has_equality` | bool | `true` = only equality problems |
| `is_unit_only` | bool | `true` = only unit clause problems |
| `problems` | list | Explicit list of problem names |
| `problems_file` | string | Path to file with problem names |

### Example Problem Sets

```json
{
  "unsat100": {
    "description": "Unsatisfiable problems with at most 100 clauses",
    "status": ["unsatisfiable"],
    "format": ["cnf", "fof"],
    "max_clauses": 100,
    "max_term_depth": 8,
    "exclude_domains": ["CSR", "HWV", "SWV"]
  },
  "unit_equality": {
    "description": "Unit equality problems",
    "status": ["unsatisfiable"],
    "format": ["cnf"],
    "has_equality": true,
    "is_unit_only": true,
    "max_clauses": 500,
    "domains": ["GRP", "RNG", "LAT"]
  },
  "test": {
    "description": "Small test set",
    "status": ["unsatisfiable"],
    "format": ["cnf"],
    "max_rating": 0.3,
    "max_clauses": 50,
    "domains": ["PUZ", "SYN"]
  }
}
```

---

## embeddings.json

Defines clause embedding architectures that convert clause graphs to vector representations.

### Structure

```json
{
  "input_dim": 8,
  "architectures": {
    "embedding_name": { ... }
  }
}
```

### Embedding Types

| Type | Description |
|------|-------------|
| `gcn` | Graph Convolutional Network |
| `sentence_transformer` | Sentence transformer for text-based embeddings |
| `none` | Use raw clause features directly |

### GCN Options

| Option | Type | Description |
|--------|------|-------------|
| `hidden_dim` | int | Hidden layer dimension |
| `num_layers` | int | Number of GCN layers |
| `dropout` | float | Dropout probability |

### Example

```json
{
  "input_dim": 8,
  "architectures": {
    "gcn": {
      "type": "gcn",
      "hidden_dim": 256,
      "num_layers": 6,
      "dropout": 0.1
    },
    "sentence_transformer": {
      "type": "sentence_transformer"
    },
    "none": {
      "type": "none"
    }
  }
}
```

---

## scorers.json

Defines scorer architectures that rank clauses from their embeddings.

### Structure

```json
{
  "architectures": {
    "scorer_name": { ... }
  }
}
```

### Scorer Types

| Type | Description |
|------|-------------|
| `mlp` | Multi-layer perceptron |
| `attention` | Attention-based scoring |
| `transformer` | Transformer encoder |

### Example

```json
{
  "architectures": {
    "mlp": {
      "type": "mlp",
      "hidden_dim": 64,
      "num_layers": 2,
      "dropout": 0.1
    },
    "attention": {
      "type": "attention",
      "num_heads": 4
    },
    "transformer": {
      "type": "transformer",
      "hidden_dim": 64,
      "num_layers": 2,
      "num_heads": 4,
      "dropout": 0.1
    }
  }
}
```

---

## training.json

Defines ML training configurations.

### Structure

```json
{
  "defaults": { ... },
  "configs": {
    "config_name": { ... }
  }
}
```

### Training Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_size` | int | 32 | Training batch size |
| `max_epochs` | int | 100 | Maximum training epochs |
| `val_ratio` | float | 0.0 | Validation set ratio |
| `margin` | float | 0.1 | Margin for ranking loss |
| `optimizer` | string | "adamw" | Optimizer: `adamw`, `adam`, `sgd` |
| `learning_rate` | float | 0.001 | Learning rate |
| `weight_decay` | float | 1e-5 | L2 regularization |
| `betas` | list | [0.9, 0.999] | Adam beta parameters |
| `gradient_clip` | float | 1.0 | Gradient clipping norm |
| `momentum` | float | 0.9 | SGD momentum |

### Example Configurations

```json
{
  "defaults": {
    "batch_size": 32,
    "max_epochs": 100,
    "optimizer": "adamw",
    "learning_rate": 0.001
  },
  "configs": {
    "standard": {
      "description": "Standard training"
    },
    "with_val": {
      "description": "With validation split",
      "val_ratio": 0.2
    },
    "quick": {
      "description": "Quick training",
      "max_epochs": 20
    }
  }
}
```

---

## Complete Example

Here's how the configuration files work together:

1. **tptp.json** defines what problems to evaluate
2. **proofatlas.json** defines how to run the prover
3. **embeddings.json** defines clause embedding architecture
4. **scorers.json** defines clause scorer architecture
5. **training.json** defines how to train models

Running:
```bash
proofatlas-bench --config gcn --problem-set unit_equality
```

This will:
1. Load problems matching `unit_equality` filters from `tptp.json`
2. Use the `gcn` config from `proofatlas.json`
3. Use embedding and scorer from `embeddings.json` / `scorers.json`
4. Apply `standard` training config from `training.json`
5. Evaluate the trained model on the problem set
