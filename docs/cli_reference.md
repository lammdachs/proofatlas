# CLI Reference

ProofAtlas provides command-line tools for theorem proving, benchmarking, and exporting results.

## proofatlas

The main theorem prover command. Uses Python wrapper with Rust core.

```bash
proofatlas <problem> [options]
proofatlas --list
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config <name>` | - | Solver config from `configs/proofatlas.json` |
| `--timeout <seconds>` | 60 | Time limit for proof search |
| `--max-clauses <n>` | 10000 | Maximum number of clauses before stopping |
| `--literal-selection <n>` | 0 | Literal selection strategy (see below) |
| `--include <dir>` | - | Add TPTP include directory (can be repeated) |
| `--json <file>` | - | Export proof attempt to JSON file |
| `--verbose` | - | Show detailed output |
| `--list` | - | List available configs |

### Literal Selection Strategies

| Value | Name | Description |
|-------|------|-------------|
| 0 | SelectAll | Select all literals (no restriction) |
| 20 | SelectMaximal | Select all maximal literals (using KBO) |
| 21 | UniqueMaximal | Unique maximal if exists, else max-weight negative, else all maximal |
| 22 | NegMaxWeight | Max-weight negative if exists, else all maximal |

### Examples

```bash
# Basic usage
proofatlas .tptp/TPTP-v9.0.0/Problems/PUZ/PUZ001-1.p

# With config
proofatlas problem.p --config time

# With timeout and literal selection
proofatlas problem.p --timeout 30 --literal-selection 21

# Export result to JSON
proofatlas problem.p --json result.json

# List available configs
proofatlas --list
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Proof found (theorem proved) |
| 1 | Error, saturated, timeout, or resource limit |

---

## proofatlas-bench

Benchmark and train theorem provers. Runs as a background daemon.

```bash
proofatlas-bench [options]
```

### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config <name>...` | all | Config(s) to evaluate |
| `--problem-set <name>` | from config | Problem set from `configs/tptp.json` |
| `--rerun` | - | Re-evaluate cached results |
| `--n-jobs <n>` | 1 | Parallel jobs |

### Training Options

| Option | Description |
|--------|-------------|
| `--retrain` | Retrain model even if weights exist |

### Job Management

| Option | Description |
|--------|-------------|
| `--status` | Check job status |
| `--kill` | Stop running job |
| `--list` | List available configs |

### Examples

```bash
# Evaluate all configs
proofatlas-bench

# Evaluate specific config
proofatlas-bench --config time

# Retrain a GCN model
proofatlas-bench --config gcn_mlp --retrain

# Run with parallel jobs
proofatlas-bench --n-jobs 4

# Check status
proofatlas-bench --status

# Kill running job
proofatlas-bench --kill

# List configs
proofatlas-bench --list
```

### Workflow for Learned Selectors

When using a config with ML (e.g., `gcn_mlp`):

1. If TorchScript models exist in `.weights/`, uses them directly
2. Otherwise, use `--retrain` to:
   - Collect traces with heuristic selector
   - Train the model in PyTorch
   - Export to TorchScript (`.pt` file)

### Output Files

| Path | Description |
|------|-------------|
| `.weights/<model>.pt` | TorchScript model (e.g., `gcn_mlp.pt`) |
| `.data/traces/<config>/` | Proof traces for training |
| `.data/runs/<config>/` | Per-problem results (JSON) |
| `.data/bench.log` | Daemon log file |
| `.data/bench_job.json` | Job status file |

---

## Setup Scripts

### setup_tptp.py

Download and configure the TPTP problem library.

```bash
python scripts/setup_tptp.py [--scan]
```

| Option | Description |
|--------|-------------|
| `--scan` | Scan problems and generate metadata |

### setup_vampire.py

Download and install Vampire prover.

```bash
python scripts/setup_vampire.py
```

### setup_spass.py

Download and compile SPASS prover.

```bash
python scripts/setup_spass.py
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TPTP` | TPTP library root directory (used by SPASS) |
| `CUDA_VISIBLE_DEVICES` | GPU device for ML training/inference |
