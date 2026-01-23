# CLI Reference

ProofAtlas provides several command-line tools for theorem proving, benchmarking, and exporting results.

## prove (Rust Binary)

The core theorem prover binary.

```bash
./target/release/prove <tptp_file> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--timeout <seconds>` | 60 | Time limit for proof search |
| `--max-clauses <n>` | 10000 | Maximum number of clauses before stopping |
| `--literal-selection <n>` | 0 | Literal selection strategy (see below) |
| `--include <dir>` | - | Add TPTP include directory (can be repeated) |
| `--age-weight <ratio>` | 0.167 | Age probability for clause selection |
| `--verbose` | - | Show detailed progress and proof steps |

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
./target/release/prove .tptp/TPTP-v9.0.0/Problems/PUZ/PUZ001-1.p

# With timeout and literal selection
./target/release/prove problem.p --timeout 30 --literal-selection 21

# Verbose output with include directory
./target/release/prove problem.p --include .tptp/TPTP-v9.0.0 --verbose
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Proof found (theorem proved) |
| 1 | Error, saturated, timeout, or resource limit |

---

## proofatlas-prove (Python)

Python wrapper for theorem proving with preset support.

```bash
proofatlas-prove <problem> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preset <name>` | time_sel21 | Solver preset from `configs/proofatlas.json` |
| `--verbose` | - | Show detailed output |

### Examples

```bash
# Use default preset
proofatlas-prove .tptp/TPTP-v9.0.0/Problems/GRP/GRP001-1.p

# Use specific preset
proofatlas-prove problem.p --preset time_sel21 --verbose
```

---

## proofatlas-bench

Benchmark and train theorem provers. Runs as a background daemon.

```bash
proofatlas-bench [options]
```

### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prover <name>` | all | Prover to run: `proofatlas`, `vampire`, `spass` |
| `--preset <name>...` | all | Preset(s) to evaluate |
| `--problem-set <name>` | from config | Problem set from `configs/tptp.json` |
| `--base-only` | - | Skip learned selectors (ML models) |
| `--rerun` | - | Re-evaluate cached results |
| `--n-jobs <n>` | 1 | Parallel jobs |

### Training Options

| Option | Description |
|--------|-------------|
| `--force-train` | Retrain even if weights exist |
| `--trace-preset <name>` | Preset name for trace collection |

### Job Management

| Option | Description |
|--------|-------------|
| `--track` | Monitor job progress (blocking) |
| `--status` | Check job status |
| `--kill` | Stop running job |

### Examples

```bash
# Evaluate all provers with all presets
proofatlas-bench

# Evaluate specific prover and preset
proofatlas-bench --prover proofatlas --preset time_sel21

# Train a GCN model
proofatlas-bench --prover proofatlas --preset gcn --force-train

# Run with parallel jobs and monitor
proofatlas-bench --prover vampire --n-jobs 4 --track

# Check status
proofatlas-bench --status

# Kill running job
proofatlas-bench --kill
```

### Workflow for Learned Selectors

When using a preset with ML (e.g., `gcn`, `sentence`):

1. If TorchScript models exist in `.weights/`, uses them directly
2. Otherwise, you need to:
   - Collect traces with age_weight selector
   - Train the model in PyTorch
   - Export to TorchScript (`.pt` file)

### Output Files

| Path | Description |
|------|-------------|
| `.weights/gcn_model.pt` | GCN TorchScript model |
| `.weights/sentence_encoder.pt` | Sentence transformer model |
| `.data/traces/<preset>/` | Proof traces for training |
| `.data/runs/<prover>/<preset>/` | Per-problem results (JSON) |
| `.data/bench.log` | Daemon log file |
| `.data/bench_job.json` | Job status file |

---

## proofatlas-export

Export benchmark and training results for web display.

```bash
proofatlas-export [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--benchmarks` | - | Export benchmarks only |
| `--training` | - | Export training only |
| `--problem-set <name>` | from config | Limit to problems in this set |
| `--prover <name>` | all | Include only this prover |
| `--preset <name>` | all | Include only this preset |
| `--base-only` | - | Skip learned selectors |
| `--output-dir <path>` | web/data/ | Output directory |
| `--commit` | - | Commit exported files to git |

### Examples

```bash
# Export everything (uses default problem set)
proofatlas-export

# Export benchmarks only
proofatlas-export --benchmarks

# Export for specific problem set
proofatlas-export --problem-set test

# Export and commit
proofatlas-export --commit
```

### Output Files

| Path | Description |
|------|-------------|
| `web/data/benchmarks.json` | Benchmark results per prover/preset |
| `web/data/training.json` | Training runs and available weights |

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
| `CUDA_VISIBLE_DEVICES` | GPU device for ML training |
