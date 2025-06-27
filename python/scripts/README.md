# ProofAtlas Scripts

This directory contains utility scripts for working with ProofAtlas.

## Available Scripts

### inspect_proof.py

Interactive proof navigator for terminal environments. Allows step-by-step navigation through proofs with keyboard controls.

```bash
# With both proof and problem files
python inspect_proof.py proof.json problem.json

# With only proof file (extracts problem from initial state)
python inspect_proof.py tests/test_data/proofs/basic_loop/modus_ponens.json

# Skip metadata display
python inspect_proof.py proof.json --no-metadata
```

**Navigation controls:**
- `n` or `→` - Next step
- `p` or `←` - Previous step
- `g` - Go to specific step
- `f` - First step
- `l` - Last step
- `h` or `?` - Show help
- `q` - Quit

**Note:** Requires an interactive terminal environment. Will not work in non-TTY contexts.

### print_proof.py

Non-interactive proof printer. Displays proofs in a readable text format.

```bash
# Print entire proof
python print_proof.py proof.json

# Print only summary
python print_proof.py proof.json --summary

# Print specific step (0-indexed)
python print_proof.py proof.json --step 2
```

**Features:**
- Handles wrapped proof files (with metadata)
- Shows rule applications and generated clauses
- Displays clause counts and statistics
- Works in any environment (no terminal control needed)

## Example Usage

```bash
# Generate proofs using tests
cd src
python -m pytest ../tests/loops/test_basic_loop_save_proofs.py

# View a generated proof summary
cd ..
python scripts/print_proof.py tests/test_data/proofs/basic_loop/chain_resolution.json --summary

# Inspect a proof interactively (requires TTY)
python scripts/inspect_proof.py tests/test_data/proofs/basic_loop/larger_proof.json

# Print specific step of a proof
python scripts/print_proof.py tests/test_data/proofs/basic_loop/simple_contradiction.json --step 1
```

## Proof File Formats

Scripts support two formats:

1. **Raw proof format**: Direct JSON serialization of Proof objects
2. **Wrapped format**: JSON with metadata wrapper containing description, generator, and proof

The scripts automatically detect and handle both formats.