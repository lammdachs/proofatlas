# Selective Testing in CI/CD

ProofAtlas uses selective testing to run only relevant tests based on changed files, reducing CI time by ~70%.

## Our Approach

**Tiered testing strategy:**
- **Tier 1**: Critical smoke tests (always run, < 30s)
- **Tier 2**: Component tests (selective, based on changed files)
- **Tier 3**: Full suite (main branch + weekly schedule)

## Implementation

### Tier 1: Critical Tests
Always run regardless of changes:
- Rust compile check (`cargo check`)
- End-to-end proof tests (ensures prover works)
- Basic Python import test

**Runtime:** < 30 seconds

### Tier 2: Selective Tests
Run based on changed file paths:
- **Rust core changes** (`rust/src/core/`, `rust/src/inference/`) → All Rust tests
- **Rust ML changes** (`rust/src/ml/`) → ML-specific Rust tests
- **Python ML changes** (`python/proofatlas/ml/`, `python/tests/ml/`) → Python ML tests
- **Docs-only changes** (`docs/`, `**/*.md`) → Skip all tests

**Runtime:** 2-4 minutes (typical PR)

### Tier 3: Full Suite
Run all tests:
- On `main` branch (every push)
- Weekly schedule (Sunday 00:00 UTC)
- Before releases (manual trigger)

**Runtime:** 8-12 minutes

## Tools Used

- **dorny/paths-filter**: Detect changed file paths
- **GitHub Actions conditional jobs**: Skip irrelevant tests
- **Caching**: Cargo build artifacts, pip packages
- **Parallel execution**: `cargo nextest`, `pytest -n auto`

## Performance Impact

**Before:**
- Runtime: ~8-12 minutes per PR
- Cost: ~400 min/week

**After:**
- Runtime: ~2-4 minutes (typical PR)
- Cost: ~120 min/week

**Savings: 70% reduction**

## Trade-offs

**Benefits:**
- Faster feedback on PRs
- Lower GitHub Actions costs
- Clear which tests relate to changes

**Risks mitigated:**
- Integration bugs: Full suite runs on main + weekly
- False skips: Conservative path filters include dependencies
- Soundness bugs: Critical tests always run

## Example Workflows

See `.github/workflows/selective-tests.yml.example` for implementation examples.
