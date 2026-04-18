"""Prover execution for proofatlas-bench.

Handles running ProofAtlas, Vampire, and SPASS on individual problems.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bench_jobs import register_pid

# Lazy imports for ML functionality (avoid loading PyTorch for --list)
_ml_module = None


def _get_ml():
    """Lazily import proofatlas.ml to avoid slow startup for simple commands."""
    global _ml_module
    if _ml_module is None:
        from proofatlas import ml as _ml
        _ml_module = _ml
    return _ml_module


@dataclass
class BenchResult:
    problem: str
    status: str  # "proof", "saturated", "resource_limit", "error"
    time_s: float
    iterations: int = 0
    clause_count: int = 0
    clause_bytes: int = 0


# ===========================================================================
# ProofAtlas pool (shared-backend architecture)
# ===========================================================================


def build_atlas_kwargs(preset: dict, tptp_root: Path, weights_path: str = None,
                       use_cuda: bool = False,
                       collect_trace: bool = False,
                       preset_name: str = None) -> dict:
    """Build ProofAtlas constructor kwargs from a preset config."""
    ml = _get_ml()
    is_learned = ml.is_learned_selector(preset)

    kwargs = {
        "timeout": float(preset.get("timeout", 10)),
        "literal_selection": preset.get("literal_selection", 21),
        "memory_limit": preset.get("memory_limit"),
        "include_dir": str(tptp_root),
    }

    max_iterations = preset.get("max_iterations", 0)
    if max_iterations > 0:
        kwargs["max_iterations"] = max_iterations

    encoder = preset.get("encoder") if is_learned else None
    if encoder:
        kwargs["encoder"] = encoder
        kwargs["scorer"] = preset["scorer"]
        kwargs["weights_path"] = weights_path
        kwargs["use_cuda"] = use_cuda
        if preset_name:
            kwargs["model_name"] = preset_name
        temperature = preset.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
    else:
        kwargs["age_weight_ratio"] = float(preset.get("age_weight_ratio", 0.5))

    if collect_trace:
        kwargs["enable_trace"] = True
        if not kwargs.get("weights_path"):
            kwargs["weights_path"] = weights_path or str(Path(__file__).parent.parent / ".weights")

    return kwargs


class ProofAtlasPool:
    """Pool of prover threads sharing a single Backend.

    Uses ProofAtlas.start_workers() for shared-backend parallel proving.
    One Backend thread handles GPU; prover threads do CPU work.
    """

    def __init__(self, n_workers, preset, base_dir, tptp_root,
                 weights_path=None, use_cuda=False,
                 collect_trace=False, trace_preset=None,
                 fallback_configs=None, preset_name=None):
        from proofatlas import ProofAtlas

        _ = fallback_configs  # Deprecated: relabeling is now offline
        timeout = preset.get("timeout", 10)
        self.process_timeout = max(timeout * 3, timeout + 60)
        self.collect_trace = collect_trace
        self.trace_preset = trace_preset
        self.base_dir = Path(base_dir)

        kwargs = build_atlas_kwargs(preset, tptp_root, weights_path, use_cuda,
                                    collect_trace=collect_trace,
                                    preset_name=preset_name)
        self.atlas = ProofAtlas(**kwargs)
        traces_dir = str(base_dir / ".data" / "traces") if collect_trace else None
        self.atlas.start_workers(
            n_workers, collect_traces=collect_trace,
            traces_dir=traces_dir, trace_preset=trace_preset,
        )

    def start(self):
        pass  # Workers start in __init__

    def submit(self, problem_path):
        """Submit a problem for proving."""
        self.atlas.submit(str(problem_path))

    def collect(self, timeout=None):
        """Collect one result. Returns BenchResult or None on timeout."""
        t = timeout or self.process_timeout
        result = self.atlas.collect(timeout=t)
        if result is None:
            return None

        status = result.status
        if status == "timeout":
            status = "resource_limit"

        return BenchResult(
            problem=result.problem, status=status, time_s=result.time_s,
            iterations=result.iterations, clause_count=result.num_clauses,
            clause_bytes=result.clause_bytes,
        )

    def shutdown(self):
        self.atlas.shutdown_workers()


# ===========================================================================
# External provers (Vampire, SPASS)
# ===========================================================================


def run_vampire(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run Vampire on a problem."""
    import subprocess

    timeout = preset.get("time_limit", 10)
    selection = preset.get("selection", 21)
    avatar = preset.get("avatar", "off")
    memory_limit = preset.get("memory_limit")
    activation_limit = preset.get("activation_limit")

    cmd = [
        str(binary),
        "--include", str(tptp_root),
        "--time_limit", str(timeout),
        "--selection", str(selection),
        "--avatar", avatar,
    ]

    if memory_limit is not None:
        cmd.extend(["--memory_limit", str(memory_limit)])
    if activation_limit is not None:
        cmd.extend(["--activation_limit", str(activation_limit)])
    cmd.append(str(problem))

    start = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        register_pid(base_dir, proc.pid)
        try:
            proc_timeout = None if timeout == 0 else timeout + 5
            stdout, stderr = proc.communicate(timeout=proc_timeout)
            output = stdout + stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return BenchResult(problem=problem.name, status="timeout", time_s=timeout)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    if "Refutation found" in output or "Termination reason: Refutation" in output:
        status = "proof"
    elif "Termination reason: Satisfiable" in output:
        status = "saturated"
    elif "Termination reason: Time limit" in output or elapsed >= timeout:
        status = "timeout"
    elif "Termination reason: Memory limit" in output:
        status = "timeout"
    elif "Termination reason: Activation limit" in output:
        status = "timeout"
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def run_spass(problem: Path, base_dir: Path, preset: dict, binary: Path, tptp_root: Path) -> BenchResult:
    """Run SPASS on a problem."""
    import subprocess

    timeout = preset.get("TimeLimit", 10)
    selection = preset.get("Select", 1)
    memory = preset.get("Memory")
    loops = preset.get("Loops")

    cmd = [str(binary), "-TPTP", f"-TimeLimit={timeout}", f"-Select={selection}"]
    if memory is not None:
        cmd.append(f"-Memory={memory}")
    if loops is not None:
        cmd.append(f"-Loops={loops}")
    cmd.append(str(problem))

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "TPTP": str(tptp_root)},
        )
        register_pid(base_dir, proc.pid)
        try:
            proc_timeout = None if timeout == 0 else timeout + 5
            stdout, stderr = proc.communicate(timeout=proc_timeout)
            output = stdout + stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return BenchResult(problem=problem.name, status="timeout", time_s=timeout)
    except Exception:
        return BenchResult(problem=problem.name, status="error", time_s=time.time() - start)

    elapsed = time.time() - start

    if "Proof found" in output:
        status = "proof"
    elif "Completion found" in output:
        status = "saturated"
    elif "Maximal number of loops exceeded" in output:
        status = "timeout"
    elif elapsed >= timeout or "SPASS broke down" in output:
        status = "timeout"
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)
