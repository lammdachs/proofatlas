"""Prover execution for proofatlas-bench.

Handles running ProofAtlas, Vampire, and SPASS on individual problems.
Depends on `proofatlas` (Rust bindings) and `bench_jobs.register_pid`.
"""

import multiprocessing
import os
import queue as queue_mod
import signal
import sys
import threading
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


def _run_proofatlas_inner(problem: Path, base_dir: Path, preset: dict, tptp_root: Path,
                          weights_path: str = None, collect_trace: bool = False,
                          trace_preset: str = None, use_cuda: bool = False,
                          socket_path: str = None) -> BenchResult:
    """Inner function that actually runs ProofAtlas (called in subprocess)."""
    from proofatlas import ProofAtlas

    timeout = preset.get("timeout", 10)
    memory_limit = preset.get("memory_limit")
    literal_selection = preset.get("literal_selection", 21)
    max_iterations = preset.get("max_iterations", 0)
    ml = _get_ml()
    is_learned = ml.is_learned_selector(preset)
    age_weight_ratio = preset.get("age_weight_ratio", 0.5)
    encoder = preset.get("encoder") if is_learned else None
    scorer = preset.get("scorer") if is_learned else None

    # Build ProofAtlas orchestrator with all configuration
    atlas_kwargs = {
        "timeout": float(timeout),
        "literal_selection": literal_selection,
        "memory_limit": memory_limit,
        "include_dir": str(tptp_root),
    }
    if max_iterations > 0:
        atlas_kwargs["max_iterations"] = max_iterations
    if encoder:
        atlas_kwargs["encoder"] = encoder
        atlas_kwargs["scorer"] = scorer
        atlas_kwargs["weights_path"] = weights_path
        atlas_kwargs["use_cuda"] = use_cuda
        if socket_path:
            atlas_kwargs["socket_path"] = socket_path
    else:
        atlas_kwargs["age_weight_ratio"] = float(age_weight_ratio)

    if collect_trace:
        atlas_kwargs["enable_trace"] = True
        if not atlas_kwargs.get("weights_path"):
            atlas_kwargs["weights_path"] = weights_path or str(base_dir / ".weights")

    start = time.time()
    try:
        atlas = ProofAtlas(**atlas_kwargs)
        prover = atlas.prove(str(problem))
    except Exception as e:
        elapsed = time.time() - start
        err_msg = str(e).lower()
        if "timed out" in err_msg or "memory limit" in err_msg:
            return BenchResult(problem=problem.name, status="resource_limit", time_s=elapsed)
        # When using a scoring server, connection/communication failures are
        # resource issues (server died/restarting), not problem-level errors.
        status = "resource_limit" if socket_path else "error"
        return BenchResult(problem=problem.name, status=status, time_s=elapsed)

    elapsed = time.time() - start
    status = prover.status

    # Collect trace for training
    if collect_trace and prover.proof_found and trace_preset:
        try:
            traces_dir = str(base_dir / ".data" / "traces")
            prover.save_trace(traces_dir, trace_preset, problem.name, elapsed)
        except Exception:
            pass

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def _worker_process(problem_str, base_dir_str, preset, tptp_root_str, weights_path,
                    collect_trace, trace_preset, result_queue, use_cuda=False,
                    socket_path=None):
    """Worker function that runs in subprocess and sends result via queue."""
    # Ensure forked workers never initialize CUDA (the scoring server handles GPU).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Reset signal handlers inherited from parent daemon process.
    # Without this, if the worker is killed (e.g., timeout), it would run
    # the parent's signal handler which deletes the job file!
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)

    try:
        result = _run_proofatlas_inner(
            Path(problem_str), Path(base_dir_str), preset, Path(tptp_root_str),
            weights_path, collect_trace, trace_preset, use_cuda,
            socket_path=socket_path,
        )
        result_queue.put((result.status, result.time_s))
    except Exception as e:
        result_queue.put(("error", 0))


def run_proofatlas(problem: Path, base_dir: Path, preset: dict, tptp_root: Path,
                   weights_path: str = None, collect_trace: bool = False,
                   trace_preset: str = None,
                   use_cuda: bool = False, socket_path: str = None) -> BenchResult:
    """Run ProofAtlas on a problem in a subprocess.

    Uses multiprocessing to isolate crashes (e.g., stack overflow on deeply
    nested terms) so they don't take down the entire benchmark process.
    """
    import multiprocessing

    timeout = preset.get("timeout", 10)
    # When using a shared scoring server, workers may queue behind a mutex
    # while the server processes other workers' requests.  With N workers on
    # one GPU, worst-case queue latency is ~N * embed_time.  Give plenty of
    # headroom so the prover's internal timeout is the one that fires, not
    # the process watchdog.
    process_timeout = max(timeout * 3, timeout + 60)

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_worker_process,
        args=(str(problem), str(base_dir), preset, str(tptp_root),
              weights_path, collect_trace, trace_preset, result_queue, use_cuda),
        kwargs={"socket_path": socket_path},
    )

    start = time.time()
    proc.start()
    proc.join(timeout=process_timeout)
    elapsed = time.time() - start

    try:
        if proc.is_alive():
            # Process hung - kill it
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return BenchResult(problem=problem.name, status="timeout", time_s=elapsed)

        # Check result queue first — the child may exit with non-zero code due to
        # cleanup issues (e.g., libtorch thread destruction when forked from a thread)
        # but still have produced a valid result.
        try:
            status, elapsed_inner = result_queue.get_nowait()
            return BenchResult(problem=problem.name, status=status, time_s=elapsed_inner)
        except Exception:
            pass

        if proc.exitcode != 0:
            # Process crashed before producing a result (e.g., stack overflow gives exit code 134)
            print(f"Worker crashed for {problem.name}: exit code {proc.exitcode} ({elapsed:.2f}s)")
            sys.stdout.flush()
            return BenchResult(problem=problem.name, status="error", time_s=elapsed)

        return BenchResult(problem=problem.name, status="error", time_s=elapsed)
    finally:
        # Close multiprocessing resources to prevent FD leak.
        # Without this, each call leaks pipe FDs from the Queue and Process,
        # and after ~340 problems the daemon hits the FD limit (ulimit -n).
        try:
            proc.close()
        except ValueError:
            pass  # Process still running — will be reaped by GC
        result_queue.close()
        result_queue.join_thread()


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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        register_pid(base_dir, proc.pid)
        try:
            # timeout=0 means no time limit, use None for communicate
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

    # Parse Vampire output
    if "Refutation found" in output or "Termination reason: Refutation" in output:
        status = "proof"
    elif "Termination reason: Satisfiable" in output:
        status = "saturated"
    elif "Termination reason: Time limit" in output or elapsed >= timeout:
        status = "timeout"
    elif "Termination reason: Memory limit" in output:
        status = "timeout"  # Memory limit treated as resource limit
    elif "Termination reason: Activation limit" in output:
        status = "timeout"  # Activation limit treated as resource limit
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

    # SPASS requires TPTP format with -TPTP flag
    cmd = [
        str(binary),
        "-TPTP",
        f"-TimeLimit={timeout}",
        f"-Select={selection}",
    ]

    if memory is not None:
        cmd.append(f"-Memory={memory}")

    if loops is not None:
        cmd.append(f"-Loops={loops}")

    cmd.append(str(problem))

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "TPTP": str(tptp_root)},
        )
        register_pid(base_dir, proc.pid)
        try:
            # timeout=0 means no time limit, use None for communicate
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

    # Parse SPASS output
    # SPASS says "SPASS beiseite: Proof found." for proofs
    if "Proof found" in output:
        status = "proof"
    elif "Completion found" in output:
        status = "saturated"
    elif "Maximal number of loops exceeded" in output:
        status = "timeout"  # Loop limit treated as resource limit
    elif elapsed >= timeout or "SPASS broke down" in output:
        status = "timeout"
    else:
        status = "error"

    return BenchResult(problem=problem.name, status=status, time_s=elapsed)


def run_single_problem(args, socket_path=None):
    """Worker function for execution. Handles caching and timeout→resource_limit remapping."""
    problem, base_dir, prover, preset, tptp_root, weights_path, collect_trace, trace_preset, binary, preset_name, rerun, use_cuda = args

    # Import here to avoid circular imports (bench.py defines these)
    from bench import load_run_result, save_run_result

    try:
        # Check if already evaluated (skip unless --rerun)
        existing = load_run_result(base_dir, prover, preset_name, problem)
        if existing and not rerun:
            return ("skip", existing)

        if prover == "proofatlas":
            result = run_proofatlas(
                problem, base_dir, preset, tptp_root,
                weights_path=weights_path, collect_trace=collect_trace,
                trace_preset=trace_preset,
                use_cuda=use_cuda, socket_path=socket_path,
            )
        elif prover == "vampire":
            result = run_vampire(problem, base_dir, preset, binary, tptp_root)
        elif prover == "spass":
            result = run_spass(problem, base_dir, preset, binary, tptp_root)
        else:
            result = BenchResult(problem=problem.name, status="error", time_s=0)

        # Normalize timeout → resource_limit
        if result.status == "timeout":
            result.status = "resource_limit"

        # Save individual result to .data/runs/
        save_run_result(base_dir, prover, preset_name, result)
        return ("run", result)
    except Exception as e:
        import traceback
        print(f"ERROR in run_single_problem({problem.name}): {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return ("error", BenchResult(problem=problem.name, status="error", time_s=0))


# ===========================================================================
# Persistent ProofAtlas worker pool — reuses ProofAtlas across problems
# ===========================================================================


def build_atlas_kwargs(preset: dict, tptp_root: Path, weights_path: str = None,
                       use_cuda: bool = False, socket_path: str = None,
                       collect_trace: bool = False) -> dict:
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
        if socket_path:
            kwargs["socket_path"] = socket_path
    else:
        kwargs["age_weight_ratio"] = float(preset.get("age_weight_ratio", 0.5))

    # Enable trace embedding via MiniLM backend
    if collect_trace:
        kwargs["enable_trace"] = True
        # Ensure weights_path is set so MiniLM model can be found
        if not kwargs.get("weights_path"):
            kwargs["weights_path"] = weights_path or str(Path(__file__).parent.parent / ".weights")

    return kwargs


def _atlas_worker_loop(task_queue, result_queue, atlas_kwargs, base_dir_str,
                       collect_trace, trace_preset):
    """Persistent worker: create ProofAtlas once, solve problems in a loop."""
    # Block CUDA unless the atlas needs in-process GPU (use_cuda without socket_path).
    # When using a scoring server (socket_path set), the server handles GPU;
    # workers must not initialize CUDA to avoid fork-inherited context corruption.
    if not atlas_kwargs.get("use_cuda") or atlas_kwargs.get("socket_path"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        signal.signal(signal.SIGQUIT, signal.SIG_DFL)
    except (AttributeError, OSError):
        pass

    from proofatlas import ProofAtlas
    atlas = ProofAtlas(**atlas_kwargs)
    base_dir = Path(base_dir_str)

    while True:
        task = task_queue.get()
        if task is None:
            break

        problem_str = task
        problem = Path(problem_str)
        start = time.time()
        try:
            prover = atlas.prove(str(problem))
            elapsed = time.time() - start
            status = prover.status

            if collect_trace and prover.proof_found and trace_preset:
                try:
                    traces_dir = str(base_dir / ".data" / "traces")
                    prover.save_trace(traces_dir, trace_preset, problem.name, elapsed)
                except Exception:
                    pass

            result_queue.put((problem.name, status, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            err_msg = str(e).lower()
            if "timed out" in err_msg or "memory limit" in err_msg:
                result_queue.put((problem.name, "resource_limit", elapsed))
            else:
                result_queue.put((problem.name, "error", elapsed))


class AtlasWorker:
    """A persistent worker process holding one ProofAtlas instance.

    Thread-safe: multiple threads may call submit() concurrently
    (a lock serializes access to the underlying queues).
    """

    def __init__(self, atlas_kwargs, base_dir, collect_trace, trace_preset):
        self.atlas_kwargs = atlas_kwargs
        self.base_dir_str = str(base_dir)
        self.collect_trace = collect_trace
        self.trace_preset = trace_preset
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self._lock = threading.Lock()
        self.proc = None

    def start(self):
        self.proc = multiprocessing.Process(
            target=_atlas_worker_loop,
            args=(self.task_queue, self.result_queue, self.atlas_kwargs,
                  self.base_dir_str, self.collect_trace, self.trace_preset),
        )
        self.proc.start()

    def submit(self, problem_path, timeout):
        """Submit a problem and block until result. Returns BenchResult.

        Thread-safe — only one caller at a time per worker.
        """
        with self._lock:
            self.task_queue.put(str(problem_path))
            deadline = time.time() + timeout

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    name, status, elapsed = self.result_queue.get(
                        timeout=min(remaining, 2.0),
                    )
                    return BenchResult(problem=name, status=status, time_s=elapsed)
                except queue_mod.Empty:
                    if not self.proc.is_alive():
                        break

            # Timeout or crash — restart worker for next problem
            problem_name = Path(problem_path).name
            elapsed = time.time() - (deadline - timeout)
            if self.proc.is_alive():
                self.proc.kill()
            self.proc.join(timeout=5)
            self._drain_queues()
            self.start()
            return BenchResult(problem=problem_name, status="resource_limit", time_s=elapsed)

    def _drain_queues(self):
        for q in (self.task_queue, self.result_queue):
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass

    def shutdown(self):
        if self.proc and self.proc.is_alive():
            self.task_queue.put(None)
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
        for q in (self.task_queue, self.result_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass


class ProofAtlasPool:
    """Pool of persistent ProofAtlas workers.

    Each worker creates a ProofAtlas once and reuses it across problems,
    avoiding per-problem model loading overhead. Workers are automatically
    restarted on crash or timeout.
    """

    def __init__(self, n_workers, preset, base_dir, tptp_root,
                 weights_path=None, use_cuda=False, socket_paths=None,
                 collect_trace=False, trace_preset=None):
        timeout = preset.get("timeout", 10)
        self.process_timeout = max(timeout * 3, timeout + 60)

        self.workers = []
        for i in range(n_workers):
            sp = socket_paths[i % len(socket_paths)] if socket_paths else None
            kwargs = build_atlas_kwargs(preset, tptp_root, weights_path, use_cuda, sp,
                                        collect_trace=collect_trace)
            worker = AtlasWorker(kwargs, base_dir, collect_trace, trace_preset)
            self.workers.append(worker)

    def start(self):
        for w in self.workers:
            w.start()

    def shutdown(self):
        for w in self.workers:
            w.shutdown()
