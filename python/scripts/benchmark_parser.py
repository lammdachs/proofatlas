#!/usr/bin/env python3
"""
Benchmark Rust parser against Vampire's clausify mode on TPTP problems.

This script compares the performance of the Rust TPTP parser with Vampire's
clausify mode. The Rust parser only supports CNF and FOF formats, so failures
on TFF/THF files are expected and tracked separately.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import sys
import signal
from contextlib import contextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import proofatlas_rust
    # Verify parser module is available
    if not hasattr(proofatlas_rust, 'parser'):
        print("Error: proofatlas_rust.parser module not found.")
        print("Please rebuild with: cd rust && maturin develop")
        sys.exit(1)
except ImportError:
    print("Error: proofatlas_rust module not found.")
    print("Please build it with: cd rust && maturin develop")
    sys.exit(1)


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class BenchmarkResult:
    """Container for benchmark results of a single file."""
    def __init__(self, filename: str, file_size: int):
        self.filename = filename
        self.file_size = file_size
        self.rust_time: Optional[float] = None
        self.vampire_time: Optional[float] = None
        self.rust_clauses: Optional[int] = None
        self.vampire_clauses: Optional[int] = None
        self.rust_error: Optional[str] = None
        self.vampire_error: Optional[str] = None
    
    @property
    def both_succeeded(self) -> bool:
        return self.rust_error is None and self.vampire_error is None
    
    @property
    def speedup(self) -> Optional[float]:
        if self.both_succeeded and self.vampire_time > 0:
            return self.vampire_time / self.rust_time
        return None


def get_file_format(filepath: Path) -> str:
    """Detect TPTP file format from content."""
    try:
        with open(filepath, 'r') as f:
            content = f.read(4000)  # Read first 4KB for better detection
            # Check for formats in order of precedence
            if 'thf(' in content:
                return 'THF'
            elif 'tff(' in content:
                return 'TFF'
            elif 'fof(' in content:
                return 'FOF'
            elif 'cnf(' in content:
                return 'CNF'
            elif 'include(' in content:
                return 'Include-only'
            else:
                return 'Unknown'
    except Exception as e:
        return f'Error: {str(e)}'


def benchmark_rust_parser(filepath: str, include_path: str, timeout_seconds: int = 30) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Benchmark Rust parser on a file with timeout using subprocess."""
    # Create a simple Python script to run the parser
    # Escape paths for safety
    filepath_escaped = filepath.replace("'", "\\'")
    include_path_escaped = include_path.replace("'", "\\'")
    parent_path = str(Path(__file__).parent.parent).replace("'", "\\'")
    
    script = f"""
import sys
import time
sys.path.insert(0, '{parent_path}')

try:
    import proofatlas_rust
    start = time.perf_counter()
    problem = proofatlas_rust.parser.parse_file('{filepath_escaped}', '{include_path_escaped}')
    end = time.perf_counter()
    print(f"SUCCESS|{{(end-start)*1000}}|{{len(problem)}}")
except Exception as e:
    print(f"ERROR|{{type(e).__name__}}: {{str(e)}}")
    sys.exit(1)
"""
    
    try:
        # Run parser in subprocess with timeout
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=Path(__file__).parent.parent  # Set working directory to python/
        )
        
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        
        if output.startswith("SUCCESS|"):
            parts = output.split("|")
            parse_time = float(parts[1])
            num_clauses = int(parts[2])
            return parse_time, num_clauses, None
        elif output.startswith("ERROR|"):
            error_msg = output.split("|", 1)[1]
            return None, None, error_msg
        else:
            # If we didn't get expected output, check stderr and return code
            if result.returncode != 0:
                if stderr:
                    return None, None, f"Process failed: {stderr}"
                elif output:
                    return None, None, f"Process failed with output: {output}"
                else:
                    return None, None, f"Process failed with exit code {result.returncode}"
            elif not output and not stderr:
                return None, None, f"Parser produced no output (possible crash or import failure)"
            else:
                return None, None, f"Unexpected output: {output[:100] if output else 'empty'}"
            
    except subprocess.TimeoutExpired:
        return None, None, f"Timeout: Parser exceeded {timeout_seconds} seconds"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {str(e)}"


def benchmark_vampire_parser(filepath: str, include_path: str, vampire_binary: str, timeout_seconds: int = 30) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Benchmark Vampire parser on a file with timeout."""
    try:
        cmd = [vampire_binary, '--mode', 'clausify', filepath, '--include', include_path]
        
        start_time = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        end_time = time.perf_counter()
        
        if result.returncode != 0:
            return None, None, f"Vampire failed: {result.stderr}"
        
        parse_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Count clauses in vampire output
        # Vampire outputs clauses in format: cnf(name, role, clause).
        num_clauses = result.stdout.count('cnf(')
        
        return parse_time, num_clauses, None
    except subprocess.TimeoutExpired:
        return None, None, f"Operation timed out after {timeout_seconds} seconds"
    except Exception as e:
        return None, None, str(e)


def collect_tptp_files(tptp_dir: Path, max_files: int, categories: List[str]) -> List[Path]:
    """Collect TPTP problem files to benchmark."""
    all_files = []
    problems_dir = tptp_dir / "Problems"
    
    if categories:
        # Collect from specific categories
        for category in categories:
            cat_dir = problems_dir / category
            if cat_dir.exists():
                files = list(cat_dir.glob("*.p"))
                all_files.extend(files)
    else:
        # Collect from all categories
        for cat_dir in problems_dir.iterdir():
            if cat_dir.is_dir():
                files = list(cat_dir.glob("*.p"))
                all_files.extend(files)
    
    # Sample if we have too many files
    if len(all_files) > max_files:
        all_files = random.sample(all_files, max_files)
    
    return sorted(all_files)


def format_time(ms: Optional[float]) -> str:
    """Format time in milliseconds."""
    if ms is None:
        return "N/A"
    elif ms < 1:
        return f"{ms:.3f}"
    elif ms < 10:
        return f"{ms:.2f}"
    elif ms < 100:
        return f"{ms:.1f}"
    else:
        return f"{ms:.0f}"


def print_results(results: List[BenchmarkResult], tptp_dir: Path):
    """Print benchmark results in a formatted table."""
    # Separate results by category with more nuanced handling
    actually_both_succeeded = []
    format_limitations = []
    only_vampire = []
    both_failed = []
    
    for r in results:
        if r.rust_error and r.vampire_error:
            both_failed.append(r)
        elif r.rust_error and not r.vampire_error:
            only_vampire.append(r)
        elif not r.rust_error and not r.vampire_error:
            # Both "succeeded" but check if Rust actually parsed anything
            if r.rust_clauses == 0 and r.vampire_clauses > 0:
                # Rust returned 0 clauses but Vampire found clauses - likely unsupported format
                full_path = tptp_dir / r.filename
                file_format = get_file_format(full_path)
                if file_format in ['THF', 'TFF']:
                    format_limitations.append(r)
                else:
                    # Unexpected: supported format but 0 clauses
                    actually_both_succeeded.append(r)
            else:
                actually_both_succeeded.append(r)
    
    # Print results where both parsers actually succeeded and parsed content
    if actually_both_succeeded:
        print("\n=== Files Both Parsers Successfully Parse ===")
        print(f"{'File':<30} {'Format':<8} {'Size':<8} {'Rust(ms)':<10} {'Vampire(ms)':<12} {'Speedup':<10} {'R_Clauses':<10} {'V_Clauses':<10}")
        print("-" * 118)
        
        total_speedup = 0
        count_with_valid_speedup = 0
        for r in actually_both_succeeded:
            full_path = tptp_dir / r.filename
            file_format = get_file_format(full_path)
            size_kb = r.file_size / 1024
            
            # Only count speedup if both parsers produced clauses
            if r.rust_clauses > 0 and r.vampire_clauses > 0 and r.speedup:
                total_speedup += r.speedup
                count_with_valid_speedup += 1
                speedup_str = f"{r.speedup:.1f}x"
            else:
                speedup_str = "N/A"
            
            print(f"{r.filename:<30} {file_format:<8} {size_kb:<8.1f} {format_time(r.rust_time):<10} "
                  f"{format_time(r.vampire_time):<12} {speedup_str:<10} "
                  f"{r.rust_clauses:<10} {r.vampire_clauses:<10}")
        
        if count_with_valid_speedup > 0:
            avg_speedup = total_speedup / count_with_valid_speedup
            print(f"\nAverage speedup (on files with clauses): {avg_speedup:.1f}x")
    
    # Print files with format limitations (THF/TFF that Rust doesn't support)
    if format_limitations:
        print("\n=== Files with Expected Format Limitations (Rust returned 0 clauses) ===")
        print(f"{'File':<30} {'Format':<8} {'Vampire Clauses':<15} {'Note':<40}")
        print("-" * 95)
        
        for r in format_limitations:
            full_path = tptp_dir / r.filename
            file_format = get_file_format(full_path)
            print(f"{r.filename:<30} {file_format:<8} {r.vampire_clauses:<15} {'Rust does not support ' + file_format:<40}")
    
    # Print files only Vampire could handle
    if only_vampire:
        print("\n=== Files Only Vampire Handles (Rust Parser Failed) ===")
        
        # Group by format for better analysis
        format_groups = {}
        for r in only_vampire:
            # Need to construct full path for format detection
            full_path = tptp_dir / r.filename
            file_format = get_file_format(full_path)
            if file_format not in format_groups:
                format_groups[file_format] = []
            format_groups[file_format].append(r)
        
        # Print by format
        for fmt in sorted(format_groups.keys()):
            files = format_groups[fmt]
            print(f"\n{fmt} files ({len(files)} total):")
            print(f"{'File':<40} {'Error':<60}")
            print("-" * 100)
            
            # Show first 5 files of each format
            for r in files[:5]:
                error_msg = r.rust_error.replace('\n', ' ') if r.rust_error else "Unknown error"
                if 'Parse error:' in error_msg:
                    error_msg = error_msg.split('Parse error:')[1].strip()
                if len(error_msg) > 60:
                    error_msg = error_msg[:57] + "..."
                print(f"{r.filename:<40} {error_msg:<60}")
            
            if len(files) > 5:
                print(f"... and {len(files) - 5} more {fmt} files")
        
        print("\n=== Format Distribution of Rust Parser Failures ===")
        for fmt, files in sorted(format_groups.items()):
            print(f"  {fmt}: {len(files)} files")
            
        # Check if most failures are expected (non-CNF/FOF)
        expected_failures = sum(len(files) for fmt, files in format_groups.items() 
                              if fmt in ['THF', 'TFF', 'Include-only', 'Unknown'])
        unexpected_failures = sum(len(files) for fmt, files in format_groups.items() 
                                if fmt in ['CNF', 'FOF'])
        
        if expected_failures > 0 or unexpected_failures > 0:
            print(f"\nExpected failures (THF/TFF/Unknown): {expected_failures}")
            print(f"Unexpected failures (CNF/FOF): {unexpected_failures}")
    
    # Summary statistics
    print("\n=== Summary ===")
    print(f"Total files tested: {len(results)}")
    print(f"Both parsers succeeded (with content): {len(actually_both_succeeded)} ({len(actually_both_succeeded)/len(results)*100:.1f}%)")
    print(f"Format limitations (THF/TFF): {len(format_limitations)} ({len(format_limitations)/len(results)*100:.1f}%)")
    print(f"Rust parser failed: {len(only_vampire)} ({len(only_vampire)/len(results)*100:.1f}%)")
    print(f"Both parsers failed: {len(both_failed)} ({len(both_failed)/len(results)*100:.1f}%)")
    
    if actually_both_succeeded:
        # Calculate average times only for files where both parsers produced clauses
        files_with_clauses = [r for r in actually_both_succeeded if r.rust_clauses > 0 and r.vampire_clauses > 0]
        if files_with_clauses:
            avg_rust_time = sum(r.rust_time for r in files_with_clauses) / len(files_with_clauses)
            avg_vampire_time = sum(r.vampire_time for r in files_with_clauses) / len(files_with_clauses)
            print(f"\nAverage parse time on files with clauses:")
            print(f"  Rust parser: {avg_rust_time:.2f}ms")
            print(f"  Vampire: {avg_vampire_time:.2f}ms")
        
        # Count file formats that succeeded
        format_success = {}
        for r in actually_both_succeeded:
            full_path = tptp_dir / r.filename
            file_format = get_file_format(full_path)
            format_success[file_format] = format_success.get(file_format, 0) + 1
        
        print(f"\nFormats successfully parsed by Rust parser:")
        for fmt, count in sorted(format_success.items()):
            print(f"  {fmt}: {count} (with {sum(1 for r in actually_both_succeeded if get_file_format(tptp_dir / r.filename) == fmt and r.rust_clauses > 0)} producing clauses)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Rust parser vs Vampire on TPTP problems")
    parser.add_argument('--tptp-dir', type=str, default=None,
                        help='Path to TPTP directory (auto-detects if not specified)')
    parser.add_argument('--vampire-binary', type=str, 
                        default=os.path.expanduser('~/.vampire/bin/vampire_z3_rel_static_casc2023_6749'),
                        help='Path to vampire binary')
    parser.add_argument('--max-files', type=int, default=100,
                        help='Maximum number of files to test')
    parser.add_argument('--categories', nargs='+', default=[],
                        help='Specific TPTP categories to test (e.g., ALG SET)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for file sampling')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds for each parser (default: 30)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Auto-detect TPTP directory if not specified
    if args.tptp_dir is None:
        # Try common locations
        possible_paths = [
            Path('.data/problems/tptp/TPTP-v9.0.0'),
            Path('../.data/problems/tptp/TPTP-v9.0.0'),
            Path('python/.data/problems/tptp/TPTP-v9.0.0'),
            Path.home() / '.data/problems/tptp/TPTP-v9.0.0',
        ]
        for path in possible_paths:
            if path.exists():
                tptp_dir = path
                print(f"Auto-detected TPTP directory: {tptp_dir}")
                break
        else:
            print("Error: Could not find TPTP directory.")
            print("Please download it with: proofatlas-download-tptp")
            print("Or specify the path with --tptp-dir")
            sys.exit(1)
    else:
        tptp_dir = Path(args.tptp_dir)
        if not tptp_dir.exists():
            print(f"Error: TPTP directory not found: {tptp_dir}")
            sys.exit(1)
    
    if not os.path.exists(args.vampire_binary):
        print(f"Error: Vampire binary not found: {args.vampire_binary}")
        sys.exit(1)
    
    # Collect files to test
    print(f"Collecting TPTP files from {tptp_dir}...")
    files = collect_tptp_files(tptp_dir, args.max_files, args.categories)
    print(f"Found {len(files)} files to benchmark")
    
    # Run benchmarks
    results = []
    include_path = str(tptp_dir)
    
    print("\nRunning benchmarks...")
    for i, filepath in enumerate(files):
        print(f"\rProgress: {i+1}/{len(files)}", end='', flush=True)
        
        result = BenchmarkResult(
            filename=str(filepath.relative_to(tptp_dir)),
            file_size=filepath.stat().st_size
        )
        
        # Benchmark Rust parser
        rust_time, rust_clauses, rust_error = benchmark_rust_parser(
            str(filepath), include_path, args.timeout)
        result.rust_time = rust_time
        result.rust_clauses = rust_clauses
        result.rust_error = rust_error
        
        # Debug empty errors
        if rust_error is not None and rust_error == "":
            print(f"\nWarning: Empty error for {filepath.name}")
        
        # Benchmark Vampire parser
        vampire_time, vampire_clauses, vampire_error = benchmark_vampire_parser(
            str(filepath), include_path, args.vampire_binary, args.timeout)
        result.vampire_time = vampire_time
        result.vampire_clauses = vampire_clauses
        result.vampire_error = vampire_error
        
        results.append(result)
    
    print("\n")  # Clear progress line
    
    # Print results
    print_results(results, tptp_dir)


if __name__ == '__main__':
    main()