#!/usr/bin/env python3
"""Convert flat per-problem trace files to per-state trace files.

Reads existing .{graph,sentence}.npz files (one per problem), iterates over
selection states, extracts U+P clause subsets for each state, and writes
per-state files into PROBLEM/ subdirectories.

Old layout:  traces/age_weight/AGT002+1.graph.npz
New layout:  traces/age_weight/AGT002+1/0.graph.npz
                                        1.graph.npz
                                        ...

Memory strategy: each problem is processed in a subprocess via
multiprocessing.Pool(maxtasksperchild=1). When the subprocess exits,
the OS reclaims ALL its memory — no fragmentation accumulation.
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def _convert_graph(args):
    """Worker function: convert one graph trace. Runs in subprocess."""
    npz_path, out_dir, max_states = args
    out_dir = Path(out_dir)
    npz_path = Path(npz_path)
    out_dir.mkdir(exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as npz:
        if not bool(npz["proof_found"][0]):
            return 0
        num_states = len(npz["state_selected"])
        if num_states == 0:
            return 0

        node_features = npz["node_features"]
        edge_src = npz["edge_src"]
        edge_dst = npz["edge_dst"]
        node_offsets = npz["node_offsets"]
        edge_offsets = npz["edge_offsets"]
        clause_features = npz["clause_features"]
        labels = npz["labels"]
        state_u_offsets = npz["state_u_offsets"]
        state_p_offsets = npz["state_p_offsets"]
        state_u_indices = npz["state_u_indices"]
        state_p_indices = npz["state_p_indices"]

        nf_cols = node_features.shape[1] if node_features.ndim == 2 else 3
        limit = min(num_states, max_states) if max_states else num_states
        written = 0

        for si in range(limit):
            u_s, u_e = int(state_u_offsets[si]), int(state_u_offsets[si + 1])
            p_s, p_e = int(state_p_offsets[si]), int(state_p_offsets[si + 1])
            u_idx = state_u_indices[u_s:u_e].astype(np.int64)
            p_idx = state_p_indices[p_s:p_e].astype(np.int64)

            if len(u_idx) == 0:
                continue

            all_idx = np.concatenate([u_idx, p_idx])
            num_u = len(u_idx)

            # Build per-state node/edge arrays
            nf_parts, es_parts, ed_parts = [], [], []
            s_noff, s_eoff = [0], [0]
            nc = 0

            for ci in all_idx:
                j = int(ci)
                ns, ne = int(node_offsets[j]), int(node_offsets[j + 1])
                es, ee = int(edge_offsets[j]), int(edge_offsets[j + 1])
                nn, nedg = ne - ns, ee - es

                if nn > 0:
                    nf_parts.append(node_features[ns:ne].copy())
                if nedg > 0:
                    es_parts.append(edge_src[es:ee].astype(np.int64) - ns + nc)
                    ed_parts.append(edge_dst[es:ee].astype(np.int64) - ns + nc)

                nc += nn
                s_noff.append(nc)
                s_eoff.append(s_eoff[-1] + nedg)

            cnf = np.concatenate(nf_parts) if nf_parts else np.zeros((0, nf_cols), dtype=np.float32)
            ces = np.concatenate(es_parts) if es_parts else np.zeros(0, dtype=np.int64)
            ced = np.concatenate(ed_parts) if ed_parts else np.zeros(0, dtype=np.int64)

            np.savez(
                out_dir / f"{si}.graph.npz",
                node_features=cnf,
                edge_src=ces,
                edge_dst=ced,
                node_offsets=np.array(s_noff, dtype=np.int64),
                edge_offsets=np.array(s_eoff, dtype=np.int64),
                clause_features=clause_features[all_idx].copy(),
                labels=labels[all_idx].copy(),
                num_u=np.array(num_u, dtype=np.int64),
            )
            written += 1

    return written


def _convert_sentence(args):
    """Worker function: convert one sentence trace. Runs in subprocess."""
    npz_path, out_dir, max_states = args
    out_dir = Path(out_dir)
    npz_path = Path(npz_path)
    out_dir.mkdir(exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as npz:
        if not bool(npz["proof_found"][0]):
            return 0
        num_states = len(npz["state_selected"])
        if num_states == 0:
            return 0

        clause_strings = npz["clause_strings"]
        labels = npz["labels"]
        state_u_offsets = npz["state_u_offsets"]
        state_p_offsets = npz["state_p_offsets"]
        state_u_indices = npz["state_u_indices"]
        state_p_indices = npz["state_p_indices"]

        limit = min(num_states, max_states) if max_states else num_states
        written = 0

        for si in range(limit):
            u_s, u_e = int(state_u_offsets[si]), int(state_u_offsets[si + 1])
            p_s, p_e = int(state_p_offsets[si]), int(state_p_offsets[si + 1])
            u_idx = state_u_indices[u_s:u_e].astype(np.int64)
            p_idx = state_p_indices[p_s:p_e].astype(np.int64)

            if len(u_idx) == 0:
                continue

            all_idx = np.concatenate([u_idx, p_idx])
            num_u = len(u_idx)

            np.savez(
                out_dir / f"{si}.sentence.npz",
                clause_strings=np.array([clause_strings[int(i)] for i in all_idx], dtype=object),
                labels=labels[all_idx].copy(),
                num_u=np.array(num_u, dtype=np.int64),
            )
            written += 1

    return written


def estimate_sizes(npz_path: Path) -> dict:
    """Estimate state count and total output size without writing."""
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if not bool(data["proof_found"][0]):
                return {"states": 0, "input_bytes": npz_path.stat().st_size}
            num_states = len(data.get("state_selected", []))
            return {
                "states": num_states,
                "input_bytes": npz_path.stat().st_size,
            }
    except Exception:
        return {"states": 0, "input_bytes": npz_path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description="Convert flat traces to per-state traces")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path(".data/traces/age_weight"),
        help="Directory containing flat trace files (default: .data/traces/age_weight)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Maximum states per problem (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report sizes without writing",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep original flat files after conversion",
    )
    args = parser.parse_args()

    traces_dir = args.traces_dir
    if not traces_dir.exists():
        print(f"Error: {traces_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    graph_files = sorted(f for f in traces_dir.glob("*.graph.npz") if f.is_file())
    sentence_files = sorted(f for f in traces_dir.glob("*.sentence.npz") if f.is_file())

    if not graph_files and not sentence_files:
        print("No flat trace files found. Already converted?")
        sys.exit(0)

    print(f"Found {len(graph_files)} graph + {len(sentence_files)} sentence trace files")

    if args.dry_run:
        total_states = 0
        total_input_bytes = 0
        for f in graph_files:
            info = estimate_sizes(f)
            total_states += info["states"]
            total_input_bytes += info["input_bytes"]
            if info["states"] > 0:
                stem = f.name.rsplit(".", 2)[0]
                print(f"  {stem}: {info['states']} states, {info['input_bytes'] / 1024 / 1024:.1f} MB")
        print(f"\nTotal: {total_states} states from {len(graph_files)} problems")
        print(f"Input size: {total_input_bytes / 1024 / 1024:.1f} MB")
        return

    # Build task lists
    graph_tasks = []
    for f in graph_files:
        stem = f.name.rsplit(".", 2)[0]
        graph_tasks.append((str(f), str(traces_dir / stem), args.max_states))

    sentence_tasks = []
    for f in sentence_files:
        stem = f.name.rsplit(".", 2)[0]
        sentence_tasks.append((str(f), str(traces_dir / stem), args.max_states))

    # Process graph traces — maxtasksperchild=1 ensures each problem runs
    # in a fresh subprocess, so memory is returned to OS after each one.
    print("Converting graph traces...")
    total_graph = 0
    skipped = 0
    with Pool(processes=1, maxtasksperchild=1) as pool:
        for i, n in enumerate(pool.imap(_convert_graph, graph_tasks), 1):
            total_graph += n
            if n == 0:
                skipped += 1
                out_dir = Path(graph_tasks[i - 1][1])
                try:
                    out_dir.rmdir()
                except OSError:
                    pass
            if i % 100 == 0 or i == len(graph_tasks):
                print(f"  Graph: {i}/{len(graph_tasks)} problems ({total_graph} states)")

    print("Converting sentence traces...")
    total_sentence = 0
    with Pool(processes=1, maxtasksperchild=1) as pool:
        for i, n in enumerate(pool.imap(_convert_sentence, sentence_tasks), 1):
            total_sentence += n
            if i % 100 == 0 or i == len(sentence_tasks):
                print(f"  Sentence: {i}/{len(sentence_tasks)} problems ({total_sentence} states)")

    print(f"\nConverted: {total_graph} graph states, {total_sentence} sentence states")
    print(f"Skipped: {skipped} problems (no proof or no states)")

    if not args.keep_originals:
        print("Deleting original flat files...")
        for f in graph_files:
            f.unlink()
        for f in sentence_files:
            f.unlink()
        print(f"Deleted {len(graph_files) + len(sentence_files)} files")
    else:
        print("Keeping original files (--keep-originals)")


if __name__ == "__main__":
    main()
