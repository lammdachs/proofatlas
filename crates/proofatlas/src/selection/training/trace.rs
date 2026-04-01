//! Trace saving for training data extraction.
//!
//! Writes graph NPZ, sentence NPZ, and proof strings JSON for a completed proof.
//! Used by both the Python `save_trace` binding and the Rust worker pool.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::logic::{Clause, Interner};
use crate::prover::Prover;
use crate::state::{ProofResult, StateChange};

use super::npz::NpzWriter;

#[cfg(feature = "ml")]
use crate::selection::pipeline::backend::BackendHandle;
#[cfg(feature = "ml")]
use crate::selection::ml::graph::GraphBuilder;

/// Save a proof trace (graph NPZ, sentence NPZ, proof strings JSON).
///
/// If `external_labels` is provided, those labels are used instead of deriving
/// them from this run's proof. This allows saving traces for failed runs using
/// labels from a different proof (e.g., the baseline).
#[cfg(feature = "ml")]
pub fn save_trace(
    prover: &Prover,
    result: &ProofResult,
    backend_handle: Option<&BackendHandle>,
    traces_dir: &str,
    preset: &str,
    problem: &str,
    external_labels: Option<Vec<u8>>,
) -> Result<(), String> {
    let interner = prover.interner();
    let clauses = prover.clauses();
    let num_clauses = clauses.len();

    if num_clauses == 0 {
        return Ok(());
    }

    // When no external labels, require a proof to derive labels from
    if external_labels.is_none() && !matches!(result, ProofResult::Proof { .. }) {
        return Ok(());
    }

    let stem = std::path::Path::new(problem)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(problem);

    let preset_dir = PathBuf::from(traces_dir).join(preset);
    std::fs::create_dir_all(&preset_dir)
        .map_err(|e| format!("Failed to create traces dir: {}", e))?;

    // --- Lifecycle arrays ---
    let (transfer_step, activate_step, simplify_step, num_steps) =
        build_clause_lifecycle(prover);

    // --- Proof labels ---
    let labels: Vec<u8> = if let Some(ext) = external_labels {
        if ext.len() != num_clauses {
            return Err(format!(
                "external_labels length {} does not match clause count {}",
                ext.len(), num_clauses
            ));
        }
        ext
    } else {
        let proof_clauses = get_proof_clause_set(prover, result);
        (0..num_clauses)
            .map(|i| if proof_clauses.contains(&i) { 1 } else { 0 })
            .collect()
    };

    // --- Clause features [C, 9] ---
    let clause_features_flat = compute_clause_features_flat(prover);

    let num_steps_arr = [num_steps];

    macro_rules! npz {
        ($w:expr, $method:ident, $($args:expr),+) => {
            $w.$method($($args),+)
                .map_err(|e| format!("NPZ write error: {}", e))?
        };
    }

    // --- Write graph NPZ ---
    {
        let clause_refs: Vec<&Clause> = clauses.iter().map(|c| c.as_ref()).collect();
        let batch_graph = GraphBuilder::build_from_clauses(&clause_refs);
        let node_names = GraphBuilder::collect_node_names(&clause_refs, interner);
        drop(clause_refs);

        let num_nodes = batch_graph.num_nodes;
        let node_features_flat: Vec<f32> = batch_graph.node_features
            .iter().flat_map(|nf| nf.iter().copied()).collect();
        let edge_src: Vec<i32> = batch_graph.edge_indices.iter().map(|&(s, _)| s as i32).collect();
        let edge_dst: Vec<i32> = batch_graph.edge_indices.iter().map(|&(_, d)| d as i32).collect();

        let mut node_offsets = Vec::with_capacity(num_clauses + 1);
        let mut edge_offsets = Vec::with_capacity(num_clauses + 1);
        node_offsets.push(0i64);
        edge_offsets.push(0i64);
        for i in 0..num_clauses {
            node_offsets.push(batch_graph.clause_boundaries[i].1 as i64);
            edge_offsets.push(batch_graph.edge_boundaries[i].1 as i64);
        }
        drop(batch_graph);

        // Node embeddings via MiniLM (if backend available)
        let node_embeddings: Option<Vec<f32>>;
        let node_sentinel_type: Option<Vec<i8>>;

        if let Some(handle) = backend_handle {
            let mut sentinel_types = vec![-1i8; num_nodes];
            let mut real_indices = Vec::new();
            let mut real_names = Vec::new();

            for (i, name) in node_names.iter().enumerate() {
                match name.as_str() {
                    "VAR" => sentinel_types[i] = 0,
                    "CLAUSE" => sentinel_types[i] = 1,
                    "LIT" => sentinel_types[i] = 2,
                    _ => {
                        real_indices.push(i);
                        real_names.push(name.clone());
                    }
                }
            }

            let mut unique_names: Vec<String> = Vec::new();
            let mut name_to_uid: HashMap<String, usize> = HashMap::new();
            for name in &real_names {
                if !name_to_uid.contains_key(name) {
                    name_to_uid.insert(name.clone(), unique_names.len());
                    unique_names.push(name.clone());
                }
            }

            let unique_embs = if unique_names.is_empty() {
                vec![]
            } else {
                encode_strings_via_handle(handle, unique_names)
            };

            let emb_dim = 384;
            let mut flat_node_emb = vec![0.0f32; num_nodes * emb_dim];
            for (j, &node_idx) in real_indices.iter().enumerate() {
                let uid = name_to_uid[&real_names[j]];
                let src = &unique_embs[uid];
                let dst_start = node_idx * emb_dim;
                flat_node_emb[dst_start..dst_start + emb_dim].copy_from_slice(src);
            }

            node_embeddings = Some(flat_node_emb);
            node_sentinel_type = Some(sentinel_types);
        } else {
            node_embeddings = None;
            node_sentinel_type = None;
        }

        let graph_path = preset_dir.join(format!("{}.graph.npz", stem));
        let mut gw = NpzWriter::new(&graph_path)
            .map_err(|e| format!("Failed to create graph NPZ: {}", e))?;

        npz!(gw, write_array_2d, "node_features", &node_features_flat, num_nodes, 3);
        npz!(gw, write_array_1d, "edge_src", &edge_src);
        npz!(gw, write_array_1d, "edge_dst", &edge_dst);
        npz!(gw, write_array_1d, "node_offsets", &node_offsets);
        npz!(gw, write_array_1d, "edge_offsets", &edge_offsets);
        npz!(gw, write_array_2d, "clause_features", &clause_features_flat, num_clauses, 9);
        npz!(gw, write_array_1d, "labels", &labels);
        npz!(gw, write_array_1d, "transfer_step", &transfer_step);
        npz!(gw, write_array_1d, "activate_step", &activate_step);
        npz!(gw, write_array_1d, "simplify_step", &simplify_step);
        npz!(gw, write_array_1d, "num_steps", &num_steps_arr);

        if let Some(ref embs) = node_embeddings {
            npz!(gw, write_array_2d, "node_embeddings", embs, num_nodes, 384);
        }
        if let Some(ref st) = node_sentinel_type {
            npz!(gw, write_array_1d, "node_sentinel_type", st);
        }

        gw.finish().map_err(|e| format!("NPZ finish error: {}", e))?;
    }

    // --- Write sentence NPZ ---
    {
        let clause_embeddings: Option<Vec<f32>> = if let Some(handle) = backend_handle {
            let clause_strings: Vec<String> = clauses
                .iter()
                .map(|c| c.display(interner).to_string())
                .collect();
            let clause_embs = encode_strings_via_handle(handle, clause_strings);
            Some(clause_embs.into_iter().flatten().collect())
        } else {
            None
        };

        let sentence_path = preset_dir.join(format!("{}.sentence.npz", stem));
        let mut sw = NpzWriter::new(&sentence_path)
            .map_err(|e| format!("Failed to create sentence NPZ: {}", e))?;

        npz!(sw, write_array_2d, "clause_features", &clause_features_flat, num_clauses, 9);
        npz!(sw, write_array_1d, "labels", &labels);
        npz!(sw, write_array_1d, "transfer_step", &transfer_step);
        npz!(sw, write_array_1d, "activate_step", &activate_step);
        npz!(sw, write_array_1d, "simplify_step", &simplify_step);
        npz!(sw, write_array_1d, "num_steps", &num_steps_arr);

        if let Some(ref embs) = clause_embeddings {
            npz!(sw, write_array_2d, "clause_embeddings", embs, num_clauses, 384);
        }

        sw.finish().map_err(|e| format!("NPZ finish error: {}", e))?;
    }

    // --- Write proof clause strings ---
    {
        let proof_strings: Vec<String> = clauses
            .iter()
            .enumerate()
            .filter(|(i, _)| labels[*i] == 1)
            .map(|(_, c)| c.display(interner).to_string())
            .collect();

        if !proof_strings.is_empty() {
            let strings_path = preset_dir.join(format!("{}.strings.json", stem));
            let json = serde_json::to_string(&proof_strings)
                .map_err(|e| format!("JSON error: {}", e))?;
            std::fs::write(&strings_path, json)
                .map_err(|e| format!("Failed to write strings: {}", e))?;
        }
    }

    Ok(())
}

// =============================================================================
// Helper functions
// =============================================================================

fn build_clause_lifecycle(prover: &Prover) -> (Vec<i32>, Vec<i32>, Vec<i32>, i32) {
    let events = prover.event_log();
    let num_clauses = prover.clauses().len();

    let mut transfer_step = vec![-1i32; num_clauses];
    let mut activate_step = vec![-1i32; num_clauses];
    let mut simplify_step = vec![-1i32; num_clauses];
    let mut step: i32 = 0;

    for event in events {
        match event {
            StateChange::Transfer(idx) => {
                if *idx < num_clauses {
                    transfer_step[*idx] = step;
                }
            }
            StateChange::Activate(idx) => {
                if *idx < num_clauses {
                    activate_step[*idx] = step;
                }
                step += 1;
            }
            StateChange::Simplify(idx, _replacement, _, _) => {
                if *idx < num_clauses {
                    simplify_step[*idx] = step;
                }
            }
            StateChange::Add(_, _, _) => {}
        }
    }

    (transfer_step, activate_step, simplify_step, step)
}

fn get_proof_clause_set(prover: &Prover, result: &ProofResult) -> HashSet<usize> {
    let empty_id = match result {
        ProofResult::Proof { empty_clause_idx } => *empty_clause_idx,
        _ => return HashSet::new(),
    };

    let events = prover.event_log();
    let mut derivation_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for event in events {
        match event {
            StateChange::Add(clause, _, premises) => {
                if let Some(idx) = clause.id {
                    derivation_map.insert(idx, crate::state::clause_indices(premises));
                }
            }
            StateChange::Simplify(_, Some(clause), _, premises) => {
                if let Some(idx) = clause.id {
                    derivation_map.insert(idx, crate::state::clause_indices(premises));
                }
            }
            _ => {}
        }
    }

    let mut proof_clauses = HashSet::new();
    let mut to_visit = vec![empty_id];
    while let Some(current_id) = to_visit.pop() {
        if proof_clauses.contains(&current_id) {
            continue;
        }
        proof_clauses.insert(current_id);
        if let Some(parents) = derivation_map.get(&current_id) {
            to_visit.extend(parents);
        }
    }

    proof_clauses
}

fn compute_clause_features_flat(prover: &Prover) -> Vec<f32> {
    let clauses = prover.clauses();
    let events = prover.event_log();
    let num_clauses = clauses.len();

    let mut derivation_info: HashMap<usize, String> = HashMap::new();
    for event in events {
        match event {
            StateChange::Add(clause, rule_name, _) => {
                if let Some(idx) = clause.id {
                    derivation_info.insert(idx, rule_name.clone());
                }
            }
            StateChange::Simplify(_, Some(clause), rule_name, _) => {
                if let Some(idx) = clause.id {
                    derivation_info.insert(idx, rule_name.clone());
                }
            }
            _ => {}
        }
    }

    let mut features = Vec::with_capacity(num_clauses * 9);
    for (idx, clause) in clauses.iter().enumerate() {
        let rule_name = derivation_info.get(&idx).map(|s| s.as_str()).unwrap_or("input");
        let rule_id = match rule_name {
            "input" => 0.0f32,
            "resolution" => 1.0,
            "factoring" => 2.0,
            "superposition" => 3.0,
            "equality_resolution" => 4.0,
            "equality_factoring" => 5.0,
            "demodulation" => 6.0,
            _ => 7.0,
        };

        let mut depth = 0usize;
        let mut symbol_count = 0usize;
        let mut variable_count = 0usize;
        let mut distinct_symbols = HashSet::new();
        let mut distinct_variables = HashSet::new();

        for lit in &clause.literals {
            symbol_count += 1;
            distinct_symbols.insert(lit.predicate.id.0 as u64 | (1u64 << 32));
            for arg in &lit.args {
                let (d, sc, vc) = term_stats(arg, &mut distinct_symbols, &mut distinct_variables);
                depth = depth.max(d);
                symbol_count += sc;
                variable_count += vc;
            }
        }

        features.extend_from_slice(&[
            clause.age as f32,
            clause.role.to_feature_value(),
            rule_id,
            clause.literals.len() as f32,
            depth as f32,
            symbol_count as f32,
            distinct_symbols.len() as f32,
            variable_count as f32,
            distinct_variables.len() as f32,
        ]);
    }
    features
}

fn term_stats(
    term: &crate::logic::Term,
    distinct_symbols: &mut HashSet<u64>,
    distinct_variables: &mut HashSet<u64>,
) -> (usize, usize, usize) {
    use crate::logic::Term;
    match term {
        Term::Variable(v) => {
            distinct_variables.insert(v.id.0 as u64);
            (0, 0, 1)
        }
        Term::Constant(c) => {
            distinct_symbols.insert(c.id.0 as u64 | (2u64 << 32));
            (0, 1, 0)
        }
        Term::Function(f, args) => {
            distinct_symbols.insert(f.id.0 as u64 | (3u64 << 32));
            let mut max_depth = 0usize;
            let mut sc = 1usize;
            let mut vc = 0usize;
            for arg in args {
                let (d, s, v) = term_stats(arg, distinct_symbols, distinct_variables);
                max_depth = max_depth.max(d);
                sc += s;
                vc += v;
            }
            (max_depth + 1, sc, vc)
        }
    }
}

/// Encode strings via BackendHandle (MiniLM). Batches large inputs.
#[cfg(feature = "ml")]
pub fn encode_strings_via_handle(
    handle: &BackendHandle,
    strings: Vec<String>,
) -> Vec<Vec<f32>> {
    if strings.is_empty() {
        return vec![];
    }
    const BATCH_SIZE: usize = 1024;
    if strings.len() <= BATCH_SIZE {
        match handle.submit_sync(0, "minilm".to_string(), Box::new(strings), true) {
            Ok(resp) => *resp.data.downcast::<Vec<Vec<f32>>>().unwrap_or(Box::new(vec![])),
            Err(_) => vec![],
        }
    } else {
        let mut all_embs = Vec::with_capacity(strings.len());
        for chunk in strings.chunks(BATCH_SIZE) {
            let batch: Vec<String> = chunk.to_vec();
            match handle.submit_sync(0, "minilm".to_string(), Box::new(batch), true) {
                Ok(resp) => {
                    let embs = *resp.data.downcast::<Vec<Vec<f32>>>().unwrap_or(Box::new(vec![]));
                    all_embs.extend(embs);
                }
                Err(_) => {
                    for _ in chunk {
                        all_embs.push(vec![0.0; 384]);
                    }
                }
            }
        }
        all_embs
    }
}
