//! Pipelined inference architecture: ChannelSink + data processing thread.
//!
//! The prover sends signals to a data processing thread via channels.
//! The data processing thread manages embedding, scoring, and selection,
//! delegating model computation to a shared Backend.
//!
//! # Architecture
//!
//! ```text
//! Prover ──→ ChannelSink ──(mpsc)──→ Data Processing Thread ──→ BackendHandle ──→ Backend
//!                                    (embedding cache,           (GPU/CPU models)
//!                                     softmax sampling)
//! ```
//!
//! The data processing thread receives `ProverSignal`s and:
//! - On `Transfer`: submits clause to Backend for embedding+scoring, caches result
//! - On `Activate`: tracks U→P transition
//! - On `Simplify`: evicts clause from caches
//! - On `Select`: softmax-samples from cached scores, responds with selected index

use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;

use indexmap::IndexMap;

use crate::logic::Clause;
pub mod backend;

use self::backend::{BackendHandle, Model};
use crate::selection::cached::{ClauseEmbedder, EmbeddingScorer};
use crate::selection::clause::ProverSink;

// =============================================================================
// ProverSignal — messages from prover to data processing thread
// =============================================================================

/// Signals sent by the prover to the data processing thread.
pub enum ProverSignal {
    /// Clause entered U (survived forward simplification).
    Transfer(usize, Arc<Clause>),
    /// Clause moved U→P (activated as given clause).
    Activate(usize),
    /// Clause removed from U or P by simplification.
    Simplify(usize),
    /// Request clause selection. The data processing thread sends the
    /// selected clause index (or None) back via the response channel.
    Select(mpsc::Sender<Option<usize>>),
}

// =============================================================================
// ChannelSink — ProverSink that sends signals to a data processing thread
// =============================================================================

/// ProverSink that sends signals to a data processing thread via channels.
///
/// The data processing thread runs concurrently with the prover, managing
/// embedding computation and scoring. On `select()`, the sink sends a
/// Select signal and blocks until the data processing thread responds.
///
/// The data processing thread's JoinHandle is owned by this struct and
/// joined on Drop, ensuring clean shutdown.
pub struct ChannelSink {
    /// Wrapped in Option so Drop can take it first (closing the channel
    /// before joining the thread).
    tx: Option<mpsc::Sender<ProverSignal>>,
    /// Data processing thread handle, joined on Drop.
    thread_handle: Option<JoinHandle<()>>,
    /// Name for profiling.
    name: String,
}

impl ChannelSink {
    /// Create a new ChannelSink with the sending end of the signal channel
    /// and the data processing thread handle.
    pub fn new(
        tx: mpsc::Sender<ProverSignal>,
        thread_handle: JoinHandle<()>,
        name: String,
    ) -> Self {
        Self {
            tx: Some(tx),
            thread_handle: Some(thread_handle),
            name,
        }
    }
}

impl ProverSink for ChannelSink {
    fn on_transfer(&mut self, clause_idx: usize, clause: &Arc<Clause>) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(ProverSignal::Transfer(clause_idx, Arc::clone(clause)));
        }
    }

    fn on_activate(&mut self, clause_idx: usize) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(ProverSignal::Activate(clause_idx));
        }
    }

    fn on_simplify(&mut self, clause_idx: usize) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(ProverSignal::Simplify(clause_idx));
        }
    }

    fn select(&mut self) -> Option<usize> {
        let tx = self.tx.as_ref()?;
        let (resp_tx, resp_rx) = mpsc::channel();
        let _ = tx.send(ProverSignal::Select(resp_tx));
        resp_rx.recv().ok().flatten()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        // ChannelSink is created fresh per problem — reset is a no-op.
    }
}

impl Drop for ChannelSink {
    fn drop(&mut self) {
        // 1. Drop the sender to close the channel, signaling the data
        //    processing thread to exit its signal loop.
        self.tx.take();
        // 2. Join the thread to ensure clean shutdown.
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

// =============================================================================
// EmbedScoreModel — Backend Model wrapping ClauseEmbedder + EmbeddingScorer
// =============================================================================

/// Backend Model that combines a clause embedder and scorer.
///
/// Handles two input modes:
/// - `Arc<Clause>` — GCN path: clauses arrive as-is, calls `embed_batch`
/// - `String` — Sentence path: clauses pre-serialized by data processing thread, calls `embed_texts`
///
/// This gives the Backend natural batching: when multiple data processing threads
/// submit concurrently, the Backend groups them and processes in one forward pass.
pub struct EmbedScoreModel {
    embedder: Box<dyn ClauseEmbedder + Send>,
    scorer: Box<dyn EmbeddingScorer + Send>,
}

impl EmbedScoreModel {
    pub fn new(
        embedder: Box<dyn ClauseEmbedder + Send>,
        scorer: Box<dyn EmbeddingScorer + Send>,
    ) -> Self {
        Self { embedder, scorer }
    }
}

impl Model for EmbedScoreModel {
    fn model_id(&self) -> &str {
        "embed_score"
    }

    fn execute_batch(
        &mut self,
        requests: Vec<(u64, Box<dyn std::any::Any + Send>)>,
    ) -> Vec<(u64, Box<dyn std::any::Any + Send>)> {
        let ids: Vec<u64> = requests.iter().map(|(id, _)| *id).collect();

        // Check if input is pre-serialized strings (sentence path) or clauses (GCN path)
        let is_string = requests.first().map(|(_, d)| d.is::<String>()).unwrap_or(false);

        let embeddings = if is_string {
            let texts: Vec<String> = requests
                .into_iter()
                .map(|(_, data)| *data.downcast::<String>().unwrap())
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            self.embedder.embed_texts(&text_refs)
        } else {
            let clauses: Vec<Arc<Clause>> = requests
                .into_iter()
                .map(|(_, data)| *data.downcast::<Arc<Clause>>().unwrap())
                .collect();
            let clause_refs: Vec<&Clause> = clauses.iter().map(|c| c.as_ref()).collect();
            self.embedder.embed_batch(&clause_refs)
        };

        let emb_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let scores = self.scorer.score_batch(&emb_refs);

        ids.into_iter()
            .zip(scores)
            .map(|(id, score)| (id, Box::new(score) as Box<dyn std::any::Any + Send>))
            .collect()
    }
}

// =============================================================================
// Data processing signal loop — incremental scoring
// =============================================================================

/// Preprocessor that converts `Arc<Clause>` to model-ready input.
///
/// For GCN/GAT: identity (passes through `Arc<Clause>`)
/// For Sentence: captures interner, converts clause→String
pub type Preprocessor = Box<dyn Fn(Arc<Clause>) -> Box<dyn std::any::Any + Send> + Send>;

/// Create an identity preprocessor that passes clauses through unchanged.
pub fn identity_preprocessor() -> Preprocessor {
    Box::new(|clause: Arc<Clause>| Box::new(clause) as Box<dyn std::any::Any + Send>)
}

/// Data processing signal loop for incremental scoring (MLP-style scorers).
///
/// On Transfer: preprocesses clause, submits to backend for embed+score, caches result.
/// On Activate: no-op (clause already removed from internal U during select).
/// On Simplify: removes clause from internal U.
/// On Select: softmax-samples from cached scores.
fn incremental_score_loop(
    rx: mpsc::Receiver<ProverSignal>,
    backend: BackendHandle,
    temperature: f32,
    preprocess: Preprocessor,
) {
    let mut unprocessed: IndexMap<usize, f32> = IndexMap::new();
    let mut rng_state: u64 = 0x12345678_9abcdef0;
    let mut request_id: u64 = 0;

    while let Ok(signal) = rx.recv() {
        match signal {
            ProverSignal::Transfer(idx, clause) => {
                request_id += 1;
                let data = preprocess(clause);
                match backend.submit_sync(
                    request_id,
                    "embed_score".to_string(),
                    data,
                ) {
                    Ok(resp) => {
                        let score = *resp.data.downcast::<f32>().unwrap_or(Box::new(0.0f32));
                        unprocessed.insert(idx, score);
                    }
                    Err(_) => {
                        // Backend closed — assign default score
                        unprocessed.insert(idx, 0.0);
                    }
                }
            }
            ProverSignal::Activate(_idx) => {
                // Already removed from unprocessed during select()
            }
            ProverSignal::Simplify(idx) => {
                unprocessed.shift_remove(&idx);
            }
            ProverSignal::Select(resp_tx) => {
                let selected = softmax_sample(&unprocessed, temperature, &mut rng_state);
                if let Some(idx) = selected {
                    unprocessed.shift_remove(&idx);
                }
                let _ = resp_tx.send(selected);
            }
        }
    }
}

// =============================================================================
// Softmax sampling
// =============================================================================

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn softmax_sample(
    scores: &IndexMap<usize, f32>,
    temperature: f32,
    rng_state: &mut u64,
) -> Option<usize> {
    if scores.is_empty() {
        return None;
    }

    let max_score = scores.values().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores
        .values()
        .map(|&s| ((s - max_score) / temperature).exp())
        .collect();
    let sum: f32 = exp_scores.iter().sum();
    if sum == 0.0 {
        return scores.keys().next().copied();
    }

    let r = (xorshift64(rng_state) as f64) / (u64::MAX as f64);
    let mut cumulative = 0.0f64;
    for (i, &exp_s) in exp_scores.iter().enumerate() {
        cumulative += (exp_s / sum) as f64;
        if r <= cumulative {
            let (idx, _) = scores.get_index(i)?;
            return Some(*idx);
        }
    }

    // Fallback: return last
    scores.keys().last().copied()
}

// =============================================================================
// Pipeline factory functions
// =============================================================================

/// Create a pipeline for ML-based clause selection with incremental scoring.
///
/// Spawns a Backend with the given embedder+scorer and a data processing thread.
/// Returns a ChannelSink that implements ProverSink.
///
/// The Backend's worker thread is detached but stays alive via the
/// BackendHandle held by the data processing thread. When the ChannelSink
/// is dropped → data processing thread exits → BackendHandle dropped →
/// Backend worker exits.
pub fn create_ml_pipeline(
    embedder: Box<dyn ClauseEmbedder + Send>,
    scorer: Box<dyn EmbeddingScorer + Send>,
    temperature: f32,
) -> ChannelSink {
    let model = EmbedScoreModel::new(embedder, scorer);
    let backend = self::backend::Backend::new(vec![Box::new(model)]);
    let handle = backend.handle();
    // Backend is dropped here; worker thread detached but alive via handle.
    create_pipeline_with_handle(handle, temperature, "ml_pipeline".to_string(), identity_preprocessor())
}

/// Create a pipeline using an existing Backend handle.
///
/// Use this when sharing a Backend across multiple problems (e.g., bench.py).
/// The caller is responsible for keeping the Backend alive.
///
/// The `preprocess` closure converts `Arc<Clause>` to model-ready input
/// in the data processing thread before submitting to the Backend.
pub fn create_pipeline_with_handle(
    handle: BackendHandle,
    temperature: f32,
    name: String,
    preprocess: Preprocessor,
) -> ChannelSink {
    let (tx, rx) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name(format!("data-{}", name))
        .spawn(move || {
            incremental_score_loop(rx, handle, temperature, preprocess);
        })
        .expect("Failed to spawn data processing thread");

    ChannelSink::new(tx, thread, name)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Clause, Literal, PredicateSymbol};
    use crate::logic::interner::PredicateId;

    /// Mock embedder: returns clause literal count as 1D embedding.
    struct MockEmbedder;

    impl ClauseEmbedder for MockEmbedder {
        fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
            clauses
                .iter()
                .map(|c| vec![c.literals.len() as f32])
                .collect()
        }
        fn embedding_dim(&self) -> usize {
            1
        }
        fn name(&self) -> &str {
            "mock"
        }
    }

    /// Mock scorer: returns embedding[0] as score.
    struct MockScorer;

    impl EmbeddingScorer for MockScorer {
        fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
            embeddings.iter().map(|e| e[0]).collect()
        }
        fn name(&self) -> &str {
            "mock"
        }
    }

    fn make_clause(n_lits: usize) -> Arc<Clause> {
        let lits: Vec<Literal> = (0..n_lits)
            .map(|i| Literal {
                predicate: PredicateSymbol {
                    id: PredicateId(i as u32),
                    arity: 0,
                },
                args: vec![],
                polarity: true,
            })
            .collect();
        Arc::new(Clause::new(lits))
    }

    #[test]
    fn test_channel_sink_transfer_and_select() {
        let mut sink = create_ml_pipeline(Box::new(MockEmbedder), Box::new(MockScorer), 1.0);

        // Transfer three clauses with different sizes
        sink.on_transfer(0, &make_clause(1));
        sink.on_transfer(1, &make_clause(3));
        sink.on_transfer(2, &make_clause(2));

        // Select should return one of them
        let s1 = sink.select();
        assert!(s1.is_some());

        let s2 = sink.select();
        assert!(s2.is_some());
        assert_ne!(s1, s2);

        let s3 = sink.select();
        assert!(s3.is_some());

        // All consumed
        let s4 = sink.select();
        assert!(s4.is_none());
    }

    #[test]
    fn test_channel_sink_simplify() {
        let mut sink = create_ml_pipeline(Box::new(MockEmbedder), Box::new(MockScorer), 1.0);

        sink.on_transfer(0, &make_clause(1));
        sink.on_transfer(1, &make_clause(2));

        // Simplify clause 0
        sink.on_simplify(0);

        // Only clause 1 should be available
        let s = sink.select();
        assert_eq!(s, Some(1));

        let s2 = sink.select();
        assert_eq!(s2, None);
    }

    #[test]
    fn test_channel_sink_empty() {
        let mut sink = create_ml_pipeline(Box::new(MockEmbedder), Box::new(MockScorer), 1.0);

        let s = sink.select();
        assert_eq!(s, None);
    }

    #[test]
    fn test_channel_sink_shutdown_on_drop() {
        let mut sink = create_ml_pipeline(Box::new(MockEmbedder), Box::new(MockScorer), 1.0);

        sink.on_transfer(0, &make_clause(1));

        // Drop the sink — should cleanly shut down data processing thread + backend
        drop(sink);
        // No deadlock or panic = success
    }

    #[test]
    fn test_softmax_sample_basic() {
        let mut scores = IndexMap::new();
        scores.insert(0, 1.0f32);
        scores.insert(1, 2.0f32);
        scores.insert(2, 3.0f32);

        let mut rng = 42u64;
        let selected = softmax_sample(&scores, 1.0, &mut rng);
        assert!(selected.is_some());
        assert!(scores.contains_key(&selected.unwrap()));
    }

    #[test]
    fn test_softmax_sample_empty() {
        let scores: IndexMap<usize, f32> = IndexMap::new();
        let mut rng = 42u64;
        assert_eq!(softmax_sample(&scores, 1.0, &mut rng), None);
    }

    #[test]
    fn test_pipeline_with_handle() {
        // Test the shared-backend factory function
        let model = EmbedScoreModel::new(Box::new(MockEmbedder), Box::new(MockScorer));
        let backend = super::backend::Backend::new(vec![Box::new(model)]);
        let handle = backend.handle();

        let mut sink = create_pipeline_with_handle(handle, 1.0, "test".to_string(), identity_preprocessor());

        sink.on_transfer(5, &make_clause(2));
        let s = sink.select();
        assert_eq!(s, Some(5));

        drop(sink);
        // Backend worker exits after sink drops (data processing thread drops its handle)
    }
}
