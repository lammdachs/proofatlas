//! Pipelined inference architecture: ChannelSink + data processing thread.
//!
//! The prover sends signals to a data processing thread via channels.
//! The data processing thread runs a `DataProcessor` that manages caching
//! and scoring, delegating model computation to a shared Backend.
//!
//! # Architecture
//!
//! ```text
//! Prover ──→ ChannelSink ──(mpsc)──→ signal_loop(DataProcessor) ──→ BackendHandle ──→ Backend
//!                                    (config-specific caching,       (GPU/CPU models)
//!                                     softmax sampling)
//! ```
//!
//! Each (encoder, scorer) configuration gets its own `DataProcessor`:
//! - **Score processors** (MLP scorers): cache pre-softmax f32 scores
//! - **Embedding processors** (attention/transformer scorers): cache embeddings,
//!   re-score with P context at select time

use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;

use indexmap::IndexMap;

use crate::logic::Clause;
pub mod backend;
pub mod processors;

use self::backend::Model;
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
// DataProcessor trait — configuration-specific data processing
// =============================================================================

/// Configuration-specific data processor for the signal loop.
///
/// Each (encoder, scorer) combination implements this trait with its own
/// caching strategy:
/// - Score processors (MLP): cache f32 scores, no-op on activate
/// - Embedding processors (attention/transformer): cache embeddings,
///   track U/P split, re-score with context at select time
pub trait DataProcessor: Send {
    /// Handle a clause entering U (survived forward simplification).
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>);
    /// Handle a clause moving U→P (activated as given clause).
    fn on_activate(&mut self, idx: usize);
    /// Handle a clause being removed by simplification.
    fn on_simplify(&mut self, idx: usize);
    /// Select the next clause. Returns its index, removing it from internal state.
    fn select(&mut self) -> Option<usize>;
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
// EmbedScoreModel — Backend Model: embed + score → f32 (MLP path)
// =============================================================================

/// Backend Model that combines a clause embedder and scorer.
///
/// Handles three input modes:
/// - `Arc<Clause>` — GCN path: clauses arrive as-is, calls `embed_batch`
/// - `String` — Sentence path: clauses pre-serialized, calls `embed_texts`
/// - `Vec<f32>` — Features path: pre-extracted features
///
/// Returns a single f32 score per clause.
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

        let embeddings = dispatch_embed(&mut *self.embedder, requests);

        let emb_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let scores = self.scorer.score_batch(&emb_refs);

        ids.into_iter()
            .zip(scores)
            .map(|(id, score)| (id, Box::new(score) as Box<dyn std::any::Any + Send>))
            .collect()
    }
}

// =============================================================================
// EmbedModel — Backend Model: embed → Vec<f32> (embedding-caching path)
// =============================================================================

/// Backend Model that returns embeddings (not scores).
///
/// Used by embedding-caching processors (attention/transformer scorers).
/// Same input dispatch as EmbedScoreModel but returns Vec<f32> per clause.
pub struct EmbedModel {
    embedder: Box<dyn ClauseEmbedder + Send>,
}

impl EmbedModel {
    pub fn new(embedder: Box<dyn ClauseEmbedder + Send>) -> Self {
        Self { embedder }
    }
}

impl Model for EmbedModel {
    fn model_id(&self) -> &str {
        "embed"
    }

    fn execute_batch(
        &mut self,
        requests: Vec<(u64, Box<dyn std::any::Any + Send>)>,
    ) -> Vec<(u64, Box<dyn std::any::Any + Send>)> {
        let ids: Vec<u64> = requests.iter().map(|(id, _)| *id).collect();

        let embeddings = dispatch_embed(&mut *self.embedder, requests);

        ids.into_iter()
            .zip(embeddings)
            .map(|(id, emb)| (id, Box::new(emb) as Box<dyn std::any::Any + Send>))
            .collect()
    }
}

// =============================================================================
// ContextScoreModel — Backend Model: (U_emb, P_emb) → Vec<f32> scores
// =============================================================================

/// Backend Model that scores unprocessed embeddings with processed context.
///
/// Input: `(Vec<Vec<f32>>, Vec<Vec<f32>>)` — (U embeddings, P embeddings)
/// Output: `Vec<f32>` — scores for each U embedding
///
/// Used at select time by embedding-caching processors.
pub struct ContextScoreModel {
    scorer: Box<dyn EmbeddingScorer + Send>,
}

impl ContextScoreModel {
    pub fn new(scorer: Box<dyn EmbeddingScorer + Send>) -> Self {
        Self { scorer }
    }
}

impl Model for ContextScoreModel {
    fn model_id(&self) -> &str {
        "score_context"
    }

    fn execute_batch(
        &mut self,
        requests: Vec<(u64, Box<dyn std::any::Any + Send>)>,
    ) -> Vec<(u64, Box<dyn std::any::Any + Send>)> {
        // Context scoring is a single logical request (all U and P embeddings at once).
        // The batch should contain exactly one request.
        requests
            .into_iter()
            .map(|(id, data)| {
                let (u_embs, p_embs) = *data
                    .downcast::<(Vec<Vec<f32>>, Vec<Vec<f32>>)>()
                    .expect("ContextScoreModel expects (Vec<Vec<f32>>, Vec<Vec<f32>>)");

                let u_refs: Vec<&[f32]> = u_embs.iter().map(|e| e.as_slice()).collect();
                let p_refs: Vec<&[f32]> = p_embs.iter().map(|e| e.as_slice()).collect();

                let scores = self.scorer.score_with_context(&u_refs, &p_refs);
                (id, Box::new(scores) as Box<dyn std::any::Any + Send>)
            })
            .collect()
    }
}

// =============================================================================
// Shared embed dispatch logic
// =============================================================================

/// Dispatch embedding computation based on input type.
///
/// Three-way dispatch:
/// - `String` → `embed_texts` (sentence path)
/// - `Arc<Clause>` → `embed_batch` (GCN path)
/// - `Vec<f32>` → wraps as single-element embedding (features path, pre-extracted)
fn dispatch_embed(
    embedder: &mut dyn ClauseEmbedder,
    requests: Vec<(u64, Box<dyn std::any::Any + Send>)>,
) -> Vec<Vec<f32>> {
    if requests.is_empty() {
        return vec![];
    }

    let is_string = requests.first().map(|(_, d)| d.is::<String>()).unwrap_or(false);
    let is_features = requests.first().map(|(_, d)| d.is::<Vec<f32>>()).unwrap_or(false);

    if is_string {
        let texts: Vec<String> = requests
            .into_iter()
            .map(|(_, data)| *data.downcast::<String>().unwrap())
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        embedder.embed_texts(&text_refs)
    } else if is_features {
        // Features path: each input is a Vec<f32> of raw features.
        // We need to build a batch tensor and run through the embedder.
        // For now, the features are pre-extracted; the embedder handles them
        // via embed_features if available, or we pass through as-is for
        // embedders that don't support this (the score will be the raw features).
        let features: Vec<Vec<f32>> = requests
            .into_iter()
            .map(|(_, data)| *data.downcast::<Vec<f32>>().unwrap())
            .collect();
        let feat_refs: Vec<&[f32]> = features.iter().map(|f| f.as_slice()).collect();
        embedder.embed_features(&feat_refs)
    } else {
        let clauses: Vec<Arc<Clause>> = requests
            .into_iter()
            .map(|(_, data)| *data.downcast::<Arc<Clause>>().unwrap())
            .collect();
        let clause_refs: Vec<&Clause> = clauses.iter().map(|c| c.as_ref()).collect();
        embedder.embed_batch(&clause_refs)
    }
}

// =============================================================================
// Signal loop — drives DataProcessor from ProverSignals
// =============================================================================

/// Generic signal loop that dispatches ProverSignals to a DataProcessor.
///
/// Runs until the channel is closed (sender dropped), which happens when
/// the ChannelSink is dropped.
fn signal_loop(rx: mpsc::Receiver<ProverSignal>, mut processor: Box<dyn DataProcessor>) {
    while let Ok(signal) = rx.recv() {
        match signal {
            ProverSignal::Transfer(idx, clause) => processor.on_transfer(idx, clause),
            ProverSignal::Activate(idx) => processor.on_activate(idx),
            ProverSignal::Simplify(idx) => processor.on_simplify(idx),
            ProverSignal::Select(resp_tx) => {
                let _ = resp_tx.send(processor.select());
            }
        }
    }
}

// =============================================================================
// Softmax sampling (shared by all processors)
// =============================================================================

pub(crate) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

pub(crate) fn softmax_sample(
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

/// Softmax-sample from a parallel list of (index, score) pairs.
///
/// Used by embedding processors that build a temporary score map at select time.
pub(crate) fn softmax_sample_vec(
    indices: &[usize],
    scores: &[f32],
    temperature: f32,
    rng_state: &mut u64,
) -> Option<usize> {
    if indices.is_empty() {
        return None;
    }

    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&s| ((s - max_score) / temperature).exp())
        .collect();
    let sum: f32 = exp_scores.iter().sum();
    if sum == 0.0 {
        return Some(indices[0]);
    }

    let r = (xorshift64(rng_state) as f64) / (u64::MAX as f64);
    let mut cumulative = 0.0f64;
    for (i, &exp_s) in exp_scores.iter().enumerate() {
        cumulative += (exp_s / sum) as f64;
        if r <= cumulative {
            return Some(indices[i]);
        }
    }

    // Fallback: return last
    indices.last().copied()
}

// =============================================================================
// Pipeline factory functions
// =============================================================================

/// Create a pipeline with a specific DataProcessor.
///
/// Spawns a data processing thread running `signal_loop` with the given processor.
/// Returns a ChannelSink that implements ProverSink.
pub fn create_pipeline(
    processor: Box<dyn DataProcessor>,
    name: String,
) -> ChannelSink {
    let (tx, rx) = mpsc::channel();
    let thread_name = format!("data-{}", name);
    let thread = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            signal_loop(rx, processor);
        })
        .expect("Failed to spawn data processing thread");

    ChannelSink::new(tx, thread, name)
}

/// Create a pipeline for ML-based clause selection (MLP scorer, convenience function).
///
/// Spawns a Backend with the given embedder+scorer, creates a GcnScoreProcessor,
/// and returns a ChannelSink.
pub fn create_ml_pipeline(
    embedder: Box<dyn ClauseEmbedder + Send>,
    scorer: Box<dyn EmbeddingScorer + Send>,
    temperature: f32,
) -> ChannelSink {
    let model = EmbedScoreModel::new(embedder, scorer);
    let backend = self::backend::Backend::new(vec![Box::new(model)]);
    let handle = backend.handle();
    // Backend is dropped here; worker thread detached but alive via handle.
    let processor = Box::new(processors::GcnScoreProcessor::new(handle, temperature));
    create_pipeline(processor, "ml_pipeline".to_string())
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
    fn test_pipeline_with_processor() {
        // Test the processor-based factory function
        let model = EmbedScoreModel::new(Box::new(MockEmbedder), Box::new(MockScorer));
        let backend = super::backend::Backend::new(vec![Box::new(model)]);
        let handle = backend.handle();

        let processor = Box::new(processors::GcnScoreProcessor::new(handle, 1.0));
        let mut sink = create_pipeline(processor, "test".to_string());

        sink.on_transfer(5, &make_clause(2));
        let s = sink.select();
        assert_eq!(s, Some(5));

        drop(sink);
        // Backend worker exits after sink drops (data processing thread drops its handle)
    }
}
