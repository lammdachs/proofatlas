//! Configuration-specific data processors.
//!
//! Each (encoder, scorer) combination gets its own processor that knows
//! exactly what data to extract, what to cache, and how to score.
//!
//! Processors specify per-request device preferences via `use_cuda` on
//! `BackendHandle::submit_sync`. The Backend lazily loads models on the
//! requested device on first use.
//!
//! | Encoder   | Scorer      | Processor               | Cache          | Embed device | Score device |
//! |-----------|-------------|-------------------------|----------------|-------------|--------------|
//! | gcn       | mlp         | GcnScoreProcessor       | f32 scores     | CPU         | (fused)      |
//! | gcn       | attention   | GcnEmbeddingProcessor   | Vec<f32> embs  | CPU         | use_cuda     |
//! | gcn       | transformer | GcnEmbeddingProcessor   | Vec<f32> embs  | CPU         | use_cuda     |
//! | sentence  | mlp         | SentenceScoreProcessor  | f32 scores     | use_cuda    | (fused)      |
//! | sentence  | attention   | SentenceEmbeddingProcessor | Vec<f32> embs | use_cuda  | use_cuda     |
//! | sentence  | transformer | SentenceEmbeddingProcessor | Vec<f32> embs | use_cuda  | use_cuda     |
//! | features  | mlp         | FeaturesScoreProcessor  | f32 scores     | CPU         | (fused)      |
//! | features  | attention   | FeaturesEmbeddingProcessor | Vec<f32> embs | CPU       | use_cuda     |
//! | features  | transformer | FeaturesEmbeddingProcessor | Vec<f32> embs | CPU       | use_cuda     |

use std::sync::Arc;

use indexmap::IndexMap;

use crate::logic::{Clause, Interner};
use crate::selection::ml::features::extract_clause_features;
use super::backend::BackendHandle;
use super::{softmax_sample, softmax_sample_vec, DataProcessor};

/// How embed requests are scheduled relative to the prover's inner loop.
///
/// * `Async`: `on_transfer` calls `submit_async` immediately; the backend can
///   batch and process while the prover continues with backward simplification.
///   Drain happens at the next score-read point (typically `select`).
/// * `Sequential`: `on_transfer` calls `submit_async` then drains inline ---
///   each new clause incurs its own backend round trip. (Strawman sync.)
/// * `Deferred`: `on_transfer` only buffers the input; at the start of
///   `select`, all buffered inputs are submitted in rapid succession so the
///   backend batches them into a single forward pass, then drained.
///   (Thoughtful sync: captures the batching benefit but loses the overlap.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    Async,
    Sequential,
    Deferred,
}

// =============================================================================
// GcnScoreProcessor — gcn_mlp
// =============================================================================

/// Score processor for GCN + MLP configurations.
///
/// Caches pre-softmax f32 scores. On transfer, submits `Arc<Clause>` to the
/// backend's `embed_score` model and caches the returned score.
///
/// Device: GCN encoding is lightweight → always CPU.
pub struct GcnScoreProcessor {
    backend: BackendHandle,
    embed_cuda: bool,
    scores: IndexMap<usize, f32>,
    /// Pending embed+score requests: (clause_idx, request_handle).
    /// Drained before any read of `scores` (select, on_simplify).
    pending: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)>,
    /// Buffered inputs for `Deferred` mode: (clause_idx, prepared input).
    /// Fired as one rapid batch of submit_async calls at start of `select`.
    deferred: Vec<(usize, Box<dyn std::any::Any + Send>)>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
    mode: InferenceMode,
}

impl GcnScoreProcessor {
    pub fn new(backend: BackendHandle, temperature: f32, embed_cuda: bool, mode: InferenceMode) -> Self {
        Self {
            backend,
            embed_cuda,
            scores: IndexMap::new(),
            pending: Vec::new(),
            deferred: Vec::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
            mode,
        }
    }

    /// Drain all pending backend responses into `scores`.
    fn drain_pending(&mut self) {
        for (idx, handle) in self.pending.drain(..) {
            let score = match handle.recv() {
                Ok(resp) => *resp.data.downcast::<f32>().unwrap_or(Box::new(0.0f32)),
                Err(_) => 0.0,
            };
            self.scores.insert(idx, score);
        }
    }

    /// In `Deferred` mode, fire all buffered submits at once so the backend
    /// can batch them.
    fn fire_deferred(&mut self) {
        for (idx, data) in std::mem::take(&mut self.deferred).into_iter() {
            self.request_id += 1;
            match self.backend.submit_async(
                self.request_id, "embed_score".to_string(), data, self.embed_cuda,
            ) {
                Ok(handle) => self.pending.push((idx, handle)),
                Err(_) => { self.scores.insert(idx, 0.0); }
            }
        }
    }
}

impl DataProcessor for GcnScoreProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        let data: Box<dyn std::any::Any + Send> = Box::new(clause);
        if self.mode == InferenceMode::Deferred {
            self.deferred.push((idx, data));
            return;
        }
        self.request_id += 1;
        match self.backend.submit_async(
            self.request_id, "embed_score".to_string(), data, self.embed_cuda,
        ) {
            Ok(handle) => {
                self.pending.push((idx, handle));
                if self.mode == InferenceMode::Sequential { self.drain_pending(); }
            }
            Err(_) => { self.scores.insert(idx, 0.0); }
        }
    }

    fn on_activate(&mut self, _idx: usize) {
        // Score is clause-intrinsic for MLP — already removed during select
    }

    fn on_simplify(&mut self, idx: usize) {
        if self.mode == InferenceMode::Deferred {
            self.deferred.retain(|(i, _)| *i != idx);
        }
        self.drain_pending();
        self.scores.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        self.fire_deferred();
        self.drain_pending();
        let selected = softmax_sample(&self.scores, self.temperature, &mut self.rng_state);
        if let Some(idx) = selected {
            self.scores.shift_remove(&idx);
        }
        selected
    }
}

// =============================================================================
// GcnEmbeddingProcessor — gcn_attention, gcn_transformer
// =============================================================================

/// Embedding processor for GCN + attention/transformer configurations.
///
/// Caches Vec<f32> embeddings separately for U and P. On select, submits all
/// U+P embeddings to the backend's `score_context` model for re-scoring.
///
/// Device: GCN encoder on `embed_cuda` (typically CPU),
/// attention/transformer scorer on `score_cuda` (typically GPU).
pub struct GcnEmbeddingProcessor {
    backend: BackendHandle,
    embed_cuda: bool,
    score_cuda: bool,
    /// If true (default), embeddings are computed once on transfer and reused
    /// across many select() calls. If false, embeddings are re-computed
    /// for all current U and P clauses at every select() — an ablation that
    /// stresses the backend much harder.
    cache_embeddings: bool,
    /// Cached embeddings (only used when cache_embeddings=true).
    u_embeddings: IndexMap<usize, Vec<f32>>,
    p_embeddings: IndexMap<usize, Vec<f32>>,
    /// Clauses tracked for uncached mode (only used when cache_embeddings=false).
    u_clauses: IndexMap<usize, Arc<Clause>>,
    p_clauses: IndexMap<usize, Arc<Clause>>,
    /// Pending embed requests: (clause_idx, request_handle).
    /// Drained before any read of u_embeddings (select, on_activate, on_simplify).
    pending: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)>,
    /// Buffered inputs for `Deferred` mode.
    deferred: Vec<(usize, Box<dyn std::any::Any + Send>)>,
    /// Async mode: buffer of submits not yet sent to the backend.
    /// When the buffer fills to `embed_batch_size`, all are fired in a tight
    /// burst (so the backend's try_recv loop catches them as one fat batch).
    to_submit: Vec<(usize, Box<dyn std::any::Any + Send>)>,
    /// Minimum clauses to accumulate before firing the burst in async mode.
    /// `1` means: fire immediately on each on_transfer (the original behavior).
    /// Any leftover < batch_size is flushed on drain (e.g. at select).
    embed_batch_size: usize,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
    mode: InferenceMode,
}

impl GcnEmbeddingProcessor {
    pub fn new(backend: BackendHandle, temperature: f32, embed_cuda: bool, score_cuda: bool, mode: InferenceMode) -> Self {
        Self::new_full(backend, temperature, embed_cuda, score_cuda, mode, true, 1)
    }

    pub fn new_with_cache(backend: BackendHandle, temperature: f32, embed_cuda: bool, score_cuda: bool, mode: InferenceMode, cache_embeddings: bool) -> Self {
        Self::new_full(backend, temperature, embed_cuda, score_cuda, mode, cache_embeddings, 1)
    }

    pub fn new_full(backend: BackendHandle, temperature: f32, embed_cuda: bool, score_cuda: bool, mode: InferenceMode, cache_embeddings: bool, embed_batch_size: usize) -> Self {
        Self {
            backend,
            embed_cuda,
            score_cuda,
            cache_embeddings,
            u_embeddings: IndexMap::new(),
            p_embeddings: IndexMap::new(),
            u_clauses: IndexMap::new(),
            p_clauses: IndexMap::new(),
            pending: Vec::new(),
            deferred: Vec::new(),
            to_submit: Vec::new(),
            embed_batch_size: embed_batch_size.max(1),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
            mode,
        }
    }

    /// Fire all currently-buffered submits at the backend in one tight burst.
    /// Caller drains the resulting handles via `drain_pending` when scores are needed.
    fn fire_buffer(&mut self) {
        for (idx, data) in std::mem::take(&mut self.to_submit).into_iter() {
            self.request_id += 1;
            match self.backend.submit_async(
                self.request_id, "embed".to_string(), data, self.embed_cuda,
            ) {
                Ok(handle) => self.pending.push((idx, handle)),
                Err(_) => { self.u_embeddings.insert(idx, vec![]); }
            }
        }
    }

    /// Drain all pending embed responses into u_embeddings.
    fn drain_pending(&mut self) {
        for (idx, handle) in self.pending.drain(..) {
            let emb = match handle.recv() {
                Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
                Err(_) => vec![],
            };
            self.u_embeddings.insert(idx, emb);
        }
    }

    fn fire_deferred(&mut self) {
        for (idx, data) in std::mem::take(&mut self.deferred).into_iter() {
            self.request_id += 1;
            match self.backend.submit_async(
                self.request_id, "embed".to_string(), data, self.embed_cuda,
            ) {
                Ok(handle) => self.pending.push((idx, handle)),
                Err(_) => { self.u_embeddings.insert(idx, vec![]); }
            }
        }
    }
}

impl DataProcessor for GcnEmbeddingProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        if !self.cache_embeddings {
            // Uncached: just track the clause; embedding happens at select.
            self.u_clauses.insert(idx, clause);
            return;
        }
        // Eagerly build the per-clause graph + features on this thread
        // (the processor worker thread, in parallel with the prover's
        // continued simp loop). The backend only has to concatenate and
        // run the forward pass.
        let prebuilt = crate::selection::ml::graph::PrebuiltGcnInput {
            graph: crate::selection::ml::graph::GraphBuilder::build_one(&clause),
            clause_features: crate::selection::ml::features::extract_clause_features(&clause),
        };
        let data: Box<dyn std::any::Any + Send> = Box::new(prebuilt);
        if self.mode == InferenceMode::Deferred {
            self.deferred.push((idx, data));
            return;
        }
        if self.mode == InferenceMode::Async && self.embed_batch_size > 1 {
            // Buffer until we hit the threshold, then fire the burst.
            self.to_submit.push((idx, data));
            if self.to_submit.len() >= self.embed_batch_size {
                self.fire_buffer();
            }
            return;
        }
        self.request_id += 1;
        match self.backend.submit_async(
            self.request_id, "embed".to_string(), data, self.embed_cuda,
        ) {
            Ok(handle) => {
                self.pending.push((idx, handle));
                if self.mode == InferenceMode::Sequential { self.drain_pending(); }
            }
            Err(_) => { self.u_embeddings.insert(idx, vec![]); }
        }
    }

    fn on_activate(&mut self, idx: usize) {
        if !self.cache_embeddings {
            if let Some(c) = self.u_clauses.shift_remove(&idx) {
                self.p_clauses.insert(idx, c);
            }
            return;
        }
        self.fire_deferred();
        self.fire_buffer();
        self.drain_pending();
        if let Some(emb) = self.u_embeddings.shift_remove(&idx) {
            self.p_embeddings.insert(idx, emb);
        }
    }

    fn on_simplify(&mut self, idx: usize) {
        if !self.cache_embeddings {
            self.u_clauses.shift_remove(&idx);
            self.p_clauses.shift_remove(&idx);
            return;
        }
        if self.mode == InferenceMode::Deferred {
            self.deferred.retain(|(i, _)| *i != idx);
        }
        // Drop buffered submit if the clause is being removed before firing.
        self.to_submit.retain(|(i, _)| *i != idx);
        self.drain_pending();
        self.u_embeddings.shift_remove(&idx);
        self.p_embeddings.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        if !self.cache_embeddings {
            // Uncached: re-embed all current U and P clauses at every select.
            self.u_embeddings.clear();
            self.p_embeddings.clear();
            // Fire all submits (mode controls whether each waits or batches at backend).
            let mut pending_u: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)> = Vec::new();
            let mut pending_p: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)> = Vec::new();
            let u_jobs: Vec<(usize, Arc<Clause>)> = self.u_clauses.iter().map(|(i,c)| (*i, c.clone())).collect();
            let p_jobs: Vec<(usize, Arc<Clause>)> = self.p_clauses.iter().map(|(i,c)| (*i, c.clone())).collect();
            for (idx, clause) in u_jobs {
                self.request_id += 1;
                let data: Box<dyn std::any::Any + Send> = Box::new(clause);
                match self.backend.submit_async(self.request_id, "embed".to_string(), data, self.embed_cuda) {
                    Ok(handle) => {
                        pending_u.push((idx, handle));
                        if self.mode == InferenceMode::Sequential {
                            if let Some((idx, handle)) = pending_u.pop() {
                                let emb = handle.recv().map(|r| *r.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![]))).unwrap_or_default();
                                self.u_embeddings.insert(idx, emb);
                            }
                        }
                    }
                    Err(_) => { self.u_embeddings.insert(idx, vec![]); }
                }
            }
            for (idx, clause) in p_jobs {
                self.request_id += 1;
                let data: Box<dyn std::any::Any + Send> = Box::new(clause);
                match self.backend.submit_async(self.request_id, "embed".to_string(), data, self.embed_cuda) {
                    Ok(handle) => {
                        pending_p.push((idx, handle));
                        if self.mode == InferenceMode::Sequential {
                            if let Some((idx, handle)) = pending_p.pop() {
                                let emb = handle.recv().map(|r| *r.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![]))).unwrap_or_default();
                                self.p_embeddings.insert(idx, emb);
                            }
                        }
                    }
                    Err(_) => { self.p_embeddings.insert(idx, vec![]); }
                }
            }
            // Drain remaining (async/deferred — fat batches at backend)
            for (idx, handle) in pending_u.drain(..) {
                let emb = handle.recv().map(|r| *r.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![]))).unwrap_or_default();
                self.u_embeddings.insert(idx, emb);
            }
            for (idx, handle) in pending_p.drain(..) {
                let emb = handle.recv().map(|r| *r.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![]))).unwrap_or_default();
                self.p_embeddings.insert(idx, emb);
            }
            if self.u_embeddings.is_empty() {
                return None;
            }
            // Fall through to existing score_context + sample (uses self.u_embeddings/p_embeddings).
        } else {
            self.fire_deferred();
            self.fire_buffer();
            self.drain_pending();
            if self.u_embeddings.is_empty() {
                return None;
            }
        }

        // Collect U and P embeddings for context scoring
        let u_embs: Vec<Vec<f32>> = self.u_embeddings.values().cloned().collect();
        let p_embs: Vec<Vec<f32>> = self.p_embeddings.values().cloned().collect();
        let u_indices: Vec<usize> = self.u_embeddings.keys().copied().collect();

        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new((u_embs, p_embs));
        let scores = match self.backend.submit_sync(self.request_id, "score_context".to_string(), data, self.score_cuda) {
            Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
            Err(_) => vec![0.0; u_indices.len()],
        };

        let selected = softmax_sample_vec(&u_indices, &scores, self.temperature, &mut self.rng_state);
        // Don't remove from u_embeddings here — on_activate will move it to p_embeddings
        selected
    }
}

// =============================================================================
// SentenceScoreProcessor — sentence_mlp
// =============================================================================

/// Score processor for sentence + MLP configurations.
///
/// Like GcnScoreProcessor but converts clauses to strings before submitting.
///
/// Device: Sentence encoding (MiniLM 33M params) → follows `embed_cuda`.
pub struct SentenceScoreProcessor {
    backend: BackendHandle,
    interner: Arc<Interner>,
    embed_cuda: bool,
    scores: IndexMap<usize, f32>,
    pending: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)>,
    deferred: Vec<(usize, Box<dyn std::any::Any + Send>)>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
    mode: InferenceMode,
}

impl SentenceScoreProcessor {
    pub fn new(backend: BackendHandle, interner: Arc<Interner>, temperature: f32, embed_cuda: bool, mode: InferenceMode) -> Self {
        Self {
            backend,
            interner,
            embed_cuda,
            scores: IndexMap::new(),
            pending: Vec::new(),
            deferred: Vec::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
            mode,
        }
    }

    fn drain_pending(&mut self) {
        for (idx, handle) in self.pending.drain(..) {
            let score = match handle.recv() {
                Ok(resp) => *resp.data.downcast::<f32>().unwrap_or(Box::new(0.0f32)),
                Err(_) => 0.0,
            };
            self.scores.insert(idx, score);
        }
    }

    fn fire_deferred(&mut self) {
        for (idx, data) in std::mem::take(&mut self.deferred).into_iter() {
            self.request_id += 1;
            match self.backend.submit_async(
                self.request_id, "embed_score".to_string(), data, self.embed_cuda,
            ) {
                Ok(handle) => self.pending.push((idx, handle)),
                Err(_) => { self.scores.insert(idx, 0.0); }
            }
        }
    }
}

impl DataProcessor for SentenceScoreProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        let s = clause.display(&self.interner).to_string();
        let data: Box<dyn std::any::Any + Send> = Box::new(s);
        if self.mode == InferenceMode::Deferred {
            self.deferred.push((idx, data));
            return;
        }
        self.request_id += 1;
        match self.backend.submit_async(
            self.request_id, "embed_score".to_string(), data, self.embed_cuda,
        ) {
            Ok(handle) => {
                self.pending.push((idx, handle));
                if self.mode == InferenceMode::Sequential { self.drain_pending(); }
            }
            Err(_) => { self.scores.insert(idx, 0.0); }
        }
    }

    fn on_activate(&mut self, _idx: usize) {}

    fn on_simplify(&mut self, idx: usize) {
        if self.mode == InferenceMode::Deferred {
            self.deferred.retain(|(i, _)| *i != idx);
        }
        self.drain_pending();
        self.scores.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        self.fire_deferred();
        self.drain_pending();
        let selected = softmax_sample(&self.scores, self.temperature, &mut self.rng_state);
        if let Some(idx) = selected {
            self.scores.shift_remove(&idx);
        }
        selected
    }
}

// =============================================================================
// SentenceEmbeddingProcessor — sentence_attention, sentence_transformer
// =============================================================================

/// Embedding processor for sentence + attention/transformer configurations.
///
/// Like GcnEmbeddingProcessor but converts clauses to strings before submitting.
///
/// Device: Both sentence encoder and scorer follow `use_cuda`.
pub struct SentenceEmbeddingProcessor {
    backend: BackendHandle,
    interner: Arc<Interner>,
    embed_cuda: bool,
    score_cuda: bool,
    u_embeddings: IndexMap<usize, Vec<f32>>,
    p_embeddings: IndexMap<usize, Vec<f32>>,
    pending: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)>,
    deferred: Vec<(usize, Box<dyn std::any::Any + Send>)>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
    mode: InferenceMode,
}

impl SentenceEmbeddingProcessor {
    pub fn new(backend: BackendHandle, interner: Arc<Interner>, temperature: f32, embed_cuda: bool, score_cuda: bool, mode: InferenceMode) -> Self {
        Self {
            backend,
            interner,
            embed_cuda,
            score_cuda,
            u_embeddings: IndexMap::new(),
            p_embeddings: IndexMap::new(),
            pending: Vec::new(),
            deferred: Vec::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
            mode,
        }
    }

    fn drain_pending(&mut self) {
        for (idx, handle) in self.pending.drain(..) {
            let emb = match handle.recv() {
                Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
                Err(_) => vec![],
            };
            self.u_embeddings.insert(idx, emb);
        }
    }

    fn fire_deferred(&mut self) {
        for (idx, data) in std::mem::take(&mut self.deferred).into_iter() {
            self.request_id += 1;
            match self.backend.submit_async(
                self.request_id, "embed".to_string(), data, self.embed_cuda,
            ) {
                Ok(handle) => self.pending.push((idx, handle)),
                Err(_) => { self.u_embeddings.insert(idx, vec![]); }
            }
        }
    }
}

impl DataProcessor for SentenceEmbeddingProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        let s = clause.display(&self.interner).to_string();
        let data: Box<dyn std::any::Any + Send> = Box::new(s);
        if self.mode == InferenceMode::Deferred {
            self.deferred.push((idx, data));
            return;
        }
        self.request_id += 1;
        match self.backend.submit_async(
            self.request_id, "embed".to_string(), data, self.embed_cuda,
        ) {
            Ok(handle) => {
                self.pending.push((idx, handle));
                if self.mode == InferenceMode::Sequential { self.drain_pending(); }
            }
            Err(_) => { self.u_embeddings.insert(idx, vec![]); }
        }
    }

    fn on_activate(&mut self, idx: usize) {
        self.fire_deferred();
        self.drain_pending();
        if let Some(emb) = self.u_embeddings.shift_remove(&idx) {
            self.p_embeddings.insert(idx, emb);
        }
    }

    fn on_simplify(&mut self, idx: usize) {
        if self.mode == InferenceMode::Deferred {
            self.deferred.retain(|(i, _)| *i != idx);
        }
        self.drain_pending();
        self.u_embeddings.shift_remove(&idx);
        self.p_embeddings.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        self.fire_deferred();
        self.drain_pending();
        if self.u_embeddings.is_empty() {
            return None;
        }

        let u_embs: Vec<Vec<f32>> = self.u_embeddings.values().cloned().collect();
        let p_embs: Vec<Vec<f32>> = self.p_embeddings.values().cloned().collect();
        let u_indices: Vec<usize> = self.u_embeddings.keys().copied().collect();

        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new((u_embs, p_embs));
        let scores = match self.backend.submit_sync(self.request_id, "score_context".to_string(), data, self.score_cuda) {
            Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
            Err(_) => vec![0.0; u_indices.len()],
        };

        let selected = softmax_sample_vec(&u_indices, &scores, self.temperature, &mut self.rng_state);
        // Don't remove from u_embeddings here — on_activate will move it to p_embeddings
        selected
    }
}

// =============================================================================
// FeaturesScoreProcessor — features_mlp
// =============================================================================

/// Score processor for features + MLP configurations.
///
/// Extracts 9 clause features and submits them as Vec<f32>.
///
/// Device: Features encoding is lightweight → follows `embed_cuda` (typically CPU).
pub struct FeaturesScoreProcessor {
    backend: BackendHandle,
    embed_cuda: bool,
    scores: IndexMap<usize, f32>,
    pending: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)>,
    deferred: Vec<(usize, Box<dyn std::any::Any + Send>)>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
    mode: InferenceMode,
}

impl FeaturesScoreProcessor {
    pub fn new(backend: BackendHandle, temperature: f32, embed_cuda: bool, mode: InferenceMode) -> Self {
        Self {
            backend,
            embed_cuda,
            scores: IndexMap::new(),
            pending: Vec::new(),
            deferred: Vec::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
            mode,
        }
    }

    fn drain_pending(&mut self) {
        for (idx, handle) in self.pending.drain(..) {
            let score = match handle.recv() {
                Ok(resp) => *resp.data.downcast::<f32>().unwrap_or(Box::new(0.0f32)),
                Err(_) => 0.0,
            };
            self.scores.insert(idx, score);
        }
    }

    fn fire_deferred(&mut self) {
        for (idx, data) in std::mem::take(&mut self.deferred).into_iter() {
            self.request_id += 1;
            match self.backend.submit_async(
                self.request_id, "embed_score".to_string(), data, self.embed_cuda,
            ) {
                Ok(handle) => self.pending.push((idx, handle)),
                Err(_) => { self.scores.insert(idx, 0.0); }
            }
        }
    }
}

impl DataProcessor for FeaturesScoreProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        let features = extract_clause_features(&clause).to_vec();
        let data: Box<dyn std::any::Any + Send> = Box::new(features);
        if self.mode == InferenceMode::Deferred {
            self.deferred.push((idx, data));
            return;
        }
        self.request_id += 1;
        match self.backend.submit_async(
            self.request_id, "embed_score".to_string(), data, self.embed_cuda,
        ) {
            Ok(handle) => {
                self.pending.push((idx, handle));
                if self.mode == InferenceMode::Sequential { self.drain_pending(); }
            }
            Err(_) => { self.scores.insert(idx, 0.0); }
        }
    }

    fn on_activate(&mut self, _idx: usize) {}

    fn on_simplify(&mut self, idx: usize) {
        if self.mode == InferenceMode::Deferred {
            self.deferred.retain(|(i, _)| *i != idx);
        }
        self.drain_pending();
        self.scores.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        self.fire_deferred();
        self.drain_pending();
        let selected = softmax_sample(&self.scores, self.temperature, &mut self.rng_state);
        if let Some(idx) = selected {
            self.scores.shift_remove(&idx);
        }
        selected
    }
}

// =============================================================================
// FeaturesEmbeddingProcessor — features_attention, features_transformer
// =============================================================================

/// Embedding processor for features + attention/transformer configurations.
///
/// Extracts 9 clause features, submits to backend for embedding, caches embeddings.
///
/// Device: Features encoder on `embed_cuda` (typically CPU),
/// attention/transformer scorer on `score_cuda` (typically GPU).
pub struct FeaturesEmbeddingProcessor {
    backend: BackendHandle,
    embed_cuda: bool,
    score_cuda: bool,
    u_embeddings: IndexMap<usize, Vec<f32>>,
    p_embeddings: IndexMap<usize, Vec<f32>>,
    pending: Vec<(usize, crate::selection::pipeline::backend::RequestHandle)>,
    deferred: Vec<(usize, Box<dyn std::any::Any + Send>)>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
    mode: InferenceMode,
}

impl FeaturesEmbeddingProcessor {
    pub fn new(backend: BackendHandle, temperature: f32, embed_cuda: bool, score_cuda: bool, mode: InferenceMode) -> Self {
        Self {
            backend,
            embed_cuda,
            score_cuda,
            u_embeddings: IndexMap::new(),
            p_embeddings: IndexMap::new(),
            pending: Vec::new(),
            deferred: Vec::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
            mode,
        }
    }

    fn drain_pending(&mut self) {
        for (idx, handle) in self.pending.drain(..) {
            let emb = match handle.recv() {
                Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
                Err(_) => vec![],
            };
            self.u_embeddings.insert(idx, emb);
        }
    }

    fn fire_deferred(&mut self) {
        for (idx, data) in std::mem::take(&mut self.deferred).into_iter() {
            self.request_id += 1;
            match self.backend.submit_async(
                self.request_id, "embed".to_string(), data, self.embed_cuda,
            ) {
                Ok(handle) => self.pending.push((idx, handle)),
                Err(_) => { self.u_embeddings.insert(idx, vec![]); }
            }
        }
    }
}

impl DataProcessor for FeaturesEmbeddingProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        let features = extract_clause_features(&clause).to_vec();
        let data: Box<dyn std::any::Any + Send> = Box::new(features);
        if self.mode == InferenceMode::Deferred {
            self.deferred.push((idx, data));
            return;
        }
        self.request_id += 1;
        match self.backend.submit_async(
            self.request_id, "embed".to_string(), data, self.embed_cuda,
        ) {
            Ok(handle) => {
                self.pending.push((idx, handle));
                if self.mode == InferenceMode::Sequential { self.drain_pending(); }
            }
            Err(_) => { self.u_embeddings.insert(idx, vec![]); }
        }
    }

    fn on_activate(&mut self, idx: usize) {
        self.fire_deferred();
        self.drain_pending();
        if let Some(emb) = self.u_embeddings.shift_remove(&idx) {
            self.p_embeddings.insert(idx, emb);
        }
    }

    fn on_simplify(&mut self, idx: usize) {
        if self.mode == InferenceMode::Deferred {
            self.deferred.retain(|(i, _)| *i != idx);
        }
        self.drain_pending();
        self.u_embeddings.shift_remove(&idx);
        self.p_embeddings.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        self.fire_deferred();
        self.drain_pending();
        if self.u_embeddings.is_empty() {
            return None;
        }

        let u_embs: Vec<Vec<f32>> = self.u_embeddings.values().cloned().collect();
        let p_embs: Vec<Vec<f32>> = self.p_embeddings.values().cloned().collect();
        let u_indices: Vec<usize> = self.u_embeddings.keys().copied().collect();

        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new((u_embs, p_embs));
        let scores = match self.backend.submit_sync(self.request_id, "score_context".to_string(), data, self.score_cuda) {
            Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
            Err(_) => vec![0.0; u_indices.len()],
        };

        let selected = softmax_sample_vec(&u_indices, &scores, self.temperature, &mut self.rng_state);
        // Don't remove from u_embeddings here — on_activate will move it to p_embeddings
        selected
    }
}
