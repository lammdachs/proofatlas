//! Configuration-specific data processors.
//!
//! Each (encoder, scorer) combination gets its own processor that knows
//! exactly what data to extract, what to cache, and how to score:
//!
//! | Encoder   | Scorer      | Processor               | Cache          |
//! |-----------|-------------|-------------------------|----------------|
//! | gcn       | mlp         | GcnScoreProcessor       | f32 scores     |
//! | gcn       | attention   | GcnEmbeddingProcessor   | Vec<f32> embs  |
//! | gcn       | transformer | GcnEmbeddingProcessor   | Vec<f32> embs  |
//! | sentence  | mlp         | SentenceScoreProcessor  | f32 scores     |
//! | sentence  | attention   | SentenceEmbeddingProcessor | Vec<f32> embs |
//! | sentence  | transformer | SentenceEmbeddingProcessor | Vec<f32> embs |
//! | features  | mlp         | FeaturesScoreProcessor  | f32 scores     |
//! | features  | attention   | FeaturesEmbeddingProcessor | Vec<f32> embs |
//! | features  | transformer | FeaturesEmbeddingProcessor | Vec<f32> embs |

use std::sync::Arc;

use indexmap::IndexMap;

use crate::logic::{Clause, Interner};
use crate::selection::ml::features::extract_clause_features;
use super::backend::BackendHandle;
use super::{softmax_sample, softmax_sample_vec, DataProcessor};

// =============================================================================
// GcnScoreProcessor — gcn_mlp
// =============================================================================

/// Score processor for GCN + MLP configurations.
///
/// Caches pre-softmax f32 scores. On transfer, submits `Arc<Clause>` to the
/// backend's `embed_score` model and caches the returned score.
pub struct GcnScoreProcessor {
    backend: BackendHandle,
    scores: IndexMap<usize, f32>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
}

impl GcnScoreProcessor {
    pub fn new(backend: BackendHandle, temperature: f32) -> Self {
        Self {
            backend,
            scores: IndexMap::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
        }
    }
}

impl DataProcessor for GcnScoreProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new(clause);
        match self.backend.submit_sync(self.request_id, "embed_score".to_string(), data) {
            Ok(resp) => {
                let score = *resp.data.downcast::<f32>().unwrap_or(Box::new(0.0f32));
                self.scores.insert(idx, score);
            }
            Err(_) => {
                self.scores.insert(idx, 0.0);
            }
        }
    }

    fn on_activate(&mut self, _idx: usize) {
        // Score is clause-intrinsic for MLP — already removed during select
    }

    fn on_simplify(&mut self, idx: usize) {
        self.scores.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
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
pub struct GcnEmbeddingProcessor {
    backend: BackendHandle,
    u_embeddings: IndexMap<usize, Vec<f32>>,
    p_embeddings: IndexMap<usize, Vec<f32>>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
}

impl GcnEmbeddingProcessor {
    pub fn new(backend: BackendHandle, temperature: f32) -> Self {
        Self {
            backend,
            u_embeddings: IndexMap::new(),
            p_embeddings: IndexMap::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
        }
    }
}

impl DataProcessor for GcnEmbeddingProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new(clause);
        match self.backend.submit_sync(self.request_id, "embed".to_string(), data) {
            Ok(resp) => {
                let emb = *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![]));
                self.u_embeddings.insert(idx, emb);
            }
            Err(_) => {
                self.u_embeddings.insert(idx, vec![]);
            }
        }
    }

    fn on_activate(&mut self, idx: usize) {
        if let Some(emb) = self.u_embeddings.shift_remove(&idx) {
            self.p_embeddings.insert(idx, emb);
        }
    }

    fn on_simplify(&mut self, idx: usize) {
        self.u_embeddings.shift_remove(&idx);
        self.p_embeddings.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        if self.u_embeddings.is_empty() {
            return None;
        }

        // Collect U and P embeddings for context scoring
        let u_embs: Vec<Vec<f32>> = self.u_embeddings.values().cloned().collect();
        let p_embs: Vec<Vec<f32>> = self.p_embeddings.values().cloned().collect();
        let u_indices: Vec<usize> = self.u_embeddings.keys().copied().collect();

        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new((u_embs, p_embs));
        let scores = match self.backend.submit_sync(self.request_id, "score_context".to_string(), data) {
            Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
            Err(_) => vec![0.0; u_indices.len()],
        };

        let selected = softmax_sample_vec(&u_indices, &scores, self.temperature, &mut self.rng_state);
        if let Some(idx) = selected {
            self.u_embeddings.shift_remove(&idx);
        }
        selected
    }
}

// =============================================================================
// SentenceScoreProcessor — sentence_mlp
// =============================================================================

/// Score processor for sentence + MLP configurations.
///
/// Like GcnScoreProcessor but converts clauses to strings before submitting.
pub struct SentenceScoreProcessor {
    backend: BackendHandle,
    interner: Arc<Interner>,
    scores: IndexMap<usize, f32>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
}

impl SentenceScoreProcessor {
    pub fn new(backend: BackendHandle, interner: Arc<Interner>, temperature: f32) -> Self {
        Self {
            backend,
            interner,
            scores: IndexMap::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
        }
    }
}

impl DataProcessor for SentenceScoreProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        self.request_id += 1;
        let s = clause.display(&self.interner).to_string();
        let data: Box<dyn std::any::Any + Send> = Box::new(s);
        match self.backend.submit_sync(self.request_id, "embed_score".to_string(), data) {
            Ok(resp) => {
                let score = *resp.data.downcast::<f32>().unwrap_or(Box::new(0.0f32));
                self.scores.insert(idx, score);
            }
            Err(_) => {
                self.scores.insert(idx, 0.0);
            }
        }
    }

    fn on_activate(&mut self, _idx: usize) {}

    fn on_simplify(&mut self, idx: usize) {
        self.scores.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
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
pub struct SentenceEmbeddingProcessor {
    backend: BackendHandle,
    interner: Arc<Interner>,
    u_embeddings: IndexMap<usize, Vec<f32>>,
    p_embeddings: IndexMap<usize, Vec<f32>>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
}

impl SentenceEmbeddingProcessor {
    pub fn new(backend: BackendHandle, interner: Arc<Interner>, temperature: f32) -> Self {
        Self {
            backend,
            interner,
            u_embeddings: IndexMap::new(),
            p_embeddings: IndexMap::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
        }
    }
}

impl DataProcessor for SentenceEmbeddingProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        self.request_id += 1;
        let s = clause.display(&self.interner).to_string();
        let data: Box<dyn std::any::Any + Send> = Box::new(s);
        match self.backend.submit_sync(self.request_id, "embed".to_string(), data) {
            Ok(resp) => {
                let emb = *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![]));
                self.u_embeddings.insert(idx, emb);
            }
            Err(_) => {
                self.u_embeddings.insert(idx, vec![]);
            }
        }
    }

    fn on_activate(&mut self, idx: usize) {
        if let Some(emb) = self.u_embeddings.shift_remove(&idx) {
            self.p_embeddings.insert(idx, emb);
        }
    }

    fn on_simplify(&mut self, idx: usize) {
        self.u_embeddings.shift_remove(&idx);
        self.p_embeddings.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        if self.u_embeddings.is_empty() {
            return None;
        }

        let u_embs: Vec<Vec<f32>> = self.u_embeddings.values().cloned().collect();
        let p_embs: Vec<Vec<f32>> = self.p_embeddings.values().cloned().collect();
        let u_indices: Vec<usize> = self.u_embeddings.keys().copied().collect();

        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new((u_embs, p_embs));
        let scores = match self.backend.submit_sync(self.request_id, "score_context".to_string(), data) {
            Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
            Err(_) => vec![0.0; u_indices.len()],
        };

        let selected = softmax_sample_vec(&u_indices, &scores, self.temperature, &mut self.rng_state);
        if let Some(idx) = selected {
            self.u_embeddings.shift_remove(&idx);
        }
        selected
    }
}

// =============================================================================
// FeaturesScoreProcessor — features_mlp
// =============================================================================

/// Score processor for features + MLP configurations.
///
/// Extracts 9 clause features and submits them as Vec<f32>.
pub struct FeaturesScoreProcessor {
    backend: BackendHandle,
    scores: IndexMap<usize, f32>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
}

impl FeaturesScoreProcessor {
    pub fn new(backend: BackendHandle, temperature: f32) -> Self {
        Self {
            backend,
            scores: IndexMap::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
        }
    }
}

impl DataProcessor for FeaturesScoreProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        self.request_id += 1;
        let features = extract_clause_features(&clause).to_vec();
        let data: Box<dyn std::any::Any + Send> = Box::new(features);
        match self.backend.submit_sync(self.request_id, "embed_score".to_string(), data) {
            Ok(resp) => {
                let score = *resp.data.downcast::<f32>().unwrap_or(Box::new(0.0f32));
                self.scores.insert(idx, score);
            }
            Err(_) => {
                self.scores.insert(idx, 0.0);
            }
        }
    }

    fn on_activate(&mut self, _idx: usize) {}

    fn on_simplify(&mut self, idx: usize) {
        self.scores.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
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
pub struct FeaturesEmbeddingProcessor {
    backend: BackendHandle,
    u_embeddings: IndexMap<usize, Vec<f32>>,
    p_embeddings: IndexMap<usize, Vec<f32>>,
    temperature: f32,
    rng_state: u64,
    request_id: u64,
}

impl FeaturesEmbeddingProcessor {
    pub fn new(backend: BackendHandle, temperature: f32) -> Self {
        Self {
            backend,
            u_embeddings: IndexMap::new(),
            p_embeddings: IndexMap::new(),
            temperature,
            rng_state: 0x12345678_9abcdef0,
            request_id: 0,
        }
    }
}

impl DataProcessor for FeaturesEmbeddingProcessor {
    fn on_transfer(&mut self, idx: usize, clause: Arc<Clause>) {
        self.request_id += 1;
        let features = extract_clause_features(&clause).to_vec();
        let data: Box<dyn std::any::Any + Send> = Box::new(features);
        match self.backend.submit_sync(self.request_id, "embed".to_string(), data) {
            Ok(resp) => {
                let emb = *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![]));
                self.u_embeddings.insert(idx, emb);
            }
            Err(_) => {
                self.u_embeddings.insert(idx, vec![]);
            }
        }
    }

    fn on_activate(&mut self, idx: usize) {
        if let Some(emb) = self.u_embeddings.shift_remove(&idx) {
            self.p_embeddings.insert(idx, emb);
        }
    }

    fn on_simplify(&mut self, idx: usize) {
        self.u_embeddings.shift_remove(&idx);
        self.p_embeddings.shift_remove(&idx);
    }

    fn select(&mut self) -> Option<usize> {
        if self.u_embeddings.is_empty() {
            return None;
        }

        let u_embs: Vec<Vec<f32>> = self.u_embeddings.values().cloned().collect();
        let p_embs: Vec<Vec<f32>> = self.p_embeddings.values().cloned().collect();
        let u_indices: Vec<usize> = self.u_embeddings.keys().copied().collect();

        self.request_id += 1;
        let data: Box<dyn std::any::Any + Send> = Box::new((u_embs, p_embs));
        let scores = match self.backend.submit_sync(self.request_id, "score_context".to_string(), data) {
            Ok(resp) => *resp.data.downcast::<Vec<f32>>().unwrap_or(Box::new(vec![])),
            Err(_) => vec![0.0; u_indices.len()],
        };

        let selected = softmax_sample_vec(&u_indices, &scores, self.temperature, &mut self.rng_state);
        if let Some(idx) = selected {
            self.u_embeddings.shift_remove(&idx);
        }
        selected
    }
}
