//! Sentence transformer clause selector
//!
//! This implements a sentence transformer for clause selection,
//! using PyTorch via tch-rs for inference. Weights are loaded from
//! TorchScript models exported from PyTorch training.
//!
//! Requires the `sentence` and `torch` features.

#[cfg(feature = "ml")]
use std::path::Path;

#[cfg(feature = "ml")]
use std::sync::Arc;

#[cfg(feature = "ml")]
use crate::logic::{Clause, Interner};

// =============================================================================
// Shared tokenization
// =============================================================================

/// Tokenize a batch of strings into padded (input_ids, attention_mask) tensors.
///
/// Shared between `SentenceEmbedder` (inference) and `MiniLMEncoderModel` (trace embedding).
#[cfg(feature = "ml")]
pub fn tokenize_batch(
    tokenizer: &tokenizers::Tokenizer,
    strings: &[String],
    max_length: usize,
    device: tch::Device,
) -> (tch::Tensor, tch::Tensor) {
    let encodings = tokenizer
        .encode_batch(strings.to_vec(), true)
        .expect("Tokenization failed");

    let batch_size = encodings.len();
    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0)
        .min(max_length);

    let mut all_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
    let mut all_masks: Vec<i64> = Vec::with_capacity(batch_size * max_len);

    for encoding in &encodings {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();

        let len = ids.len().min(max_len);
        for i in 0..len {
            all_ids.push(ids[i] as i64);
            all_masks.push(mask[i] as i64);
        }
        for _ in len..max_len {
            all_ids.push(0);
            all_masks.push(0);
        }
    }

    let ids_tensor = tch::Tensor::from_slice(&all_ids)
        .view([batch_size as i64, max_len as i64])
        .to_device(device);
    let mask_tensor = tch::Tensor::from_slice(&all_masks)
        .view([batch_size as i64, max_len as i64])
        .to_device(device);

    (ids_tensor, mask_tensor)
}

// =============================================================================
// SentenceEmbedder (inference scorer)
// =============================================================================

/// Sentence embedder using PyTorch for inference
///
/// Uses a TorchScript model that includes the encoder, projection, and scorer.
/// The model outputs scores directly, but we use it via the CachingSelector
/// which expects embeddings, so we output the projected embeddings instead.
#[cfg(feature = "ml")]
pub struct SentenceEmbedder {
    model: tch::CModule,
    tokenizer: tokenizers::Tokenizer,
    device: tch::Device,
    max_length: usize,
    /// Symbol interner for resolving clause IDs to symbol names.
    /// Set via `set_interner` before the saturation loop.
    interner: Option<Arc<Interner>>,
}

#[cfg(feature = "ml")]
impl SentenceEmbedder {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        use_cuda: bool,
    ) -> Result<Self, String> {
        let device = if use_cuda && tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        let model = tch::CModule::load_on_device(model_path.as_ref(), device)
            .map_err(|e| format!("Failed to load TorchScript model: {}", e))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            max_length: 128,
            interner: None,
        })
    }
}

#[cfg(feature = "ml")]
impl SentenceEmbedder {
    /// Core implementation: embed pre-serialized text strings.
    fn embed_strings(&self, clause_strings: &[&str]) -> Vec<Vec<f32>> {
        if clause_strings.is_empty() {
            return vec![];
        }

        let owned: Vec<String> = clause_strings.iter().map(|s| s.to_string()).collect();

        // Tokenize using shared function
        let (input_ids, attention_mask) =
            tokenize_batch(&self.tokenizer, &owned, self.max_length, self.device);

        // Run inference - model outputs scores directly
        let scores = tch::no_grad(|| {
            self.model
                .forward_ts(&[input_ids, attention_mask])
                .expect("Model forward failed")
        });

        // Convert scores to embeddings (1-dim per clause for caching)
        let scores_cpu = scores.to_device(tch::Device::Cpu).view([-1]);
        let scores_vec: Vec<f32> = Vec::<f32>::try_from(&scores_cpu)
            .expect("Failed to convert tensor to Vec<f32>");

        scores_vec.iter().map(|&s| vec![s]).collect()
    }
}

#[cfg(feature = "ml")]
impl crate::selection::cached::ClauseEmbedder for SentenceEmbedder {
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
        if clauses.is_empty() {
            return vec![];
        }

        // Convert clauses to strings using symbol names (not raw IDs)
        let clause_strings: Vec<String> = if let Some(ref interner) = self.interner {
            clauses.iter().map(|c| c.display(interner).to_string()).collect()
        } else {
            // Fallback to raw IDs if interner not set (should not happen in practice)
            clauses.iter().map(|c| c.to_string()).collect()
        };

        let refs: Vec<&str> = clause_strings.iter().map(|s| s.as_str()).collect();
        self.embed_strings(&refs)
    }

    fn embed_texts(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.embed_strings(texts)
    }

    fn embedding_dim(&self) -> usize {
        1  // We return scores as 1-dim embeddings
    }

    fn name(&self) -> &str {
        "sentence"
    }

    fn set_interner(&mut self, interner: Arc<Interner>) {
        self.interner = Some(interner);
    }
}

/// Simple pass-through scorer for SentenceEmbedder
///
/// Since the TorchScript model already computes scores, this scorer just returns them.
#[cfg(feature = "ml")]
pub struct PassThroughScorer;

#[cfg(feature = "ml")]
impl crate::selection::cached::EmbeddingScorer for PassThroughScorer {
    fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
        // Embeddings are already scores (1-element each)
        embeddings.iter().map(|e| e[0]).collect()
    }

    fn name(&self) -> &str {
        "pass_through"
    }
}

// =============================================================================
// SentenceEncoder — encoder-only, returns hidden_dim embeddings
// =============================================================================

/// Sentence encoder that returns real embeddings (not scores).
///
/// Loads an encoder-only TorchScript model (exported from _export_sentence_encoder).
/// Returns `[N, hidden_dim]` embeddings. Used with `TorchScriptScorer` for
/// separated encoder/scorer architecture (attention/transformer scorers).
#[cfg(feature = "ml")]
pub struct SentenceEncoder {
    model: tch::CModule,
    tokenizer: tokenizers::Tokenizer,
    device: tch::Device,
    max_length: usize,
    hidden_dim: usize,
    interner: Option<Arc<Interner>>,
}

#[cfg(feature = "ml")]
impl SentenceEncoder {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        hidden_dim: usize,
        use_cuda: bool,
    ) -> Result<Self, String> {
        let device = if use_cuda && tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        let model = tch::CModule::load_on_device(model_path.as_ref(), device)
            .map_err(|e| format!("Failed to load TorchScript sentence encoder: {}", e))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            max_length: 128,
            hidden_dim,
            interner: None,
        })
    }

    fn embed_strings(&self, clause_strings: &[&str]) -> Vec<Vec<f32>> {
        if clause_strings.is_empty() {
            return vec![];
        }

        let owned: Vec<String> = clause_strings.iter().map(|s| s.to_string()).collect();
        let (input_ids, attention_mask) =
            tokenize_batch(&self.tokenizer, &owned, self.max_length, self.device);

        let embeddings = tch::no_grad(|| {
            self.model
                .forward_ts(&[input_ids, attention_mask])
                .expect("Sentence encoder forward failed")
        });

        let emb_cpu = embeddings.to_device(tch::Device::Cpu);
        let shape = emb_cpu.size();
        let n = shape[0] as usize;
        let dim = if shape.len() > 1 { shape[1] as usize } else { 1 };
        let flat: Vec<f32> =
            Vec::<f32>::try_from(&emb_cpu.view([-1])).expect("Failed to convert tensor");

        flat.chunks(dim).take(n).map(|c| c.to_vec()).collect()
    }
}

#[cfg(feature = "ml")]
impl crate::selection::cached::ClauseEmbedder for SentenceEncoder {
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
        if clauses.is_empty() {
            return vec![];
        }

        let clause_strings: Vec<String> = if let Some(ref interner) = self.interner {
            clauses.iter().map(|c| c.display(interner).to_string()).collect()
        } else {
            clauses.iter().map(|c| c.to_string()).collect()
        };

        let refs: Vec<&str> = clause_strings.iter().map(|s| s.as_str()).collect();
        self.embed_strings(&refs)
    }

    fn embed_texts(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.embed_strings(texts)
    }

    fn embedding_dim(&self) -> usize {
        self.hidden_dim
    }

    fn name(&self) -> &str {
        "sentence_encoder"
    }

    fn set_interner(&mut self, interner: Arc<Interner>) {
        self.interner = Some(interner);
    }
}

/// Sentence selector with GPU acceleration
#[cfg(feature = "ml")]
pub type SentenceSelector = crate::selection::cached::CachingSelector<SentenceEmbedder, PassThroughScorer>;

/// Load sentence selector
#[cfg(feature = "ml")]
pub fn load_sentence_selector<P: AsRef<Path>>(
    model_path: P,
    tokenizer_path: P,
    use_cuda: bool,
) -> Result<SentenceSelector, String> {
    let embedder = SentenceEmbedder::new(&model_path, &tokenizer_path, use_cuda)?;
    let scorer = PassThroughScorer;
    Ok(crate::selection::cached::CachingSelector::new(embedder, scorer))
}

/// Load a standalone sentence embedder (for use with ScoringServer).
#[cfg(feature = "ml")]
pub fn load_sentence_embedder<P: AsRef<Path>>(
    model_path: P,
    tokenizer_path: P,
    use_cuda: bool,
) -> Result<SentenceEmbedder, String> {
    SentenceEmbedder::new(model_path, tokenizer_path, use_cuda)
}

// =============================================================================
// MiniLMEncoderModel — base MiniLM for trace embedding via Backend
// =============================================================================

/// Base MiniLM encoder model for pre-computing embeddings at trace time.
///
/// Unlike `SentenceEmbedder` (which wraps a trained scorer model), this loads
/// the frozen base MiniLM and returns raw 384-D embeddings. Used via the
/// Backend to embed clause strings and node names before writing NPZ traces.
#[cfg(feature = "ml")]
pub struct MiniLMEncoderModel {
    model: tch::CModule,
    tokenizer: tokenizers::Tokenizer,
    device: tch::Device,
    max_length: usize,
}

#[cfg(feature = "ml")]
impl MiniLMEncoderModel {
    pub fn new(model_path: &str, tokenizer_path: &str, use_cuda: bool) -> Result<Self, String> {
        let device = if use_cuda && tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        let model = tch::CModule::load_on_device(model_path, device)
            .map_err(|e| format!("Failed to load base MiniLM model: {}", e))?;

        // The tokenizer directory should contain tokenizer.json
        let tok_json = std::path::PathBuf::from(tokenizer_path).join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_json)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            max_length: 128,
        })
    }

    /// Encode strings to 384-D embeddings.
    ///
    /// Mini-batches internally to bound peak memory. MiniLM's self-attention
    /// allocates O(batch * heads * seq * seq) intermediate tensors — encoding
    /// thousands of strings at once easily exhausts RAM on constrained systems.
    pub fn encode_strings(&self, strings: &[String]) -> Vec<Vec<f32>> {
        if strings.is_empty() {
            return vec![];
        }

        // Cap batch size to keep MiniLM attention memory bounded.
        // At max_length=128, batch=64: attention ≈ 64*12*128*128*4 ≈ 50 MB.
        const MAX_BATCH: usize = 64;

        let mut all_results: Vec<Vec<f32>> = Vec::with_capacity(strings.len());

        for chunk in strings.chunks(MAX_BATCH) {
            let (input_ids, attention_mask) =
                tokenize_batch(&self.tokenizer, chunk, self.max_length, self.device);

            let embeddings = tch::no_grad(|| {
                self.model
                    .forward_ts(&[input_ids, attention_mask])
                    .expect("MiniLM forward failed")
            });

            let emb_cpu = embeddings.to_device(tch::Device::Cpu);
            let shape = emb_cpu.size();
            let n = shape[0] as usize;
            let dim = shape[1] as usize;
            let flat: Vec<f32> =
                Vec::<f32>::try_from(&emb_cpu.view([-1])).expect("Failed to convert tensor");

            all_results.extend(
                flat.chunks_exact(dim)
                    .take(n)
                    .map(|chunk| chunk.to_vec()),
            );
        }

        all_results
    }
}

#[cfg(feature = "ml")]
impl crate::selection::pipeline::backend::Model for MiniLMEncoderModel {
    fn model_id(&self) -> &str {
        "minilm"
    }

    fn execute_batch(
        &mut self,
        requests: Vec<(u64, Box<dyn std::any::Any + Send>)>,
    ) -> Vec<(u64, Box<dyn std::any::Any + Send>)> {
        // Collect all strings from requests
        let mut ids = Vec::with_capacity(requests.len());
        let mut all_strings = Vec::new();
        let mut counts = Vec::new();

        for (id, data) in &requests {
            ids.push(*id);
            if let Some(strings) = data.downcast_ref::<Vec<String>>() {
                counts.push(strings.len());
                all_strings.extend(strings.iter().cloned());
            } else {
                counts.push(0);
            }
        }

        // Batch encode all strings at once
        let all_embeddings = self.encode_strings(&all_strings);

        // Split results back per request
        let mut results = Vec::with_capacity(ids.len());
        let mut offset = 0;
        for (i, id) in ids.into_iter().enumerate() {
            let count = counts[i];
            let embs: Vec<Vec<f32>> = all_embeddings[offset..offset + count].to_vec();
            offset += count;
            results.push((id, Box::new(embs) as Box<dyn std::any::Any + Send>));
        }
        results
    }
}

#[cfg(test)]
#[cfg(feature = "ml")]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_selector_creation() {
        // Skip if model doesn't exist
        let model_path = std::path::Path::new(".weights/sentence_encoder.pt");
        let tokenizer_path = std::path::Path::new(".weights/sentence_tokenizer/tokenizer.json");

        if !model_path.exists() || !tokenizer_path.exists() {
            println!("Skipping test: sentence model not found");
            return;
        }

        let selector = load_sentence_selector(model_path, tokenizer_path, false);
        assert!(selector.is_ok(), "Failed to create selector: {:?}", selector);
    }
}
