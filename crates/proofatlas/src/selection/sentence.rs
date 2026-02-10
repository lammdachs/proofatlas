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

    fn tokenize_clauses(&self, clause_strings: &[String]) -> (tch::Tensor, tch::Tensor) {
        let encodings = self
            .tokenizer
            .encode_batch(clause_strings.to_vec(), true)
            .expect("Tokenization failed");

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

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
            // Pad to max_len
            for _ in len..max_len {
                all_ids.push(0);
                all_masks.push(0);
            }
        }

        let ids_tensor = tch::Tensor::from_slice(&all_ids)
            .view([batch_size as i64, max_len as i64])
            .to_device(self.device);
        let mask_tensor = tch::Tensor::from_slice(&all_masks)
            .view([batch_size as i64, max_len as i64])
            .to_device(self.device);

        (ids_tensor, mask_tensor)
    }
}

#[cfg(feature = "ml")]
impl super::cached::ClauseEmbedder for SentenceEmbedder {
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

        // Tokenize
        let (input_ids, attention_mask) = self.tokenize_clauses(&clause_strings);

        // Run inference - model outputs scores directly
        let scores = tch::no_grad(|| {
            self.model
                .forward_ts(&[input_ids, attention_mask])
                .expect("Model forward failed")
        });

        // Convert scores to embeddings (1-dim per clause for caching)
        // The CachingSelector will use these as "embeddings" and pass them to the scorer
        // Since our TorchScript model already includes the scorer, we return the scores
        // wrapped as 1-element embeddings
        let scores_cpu = scores.to_device(tch::Device::Cpu).view([-1]);
        let scores_vec: Vec<f32> = Vec::<f32>::try_from(&scores_cpu)
            .expect("Failed to convert tensor to Vec<f32>");

        scores_vec.iter().map(|&s| vec![s]).collect()
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
impl super::cached::EmbeddingScorer for PassThroughScorer {
    fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
        // Embeddings are already scores (1-element each)
        embeddings.iter().map(|e| e[0]).collect()
    }

    fn name(&self) -> &str {
        "pass_through"
    }
}

/// Sentence selector with GPU acceleration
#[cfg(feature = "ml")]
pub type SentenceSelector = super::cached::CachingSelector<SentenceEmbedder, PassThroughScorer>;

/// Load sentence selector
#[cfg(feature = "ml")]
pub fn load_sentence_selector<P: AsRef<Path>>(
    model_path: P,
    tokenizer_path: P,
    use_cuda: bool,
) -> Result<SentenceSelector, String> {
    let embedder = SentenceEmbedder::new(&model_path, &tokenizer_path, use_cuda)?;
    let scorer = PassThroughScorer;
    Ok(super::cached::CachingSelector::new(embedder, scorer))
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
