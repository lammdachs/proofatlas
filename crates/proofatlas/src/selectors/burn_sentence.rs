//! Burn-based sentence encoder for clause selection
//!
//! This implements a BERT/MiniLM-style transformer encoder for clause selection,
//! using the Burn ML framework for inference. Weights are loaded from
//! safetensors files exported from HuggingFace transformers.
//!
//! The architecture exactly matches HuggingFace BERT for weight compatibility.
//!
//! Requires the `sentence` feature for tokenizer support.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;

#[cfg(feature = "sentence")]
use crate::core::Clause;
#[cfg(feature = "sentence")]
use std::path::Path;

// ============================================================================
// BERT-compatible attention (separate Q, K, V projections)
// ============================================================================

/// BERT-style self-attention with separate Q, K, V projections
///
/// Matches HuggingFace naming:
/// - self.query, self.key, self.value
/// - output.dense
#[derive(Module, Debug)]
pub struct BertSelfAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl<B: Backend> BertSelfAttention<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        Self {
            query: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            key: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            value: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden_dim] = x.dims();

        // Separate Q, K, V projections
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_heads, self.head_dim]);

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Attention scores: [batch, num_heads, seq_len, seq_len]
        let attn = q.matmul(k.swap_dims(2, 3)) * self.scale;

        // Apply mask if provided
        let attn = if let Some(mask) = mask {
            let mask = mask.reshape([batch, 1, 1, seq_len]);
            let neg_inf = Tensor::full(
                [batch, self.num_heads, seq_len, seq_len],
                f32::NEG_INFINITY,
                &attn.device(),
            );
            let mask_expanded = mask.repeat_dim(1, self.num_heads).repeat_dim(2, seq_len);
            attn.mask_where(mask_expanded.equal_elem(0.0), neg_inf)
        } else {
            attn
        };

        // Softmax
        let attn = burn::tensor::activation::softmax(attn, 3);

        // Apply attention to values
        let out = attn.matmul(v);

        // Transpose back: [batch, seq_len, num_heads, head_dim]
        let out = out.swap_dims(1, 2);

        // Reshape: [batch, seq_len, hidden_dim]
        let hidden_dim = self.num_heads * self.head_dim;
        out.reshape([batch, seq_len, hidden_dim])
    }
}

/// BERT attention output (dense + LayerNorm)
#[derive(Module, Debug)]
pub struct BertSelfOutput<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> BertSelfOutput<B> {
    pub fn new(device: &B::Device, hidden_dim: usize) -> Self {
        Self {
            dense: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            layer_norm: LayerNormConfig::new(hidden_dim).init(device),
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.dense.forward(hidden_states);
        self.layer_norm.forward(out + input)
    }
}

/// Complete BERT attention block
#[derive(Module, Debug)]
pub struct BertAttention<B: Backend> {
    self_attn: BertSelfAttention<B>,
    output: BertSelfOutput<B>,
}

impl<B: Backend> BertAttention<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            self_attn: BertSelfAttention::new(device, hidden_dim, num_heads),
            output: BertSelfOutput::new(device, hidden_dim),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let attn_out = self.self_attn.forward(x.clone(), mask);
        self.output.forward(attn_out, x)
    }
}

// ============================================================================
// BERT-compatible FFN (intermediate + output)
// ============================================================================

/// BERT intermediate layer
#[derive(Module, Debug)]
pub struct BertIntermediate<B: Backend> {
    dense: Linear<B>,
}

impl<B: Backend> BertIntermediate<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            dense: LinearConfig::new(hidden_dim, intermediate_dim).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        gelu(self.dense.forward(x))
    }
}

/// BERT output layer
#[derive(Module, Debug)]
pub struct BertOutput<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> BertOutput<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            dense: LinearConfig::new(intermediate_dim, hidden_dim).init(device),
            layer_norm: LayerNormConfig::new(hidden_dim).init(device),
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.dense.forward(hidden_states);
        self.layer_norm.forward(out + input)
    }
}

// ============================================================================
// BERT encoder layer
// ============================================================================

/// Single BERT encoder layer
///
/// Matches HuggingFace naming:
/// - attention.self.{query,key,value}
/// - attention.output.dense, attention.output.LayerNorm
/// - intermediate.dense
/// - output.dense, output.LayerNorm
#[derive(Module, Debug)]
pub struct BertLayer<B: Backend> {
    attention: BertAttention<B>,
    intermediate: BertIntermediate<B>,
    output: BertOutput<B>,
}

impl<B: Backend> BertLayer<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, num_heads: usize, intermediate_dim: usize) -> Self {
        Self {
            attention: BertAttention::new(device, hidden_dim, num_heads),
            intermediate: BertIntermediate::new(device, hidden_dim, intermediate_dim),
            output: BertOutput::new(device, hidden_dim, intermediate_dim),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let attn_out = self.attention.forward(x, mask);
        let intermediate_out = self.intermediate.forward(attn_out.clone());
        self.output.forward(intermediate_out, attn_out)
    }
}

// ============================================================================
// BERT embeddings
// ============================================================================

/// BERT embeddings layer
///
/// Matches HuggingFace naming:
/// - word_embeddings
/// - position_embeddings
/// - token_type_embeddings
/// - LayerNorm
#[derive(Module, Debug)]
pub struct BertEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> BertEmbeddings<B> {
    pub fn new(
        device: &B::Device,
        vocab_size: usize,
        hidden_dim: usize,
        max_position_embeddings: usize,
        type_vocab_size: usize,
    ) -> Self {
        Self {
            word_embeddings: EmbeddingConfig::new(vocab_size, hidden_dim).init(device),
            position_embeddings: EmbeddingConfig::new(max_position_embeddings, hidden_dim).init(device),
            token_type_embeddings: EmbeddingConfig::new(type_vocab_size, hidden_dim).init(device),
            layer_norm: LayerNormConfig::new(hidden_dim).init(device),
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Word embeddings
        let word_emb = self.word_embeddings.forward(input_ids);

        // Position embeddings
        let positions: Tensor<B, 1, Int> = Tensor::arange(0..seq_len as i64, &device);
        let positions = positions.reshape([1, seq_len]).repeat_dim(0, batch);
        let pos_emb = self.position_embeddings.forward(positions);

        // Token type embeddings (default to zeros if not provided)
        let token_type_ids = token_type_ids.unwrap_or_else(|| {
            Tensor::zeros([batch, seq_len], &device)
        });
        let type_emb = self.token_type_embeddings.forward(token_type_ids);

        // Combine and normalize
        let embeddings = word_emb + pos_emb + type_emb;
        self.layer_norm.forward(embeddings)
    }
}

// ============================================================================
// BERT encoder (stack of layers)
// ============================================================================

/// BERT encoder (stack of transformer layers)
#[derive(Module, Debug)]
pub struct BertEncoder<B: Backend> {
    layer: Vec<BertLayer<B>>,
}

impl<B: Backend> BertEncoder<B> {
    pub fn new(
        device: &B::Device,
        num_layers: usize,
        hidden_dim: usize,
        num_heads: usize,
        intermediate_dim: usize,
    ) -> Self {
        let layer = (0..num_layers)
            .map(|_| BertLayer::new(device, hidden_dim, num_heads, intermediate_dim))
            .collect();
        Self { layer }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let mut hidden = x;
        for layer in &self.layer {
            hidden = layer.forward(hidden, mask.clone());
        }
        hidden
    }
}

// ============================================================================
// Complete BERT model
// ============================================================================

/// BERT model matching HuggingFace structure
///
/// Weight names match HuggingFace for loading:
/// - embeddings.word_embeddings
/// - embeddings.position_embeddings
/// - embeddings.token_type_embeddings
/// - embeddings.LayerNorm
/// - encoder.layer.{i}.attention.self.{query,key,value}
/// - encoder.layer.{i}.attention.output.{dense,LayerNorm}
/// - encoder.layer.{i}.intermediate.dense
/// - encoder.layer.{i}.output.{dense,LayerNorm}
#[derive(Module, Debug)]
pub struct BertModel<B: Backend> {
    embeddings: BertEmbeddings<B>,
    encoder: BertEncoder<B>,
    hidden_dim: usize,
}

impl<B: Backend> BertModel<B> {
    pub fn new(
        device: &B::Device,
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_dim: usize,
        max_position_embeddings: usize,
    ) -> Self {
        Self {
            embeddings: BertEmbeddings::new(device, vocab_size, hidden_dim, max_position_embeddings, 2),
            encoder: BertEncoder::new(device, num_layers, hidden_dim, num_heads, intermediate_dim),
            hidden_dim,
        }
    }

    /// Forward pass returning hidden states
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2>>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let hidden = self.embeddings.forward(input_ids, token_type_ids);
        self.encoder.forward(hidden, attention_mask)
    }

    /// Mean pooling over tokens (sentence-transformers style)
    pub fn mean_pooling(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, seq_len, hidden] = hidden_states.dims();
        let mask = attention_mask.reshape([batch, seq_len, 1]);
        let masked = hidden_states * mask.clone();
        let sum: Tensor<B, 2> = masked.sum_dim(1).reshape([batch, hidden]);
        let count: Tensor<B, 2> = mask.sum_dim(1).reshape([batch, 1]).clamp_min(1e-9);
        sum / count
    }
}

// ============================================================================
// Sentence encoder (BERT + projection) - produces cacheable embeddings
// ============================================================================

/// Sentence encoder: BERT + mean pooling + projection
///
/// This produces fixed-size embeddings that can be cached.
/// The projection maps from BERT hidden dim to scorer hidden dim.
#[derive(Module, Debug)]
pub struct SentenceEncoder<B: Backend> {
    /// BERT encoder (weights from HuggingFace)
    bert: BertModel<B>,
    /// Projection to scorer dimension
    projection: Linear<B>,
    /// Output embedding dimension
    embedding_dim: usize,
}

impl<B: Backend> SentenceEncoder<B> {
    pub fn new(
        device: &B::Device,
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_dim: usize,
        max_position_embeddings: usize,
        embedding_dim: usize,
    ) -> Self {
        Self {
            bert: BertModel::new(
                device,
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
                intermediate_dim,
                max_position_embeddings,
            ),
            projection: LinearConfig::new(hidden_dim, embedding_dim).init(device),
            embedding_dim,
        }
    }

    /// Encode tokens to embeddings (BERT + mean pooling + projection)
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let hidden = self.bert.forward(input_ids, Some(attention_mask.clone()), None);
        let pooled = self.bert.mean_pooling(hidden, attention_mask);
        self.projection.forward(pooled)
    }

    /// Get the output embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

// ============================================================================
// MLP scorer - scores pre-computed embeddings
// ============================================================================

/// MLP scorer head for clause scoring
///
/// Takes pre-computed embeddings and produces scores.
/// This is fast since embeddings are already computed.
#[derive(Module, Debug)]
pub struct MlpScorer<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    input_dim: usize,
}

impl<B: Backend> MlpScorer<B> {
    pub fn new(device: &B::Device, input_dim: usize) -> Self {
        Self {
            linear1: LinearConfig::new(input_dim, input_dim).init(device),
            linear2: LinearConfig::new(input_dim, 1).init(device),
            input_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = gelu(self.linear1.forward(x));
        self.linear2.forward(h)
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
}

// ============================================================================
// Complete sentence model (for backward compatibility)
// ============================================================================

/// Complete sentence model for clause scoring
///
/// Architecture:
/// - BERT encoder (matches HuggingFace)
/// - Mean pooling
/// - Projection to scorer dimension
/// - MLP scorer
///
/// For new code, prefer using SentenceEncoder + MlpScorer separately
/// with the CachingSelector for automatic embedding caching.
#[derive(Module, Debug)]
pub struct SentenceModel<B: Backend> {
    /// Sentence encoder (BERT + projection)
    encoder: SentenceEncoder<B>,
    /// Scoring head
    scorer: MlpScorer<B>,
}

impl<B: Backend> SentenceModel<B> {
    pub fn new(
        device: &B::Device,
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_dim: usize,
        max_position_embeddings: usize,
        scorer_hidden_dim: usize,
    ) -> Self {
        Self {
            encoder: SentenceEncoder::new(
                device,
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
                intermediate_dim,
                max_position_embeddings,
                scorer_hidden_dim,
            ),
            scorer: MlpScorer::new(device, scorer_hidden_dim),
        }
    }

    /// Get the encoder component
    pub fn encoder(&self) -> &SentenceEncoder<B> {
        &self.encoder
    }

    /// Get the scorer component
    pub fn scorer(&self) -> &MlpScorer<B> {
        &self.scorer
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.encoder.embedding_dim()
    }

    /// Encode and project to scorer dimension (cacheable)
    pub fn encode_and_project(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.encoder.forward(input_ids, attention_mask)
    }

    /// Score pre-computed projected embeddings (fast, just MLP)
    pub fn score(&self, projected: Tensor<B, 2>) -> Tensor<B, 2> {
        self.scorer.forward(projected)
    }

    /// Forward pass: encode and score
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let embeddings = self.encoder.forward(input_ids, attention_mask);
        self.scorer.forward(embeddings)
    }

    /// Consume the model and return its components
    ///
    /// This is useful for creating a CachingSelector from a loaded model.
    pub fn into_parts(self) -> (SentenceEncoder<B>, MlpScorer<B>) {
        (self.encoder, self.scorer)
    }
}

// ============================================================================
// Sentence embedder - implements ClauseEmbedder trait
// ============================================================================

/// Sentence embedder that converts clauses to embeddings using BERT
///
/// This implements the ClauseEmbedder trait, allowing it to be used with
/// CachingSelector for automatic embedding caching.
#[cfg(feature = "sentence")]
pub struct SentenceEmbedder<B: Backend> {
    encoder: SentenceEncoder<B>,
    tokenizer: tokenizers::Tokenizer,
    device: B::Device,
    max_length: usize,
}

#[cfg(feature = "sentence")]
impl<B: Backend> SentenceEmbedder<B> {
    pub fn new(encoder: SentenceEncoder<B>, tokenizer: tokenizers::Tokenizer, device: B::Device) -> Self {
        Self {
            encoder,
            tokenizer,
            device,
            max_length: 128,
        }
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    fn tokenize_clauses(&self, clause_strings: &[String]) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
        let encodings = self
            .tokenizer
            .encode_batch(clause_strings.to_vec(), true)
            .expect("Tokenization failed");

        let mut all_ids = Vec::new();
        let mut all_masks = Vec::new();

        for encoding in encodings {
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let mask: Vec<f32> = encoding.get_attention_mask().iter().map(|&x| x as f32).collect();
            all_ids.push(ids);
            all_masks.push(mask);
        }

        // Pad to max length in batch
        let max_len = all_ids.iter().map(|v| v.len()).max().unwrap_or(0).min(self.max_length);

        for ids in &mut all_ids {
            ids.truncate(max_len);
            ids.resize(max_len, 0);
        }
        for mask in &mut all_masks {
            mask.truncate(max_len);
            mask.resize(max_len, 0.0);
        }

        (all_ids, all_masks)
    }
}

#[cfg(feature = "sentence")]
impl<B: Backend> super::cached::ClauseEmbedder for SentenceEmbedder<B> {
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
        if clauses.is_empty() {
            return vec![];
        }

        // Convert clauses to strings
        let clause_strings: Vec<String> = clauses.iter().map(|c| c.to_string()).collect();

        // Tokenize
        let (input_ids, attention_masks) = self.tokenize_clauses(&clause_strings);
        let batch_size = input_ids.len();
        let seq_len = input_ids[0].len();

        // Flatten for tensor creation
        let flat_ids: Vec<i64> = input_ids.into_iter().flatten().collect();
        let flat_mask: Vec<f32> = attention_masks.into_iter().flatten().collect();

        // Create tensors
        let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(flat_ids, [batch_size, seq_len]),
            &self.device,
        );
        let mask_tensor: Tensor<B, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(flat_mask, [batch_size, seq_len]),
            &self.device,
        );

        // Encode to embeddings
        let embeddings = self.encoder.forward(input_tensor, mask_tensor);
        let embedding_data: Vec<f32> = embeddings.into_data().to_vec().unwrap();
        let embedding_dim = self.encoder.embedding_dim();

        // Split into individual embeddings
        embedding_data
            .chunks(embedding_dim)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    fn embedding_dim(&self) -> usize {
        self.encoder.embedding_dim()
    }

    fn name(&self) -> &str {
        "sentence"
    }
}

// ============================================================================
// Burn MLP scorer - implements EmbeddingScorer trait
// ============================================================================

/// MLP scorer wrapper that implements EmbeddingScorer trait
#[cfg(feature = "sentence")]
pub struct BurnMlpScorerWrapper<B: Backend> {
    scorer: MlpScorer<B>,
    device: B::Device,
}

#[cfg(feature = "sentence")]
impl<B: Backend> BurnMlpScorerWrapper<B> {
    pub fn new(scorer: MlpScorer<B>, device: B::Device) -> Self {
        Self { scorer, device }
    }
}

#[cfg(feature = "sentence")]
impl<B: Backend> super::cached::EmbeddingScorer for BurnMlpScorerWrapper<B> {
    fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![];
        }

        let batch_size = embeddings.len();
        let embedding_dim = embeddings[0].len();

        // Flatten embeddings
        let flat: Vec<f32> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();

        // Create tensor
        let tensor: Tensor<B, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(flat, [batch_size, embedding_dim]),
            &self.device,
        );

        // Score
        let scores = self.scorer.forward(tensor);
        scores.into_data().to_vec().unwrap()
    }

    fn name(&self) -> &str {
        "mlp"
    }
}

// ============================================================================
// Type aliases for caching selector
// ============================================================================

/// Sentence selector with caching - the recommended way to use sentence embeddings
#[cfg(feature = "sentence")]
pub type BurnSentenceSelector<B> = super::cached::CachingSelector<SentenceEmbedder<B>, BurnMlpScorerWrapper<B>>;

/// Sentence selector with ndarray backend (CPU)
#[cfg(feature = "sentence")]
pub type NdarraySentenceSelector = BurnSentenceSelector<burn_ndarray::NdArray<f32>>;


// ============================================================================
// Weight loading
// ============================================================================

/// Load sentence selector from safetensors and tokenizer files
///
/// Creates a CachingSelector with SentenceEmbedder and MLP scorer.
/// Embeddings are automatically cached for efficient clause selection.
#[cfg(feature = "sentence")]
pub fn load_ndarray_sentence_selector<P: AsRef<Path>>(
    weights_path: P,
    tokenizer_path: P,
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_dim: usize,
    max_position_embeddings: usize,
    scorer_hidden_dim: usize,
) -> Result<NdarraySentenceSelector, String> {
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};

    let device = burn_ndarray::NdArrayDevice::Cpu;

    // Create model structure
    let model = SentenceModel::new(
        &device,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        max_position_embeddings,
        scorer_hidden_dim,
    );

    // Load weights with PyTorch adapter
    let load_args = LoadArgs::new(weights_path.as_ref().into()).with_adapter_type(AdapterType::PyTorch);

    let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .map_err(|e| format!("Failed to load safetensors: {}", e))?;

    let model = model.load_record(record);

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Split model into encoder and scorer for caching architecture
    let (encoder, scorer) = model.into_parts();

    // Create embedder and scorer wrappers
    let embedder = SentenceEmbedder::new(encoder, tokenizer, device.clone());
    let scorer_wrapper = BurnMlpScorerWrapper::new(scorer, device);

    // Create caching selector
    Ok(super::cached::CachingSelector::new(embedder, scorer_wrapper))
}

// ============================================================================
// ONNX Runtime GPU embedder
// ============================================================================

/// ONNX-based sentence embedder for GPU-accelerated inference
///
/// Uses ONNX Runtime with CUDA execution provider for fast GPU inference.
#[cfg(all(feature = "sentence", feature = "onnx"))]
pub struct OnnxSentenceEmbedder {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    hidden_dim: usize,
    max_length: usize,
}

#[cfg(all(feature = "sentence", feature = "onnx"))]
impl OnnxSentenceEmbedder {
    pub fn new<P: AsRef<Path>>(
        onnx_path: P,
        tokenizer_path: P,
        hidden_dim: usize,
    ) -> Result<Self, String> {
        use ort::execution_providers::{CUDAExecutionProvider, CPUExecutionProvider};
        use ort::session::Session;

        // Initialize ONNX Runtime with CUDA provider (falls back to CPU if CUDA unavailable)
        let session = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e))?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| format!("Failed to set execution providers: {}", e))?
            .commit_from_file(onnx_path.as_ref())
            .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            hidden_dim,
            max_length: 128,
        })
    }

    fn tokenize_clauses(&self, clause_strings: &[String]) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let encodings = self
            .tokenizer
            .encode_batch(clause_strings.to_vec(), true)
            .expect("Tokenization failed");

        let mut all_ids = Vec::new();
        let mut all_masks = Vec::new();

        for encoding in encodings {
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
            all_ids.push(ids);
            all_masks.push(mask);
        }

        // Pad to max length in batch
        let max_len = all_ids.iter().map(|v| v.len()).max().unwrap_or(0).min(self.max_length);

        for ids in &mut all_ids {
            ids.truncate(max_len);
            ids.resize(max_len, 0);
        }
        for mask in &mut all_masks {
            mask.truncate(max_len);
            mask.resize(max_len, 0);
        }

        (all_ids, all_masks)
    }

    fn mean_pooling(&self, hidden_states: &[f32], attention_mask: &[i64], batch_size: usize, seq_len: usize) -> Vec<Vec<f32>> {
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut sum = vec![0.0f32; self.hidden_dim];
            let mut count = 0.0f32;

            for s in 0..seq_len {
                let mask_val = attention_mask[b * seq_len + s] as f32;
                if mask_val > 0.0 {
                    for h in 0..self.hidden_dim {
                        sum[h] += hidden_states[b * seq_len * self.hidden_dim + s * self.hidden_dim + h];
                    }
                    count += 1.0;
                }
            }

            if count > 0.0 {
                for h in 0..self.hidden_dim {
                    sum[h] /= count;
                }
            }

            results.push(sum);
        }

        results
    }
}

#[cfg(all(feature = "sentence", feature = "onnx"))]
impl super::cached::ClauseEmbedder for OnnxSentenceEmbedder {
    fn embed_batch(&self, clauses: &[&Clause]) -> Vec<Vec<f32>> {
        use ort::value::Tensor;

        if clauses.is_empty() {
            return vec![];
        }

        // Convert clauses to strings
        let clause_strings: Vec<String> = clauses.iter().map(|c| c.to_string()).collect();

        // Tokenize
        let (input_ids, attention_masks) = self.tokenize_clauses(&clause_strings);
        let batch_size = input_ids.len();
        let seq_len = input_ids[0].len();

        // Flatten for tensor creation
        let flat_ids: Vec<i64> = input_ids.into_iter().flatten().collect();
        let flat_mask: Vec<i64> = attention_masks.iter().flatten().copied().collect();

        // Create ONNX tensors using ort's Tensor type
        let ids_tensor = Tensor::from_array(([batch_size, seq_len], flat_ids.clone()))
            .expect("Failed to create input_ids tensor");
        let mask_tensor = Tensor::from_array(([batch_size, seq_len], flat_mask.clone()))
            .expect("Failed to create attention_mask tensor");

        // Run inference with named inputs
        let mut session = self.session.lock().expect("Session lock poisoned");
        let outputs = session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        ]).expect("ONNX inference failed");

        // Extract embeddings [batch, hidden_dim] - already mean-pooled and projected
        let embeddings_view: ndarray::ArrayViewD<f32> = outputs[0]
            .try_extract_array()
            .expect("Failed to extract tensor");

        // Split into individual embeddings
        let embedding_dim = self.hidden_dim;
        embeddings_view
            .as_slice()
            .unwrap()
            .chunks(embedding_dim)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    fn embedding_dim(&self) -> usize {
        self.hidden_dim
    }

    fn name(&self) -> &str {
        "onnx_sentence"
    }
}

/// ONNX sentence selector with GPU acceleration
#[cfg(all(feature = "sentence", feature = "onnx"))]
pub type OnnxSentenceSelector = super::cached::CachingSelector<OnnxSentenceEmbedder, BurnMlpScorerWrapper<burn_ndarray::NdArray<f32>>>;

/// Load ONNX GPU-accelerated sentence selector
///
/// Uses ONNX Runtime with CUDA for the encoder and Burn ndarray for the MLP scorer.
#[cfg(all(feature = "sentence", feature = "onnx"))]
pub fn load_onnx_sentence_selector<P: AsRef<Path>>(
    onnx_path: P,
    scorer_weights_path: P,
    tokenizer_path: P,
    hidden_dim: usize,
    scorer_hidden_dim: usize,
) -> Result<OnnxSentenceSelector, String> {
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};

    // Create ONNX embedder
    let embedder = OnnxSentenceEmbedder::new(&onnx_path, &tokenizer_path, hidden_dim)?;

    // Load MLP scorer weights with Burn
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let scorer = MlpScorer::new(&device, scorer_hidden_dim);

    let load_args = LoadArgs::new(scorer_weights_path.as_ref().into()).with_adapter_type(AdapterType::PyTorch);
    let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .map_err(|e| format!("Failed to load scorer weights: {}", e))?;
    let scorer = scorer.load_record(record);

    let scorer_wrapper = BurnMlpScorerWrapper::new(scorer, device);

    Ok(super::cached::CachingSelector::new(embedder, scorer_wrapper))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_model_forward() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let model: BertModel<burn_ndarray::NdArray<f32>> = BertModel::new(
            &device,
            1000,  // vocab_size
            64,    // hidden_dim
            2,     // num_layers
            4,     // num_heads
            256,   // intermediate_dim
            128,   // max_position_embeddings
        );

        let input_ids: Tensor<burn_ndarray::NdArray<f32>, 2, Int> =
            Tensor::from_ints([[1, 2, 3, 0], [4, 5, 0, 0]], &device);
        let attention_mask = Tensor::from_floats([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], &device);

        let output = model.forward(input_ids, Some(attention_mask), None);
        assert_eq!(output.dims(), [2, 4, 64]);
    }

    #[test]
    fn test_sentence_model_forward() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let model: SentenceModel<burn_ndarray::NdArray<f32>> = SentenceModel::new(
            &device,
            1000,  // vocab_size
            64,    // hidden_dim
            2,     // num_layers
            4,     // num_heads
            256,   // intermediate_dim
            128,   // max_position_embeddings
            32,    // scorer_hidden_dim
        );

        let input_ids: Tensor<burn_ndarray::NdArray<f32>, 2, Int> =
            Tensor::from_ints([[1, 2, 3, 0], [4, 5, 0, 0]], &device);
        let attention_mask = Tensor::from_floats([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], &device);

        let output = model.forward(input_ids, attention_mask);
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_bert_attention() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let attn: BertAttention<burn_ndarray::NdArray<f32>> = BertAttention::new(&device, 64, 4);

        let x = Tensor::zeros([2, 8, 64], &device);
        let mask = Tensor::ones([2, 8], &device);

        let output = attn.forward(x, Some(mask));
        assert_eq!(output.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_mean_pooling() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let model: BertModel<burn_ndarray::NdArray<f32>> = BertModel::new(
            &device,
            1000, 64, 1, 4, 256, 128,
        );

        let hidden = Tensor::ones([2, 4, 64], &device);
        let mask = Tensor::from_floats([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], &device);

        let pooled = model.mean_pooling(hidden, mask);
        assert_eq!(pooled.dims(), [2, 64]);
    }
}
