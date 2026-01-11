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
use super::ClauseSelector;

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
// Sentence model (BERT + projection + scorer)
// ============================================================================

/// MLP scorer head for clause scoring
#[derive(Module, Debug)]
pub struct ScorerHead<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> ScorerHead<B> {
    pub fn new(device: &B::Device, hidden_dim: usize) -> Self {
        Self {
            linear1: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            linear2: LinearConfig::new(hidden_dim, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = gelu(self.linear1.forward(x));
        self.linear2.forward(h)
    }
}

/// Complete sentence model for clause scoring
///
/// Architecture:
/// - BERT encoder (matches HuggingFace)
/// - Mean pooling
/// - Projection to scorer dimension
/// - MLP scorer
#[derive(Module, Debug)]
pub struct SentenceModel<B: Backend> {
    /// BERT encoder (weights from HuggingFace)
    bert: BertModel<B>,
    /// Projection to scorer dimension
    projection: Linear<B>,
    /// Scoring head
    scorer: ScorerHead<B>,
    /// Scorer hidden dimension
    scorer_hidden_dim: usize,
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
            bert: BertModel::new(
                device,
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
                intermediate_dim,
                max_position_embeddings,
            ),
            projection: LinearConfig::new(hidden_dim, scorer_hidden_dim).init(device),
            scorer: ScorerHead::new(device, scorer_hidden_dim),
            scorer_hidden_dim,
        }
    }

    /// Encode clauses to embeddings (before projection)
    pub fn encode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let hidden = self.bert.forward(input_ids, Some(attention_mask.clone()), None);
        self.bert.mean_pooling(hidden, attention_mask)
    }

    /// Forward pass: encode and score
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let embeddings = self.encode(input_ids, attention_mask);
        let projected = self.projection.forward(embeddings);
        self.scorer.forward(projected)
    }
}

// ============================================================================
// Type aliases for ndarray backend
// ============================================================================

/// Sentence selector with ndarray backend
#[cfg(feature = "sentence")]
pub type NdarraySentenceSelector = BurnSentenceSelector<burn_ndarray::NdArray<f32>>;

// ============================================================================
// Sentence selector with tokenizer
// ============================================================================

/// Sentence encoder selector with tokenizer
#[cfg(feature = "sentence")]
pub struct BurnSentenceSelector<B: Backend> {
    model: SentenceModel<B>,
    tokenizer: tokenizers::Tokenizer,
    device: B::Device,
    rng_state: u64,
    max_length: usize,
}

#[cfg(feature = "sentence")]
impl<B: Backend> BurnSentenceSelector<B> {
    pub fn new(model: SentenceModel<B>, tokenizer: tokenizers::Tokenizer, device: B::Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
            rng_state: 12345,
            max_length: 128,
        }
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    fn next_random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn clause_to_string(&self, clause: &Clause) -> String {
        clause.to_string()
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
impl<B: Backend> ClauseSelector for BurnSentenceSelector<B> {
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        if unprocessed.len() == 1 {
            return unprocessed.pop_front();
        }

        // Convert clauses to strings
        let clause_strings: Vec<String> = unprocessed
            .iter()
            .map(|&idx| self.clause_to_string(&clauses[idx]))
            .collect();

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

        // Get scores
        let scores = self.model.forward(input_tensor, mask_tensor);
        let scores: Vec<f32> = scores.into_data().to_vec().unwrap();

        // Softmax sampling
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| ((s - max_score) as f64).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = exp_scores.iter().map(|&e| e / sum).collect();

        // Sample
        let r = self.next_random();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return unprocessed.remove(i);
            }
        }

        unprocessed.pop_back()
    }

    fn name(&self) -> &str {
        "burn_sentence"
    }
}

// ============================================================================
// Weight loading
// ============================================================================

/// Load sentence selector from safetensors and tokenizer files
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

    Ok(BurnSentenceSelector::new(model, tokenizer, device))
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
