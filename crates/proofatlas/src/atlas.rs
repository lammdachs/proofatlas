//! ProofAtlas orchestrator: reusable across problems.
//!
//! The `ProofAtlas` struct holds configuration, ML backend, and include dirs.
//! It creates the appropriate sink and [`Prover`] per problem.
//!
//! ```ignore
//! let atlas = ProofAtlas::builder(config)
//!     .include_dir(".tptp/TPTP-v9.0.0")
//!     .build()?;
//!
//! let (result, prover) = atlas.prove_file("problem.p")?;
//! ```

use std::sync::Arc;

use crate::config::ProverConfig;
use crate::logic::Interner;
use crate::parser::{parse_tptp, parse_tptp_file};
use crate::prover::Prover;
use crate::selection::clause::ProverSink;
use crate::selection::AgeWeightSink;
use crate::state::ProofResult;

#[cfg(feature = "ml")]
use crate::selection::pipeline::backend::Backend;
#[cfg(feature = "ml")]
use crate::selection::pipeline::backend::BackendHandle;

/// What kind of sink to create per problem.
enum SinkKind {
    /// Heuristic age-weight ratio selector (no ML).
    AgeWeight { ratio: f64 },
    /// In-process pipelined ML inference via a shared Backend.
    #[cfg(feature = "ml")]
    Pipeline {
        handle: BackendHandle,
        /// Encoder type: "gcn", "gat", "graphsage", "sentence", "features".
        encoder: String,
        /// Scorer type: "mlp", "attention", "transformer".
        scorer: String,
    },
    /// Remote scoring server via Unix socket.
    #[cfg(unix)]
    Remote { socket_path: String },
}

/// Reusable orchestrator for proving multiple problems.
///
/// Holds configuration, ML backend (if any), and include directories.
/// Call [`prove_file`](ProofAtlas::prove_file) or [`prove_string`](ProofAtlas::prove_string)
/// to prove individual problems.
pub struct ProofAtlas {
    config: ProverConfig,
    include_dirs: Vec<String>,
    sink_kind: SinkKind,
    temperature: f32,
    /// Keep the Backend alive for the lifetime of the orchestrator.
    #[cfg(feature = "ml")]
    _backend: Option<Backend>,
}

impl ProofAtlas {
    /// Create a builder for constructing a ProofAtlas orchestrator.
    pub fn builder(config: ProverConfig) -> ProofAtlasBuilder {
        ProofAtlasBuilder {
            config,
            include_dirs: Vec::new(),
            encoder: None,
            scorer: None,
            weights_path: None,
            use_cuda: false,
            temperature: 1.0,
            age_weight_ratio: 0.5,
            socket_path: None,
        }
    }

    /// Prove a TPTP file. Returns `(ProofResult, Prover)` with the prover
    /// retaining all state (clauses, event log, interner) for inspection.
    pub fn prove_file(&self, path: &str) -> Result<(ProofResult, Prover), String> {
        let include_refs: Vec<&str> = self.include_dirs.iter().map(|s| s.as_str()).collect();
        let timeout_instant = Some(std::time::Instant::now() + self.config.timeout);
        let parsed = parse_tptp_file(path, &include_refs, timeout_instant, self.config.memory_limit)?;
        self.prove_impl(parsed.formula.clauses, parsed.interner)
    }

    /// Prove TPTP content from a string. Returns `(ProofResult, Prover)`.
    pub fn prove_string(&self, content: &str) -> Result<(ProofResult, Prover), String> {
        let include_refs: Vec<&str> = self.include_dirs.iter().map(|s| s.as_str()).collect();
        let timeout_instant = Some(std::time::Instant::now() + self.config.timeout);
        let parsed = parse_tptp(content, &include_refs, timeout_instant, self.config.memory_limit)?;
        self.prove_impl(parsed.formula.clauses, parsed.interner)
    }

    /// Get the prover config.
    pub fn config(&self) -> &ProverConfig {
        &self.config
    }

    /// Get the include directories.
    pub fn include_dirs(&self) -> &[String] {
        &self.include_dirs
    }

    /// Parse a TPTP file without running saturation.
    pub fn parse_file(&self, path: &str) -> Result<crate::parser::ParsedProblem, String> {
        let include_refs: Vec<&str> = self.include_dirs.iter().map(|s| s.as_str()).collect();
        let timeout_instant = Some(std::time::Instant::now() + self.config.timeout);
        parse_tptp_file(path, &include_refs, timeout_instant, self.config.memory_limit)
    }

    /// Parse TPTP content from a string without running saturation.
    pub fn parse_string(&self, content: &str) -> Result<crate::parser::ParsedProblem, String> {
        let include_refs: Vec<&str> = self.include_dirs.iter().map(|s| s.as_str()).collect();
        let timeout_instant = Some(std::time::Instant::now() + self.config.timeout);
        parse_tptp(content, &include_refs, timeout_instant, self.config.memory_limit)
    }

    /// Core proving logic: create sink, build Prover, run saturation.
    fn prove_impl(
        &self,
        clauses: Vec<crate::logic::Clause>,
        interner: Interner,
    ) -> Result<(ProofResult, Prover), String> {
        let sink = self.create_sink(&interner)?;
        let mut prover = Prover::new(clauses, self.config.clone(), sink, interner);
        let result = prover.prove();
        Ok((result, prover))
    }

    /// Create the appropriate sink based on `sink_kind`.
    pub fn create_sink(&self, interner: &Interner) -> Result<Box<dyn ProverSink>, String> {
        match &self.sink_kind {
            SinkKind::AgeWeight { ratio } => {
                Ok(Box::new(AgeWeightSink::new(*ratio)))
            }
            #[cfg(feature = "ml")]
            SinkKind::Pipeline { handle, encoder, scorer } => {
                use crate::selection::pipeline::processors;

                let int = Arc::new(interner.clone());
                let temp = self.temperature;
                let h = handle.clone();

                let processor: Box<dyn crate::selection::DataProcessor> = match (encoder.as_str(), scorer.as_str()) {
                    // GCN + MLP: cache scores
                    ("gcn" | "gat" | "graphsage", "mlp") => {
                        Box::new(processors::GcnScoreProcessor::new(h, temp))
                    }
                    // GCN + attention/transformer: cache embeddings
                    ("gcn" | "gat" | "graphsage", "attention" | "transformer") => {
                        Box::new(processors::GcnEmbeddingProcessor::new(h, temp))
                    }
                    // Sentence + MLP: cache scores
                    ("sentence", "mlp") => {
                        Box::new(processors::SentenceScoreProcessor::new(h, int, temp))
                    }
                    // Sentence + attention/transformer: cache embeddings
                    ("sentence", "attention" | "transformer") => {
                        Box::new(processors::SentenceEmbeddingProcessor::new(h, int, temp))
                    }
                    // Features + MLP: cache scores
                    ("features", "mlp") => {
                        Box::new(processors::FeaturesScoreProcessor::new(h, temp))
                    }
                    // Features + attention/transformer: cache embeddings
                    ("features", "attention" | "transformer") => {
                        Box::new(processors::FeaturesEmbeddingProcessor::new(h, temp))
                    }
                    _ => {
                        return Err(format!(
                            "Unknown encoder/scorer combination: '{}' + '{}'",
                            encoder, scorer
                        ));
                    }
                };

                Ok(Box::new(crate::selection::create_pipeline(
                    processor,
                    format!("{}_{}", encoder, scorer),
                )))
            }
            #[cfg(unix)]
            SinkKind::Remote { socket_path } => {
                use crate::selection::ClauseSelector; // for set_interner
                let mut selector = crate::selection::RemoteSelector::connect(socket_path)?;
                selector.set_interner(Arc::new(interner.clone()));
                Ok(Box::new(crate::selection::RemoteSelectorSink::new(selector)))
            }
        }
    }
}

/// Builder for constructing a [`ProofAtlas`] orchestrator.
pub struct ProofAtlasBuilder {
    config: ProverConfig,
    include_dirs: Vec<String>,
    encoder: Option<String>,
    scorer: Option<String>,
    weights_path: Option<String>,
    use_cuda: bool,
    temperature: f32,
    age_weight_ratio: f64,
    socket_path: Option<String>,
}

impl ProofAtlasBuilder {
    /// Add an include directory for resolving TPTP `include()` directives.
    pub fn include_dir(mut self, dir: impl Into<String>) -> Self {
        self.include_dirs.push(dir.into());
        self
    }

    /// Set the ML encoder type (e.g., "gcn", "gat", "graphsage", "sentence", "features").
    pub fn encoder(mut self, encoder: impl Into<String>) -> Self {
        self.encoder = Some(encoder.into());
        self
    }

    /// Set the scorer type (e.g., "mlp", "attention", "transformer").
    pub fn scorer(mut self, scorer: impl Into<String>) -> Self {
        self.scorer = Some(scorer.into());
        self
    }

    /// Set the path to model weights directory.
    pub fn weights_path(mut self, path: impl Into<String>) -> Self {
        self.weights_path = Some(path.into());
        self
    }

    /// Enable CUDA for ML inference.
    pub fn use_cuda(mut self, cuda: bool) -> Self {
        self.use_cuda = cuda;
        self
    }

    /// Set softmax temperature for ML clause selection.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set age-weight ratio for heuristic selection (only used when no encoder).
    pub fn age_weight_ratio(mut self, ratio: f64) -> Self {
        self.age_weight_ratio = ratio;
        self
    }

    /// Set Unix socket path for remote scoring server.
    pub fn socket_path(mut self, path: impl Into<String>) -> Self {
        self.socket_path = Some(path.into());
        self
    }

    /// Build the `ProofAtlas` orchestrator.
    pub fn build(self) -> Result<ProofAtlas, String> {
        #[cfg(unix)]
        if let Some(ref socket_path) = self.socket_path {
            return Ok(ProofAtlas {
                config: self.config,
                include_dirs: self.include_dirs,
                sink_kind: SinkKind::Remote { socket_path: socket_path.clone() },
                temperature: self.temperature,
                #[cfg(feature = "ml")]
                _backend: None,
            });
        }

        match self.encoder.as_deref() {
            None => {
                // No ML — use heuristic selector
                Ok(ProofAtlas {
                    config: self.config,
                    include_dirs: self.include_dirs,
                    sink_kind: SinkKind::AgeWeight { ratio: self.age_weight_ratio },
                    temperature: self.temperature,
                    #[cfg(feature = "ml")]
                    _backend: None,
                })
            }
            #[cfg(feature = "ml")]
            Some(enc @ ("gcn" | "gat" | "graphsage" | "sentence" | "features")) => {
                let scorer_name = self.scorer.as_deref()
                    .ok_or_else(|| "scorer required when encoder is set".to_string())?;
                let model_name = format!("{}_{}", enc, scorer_name);
                let is_mlp = scorer_name == "mlp";

                let weights_dir = std::path::PathBuf::from(
                    self.weights_path.as_deref().unwrap_or(".weights")
                );

                let models: Vec<Box<dyn crate::selection::Model>> = if is_mlp {
                    // MLP path: load combined model → EmbedScoreModel
                    let (embedder, scorer_box) = load_embedder_scorer(
                        enc, &model_name, &weights_dir, self.use_cuda,
                    )?;
                    vec![Box::new(crate::selection::EmbedScoreModel::new(embedder, scorer_box))]
                } else {
                    // Attention/Transformer path: load encoder + scorer separately
                    let (embedder, scorer_box) = load_split_encoder_scorer(
                        enc, &model_name, &weights_dir, scorer_name, self.use_cuda,
                    )?;
                    vec![
                        Box::new(crate::selection::EmbedModel::new(embedder)),
                        Box::new(crate::selection::ContextScoreModel::new(scorer_box)),
                    ]
                };

                let backend = crate::selection::Backend::new(models);
                let handle = backend.handle();

                Ok(ProofAtlas {
                    config: self.config,
                    include_dirs: self.include_dirs,
                    sink_kind: SinkKind::Pipeline {
                        handle,
                        encoder: enc.to_string(),
                        scorer: scorer_name.to_string(),
                    },
                    temperature: self.temperature,
                    _backend: Some(backend),
                })
            }
            Some(other) => {
                #[cfg(feature = "ml")]
                let available = "None, 'gcn', 'gat', 'graphsage', 'sentence', or 'features'";
                #[cfg(not(feature = "ml"))]
                let available = "None (ML features not enabled)";
                Err(format!("Unknown encoder: '{}'. Use {}", other, available))
            }
        }
    }
}

/// Load embedder + scorer for MLP configurations (combined model).
#[cfg(feature = "ml")]
fn load_embedder_scorer(
    enc: &str,
    model_name: &str,
    weights_dir: &std::path::Path,
    use_cuda: bool,
) -> Result<(
    Box<dyn crate::selection::cached::ClauseEmbedder + Send>,
    Box<dyn crate::selection::cached::EmbeddingScorer + Send>,
), String> {
    let model_path = weights_dir.join(format!("{}.pt", model_name));
    if !model_path.exists() {
        return Err(format!("Model not found at {}", model_path.display()));
    }

    match enc {
        "sentence" => {
            let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", model_name));
            let emb = crate::selection::load_sentence_embedder(&model_path, &tokenizer_path, use_cuda)?;
            Ok((Box::new(emb), Box::new(crate::selection::PassThroughScorer) as _))
        }
        "gcn" | "gat" | "graphsage" => {
            let emb = crate::selection::load_gcn_embedder(&model_path, use_cuda)?;
            Ok((Box::new(emb) as _, Box::new(crate::selection::GcnScorer) as _))
        }
        "features" => {
            // Features models: TorchScript model takes [N, 9] features → scores
            // We use GcnEmbedder-like approach: load as combined model
            let emb = crate::selection::load_gcn_embedder(&model_path, use_cuda)?;
            Ok((Box::new(emb) as _, Box::new(crate::selection::GcnScorer) as _))
        }
        _ => Err(format!("Unknown encoder for MLP path: '{}'", enc)),
    }
}

/// Load encoder + scorer separately for attention/transformer configurations.
#[cfg(feature = "ml")]
fn load_split_encoder_scorer(
    enc: &str,
    model_name: &str,
    weights_dir: &std::path::Path,
    scorer_name: &str,
    use_cuda: bool,
) -> Result<(
    Box<dyn crate::selection::cached::ClauseEmbedder + Send>,
    Box<dyn crate::selection::cached::EmbeddingScorer + Send>,
), String> {
    let encoder_path = weights_dir.join(format!("{}_encoder.pt", model_name));
    let scorer_path = weights_dir.join(format!("{}_scorer.pt", model_name));

    // Fall back to combined model if split models don't exist yet
    if !encoder_path.exists() || !scorer_path.exists() {
        let combined_path = weights_dir.join(format!("{}.pt", model_name));
        if combined_path.exists() {
            eprintln!(
                "Warning: Split encoder/scorer models not found for '{}'. \
                 Falling back to combined model (re-export with updated export.py for optimal performance).",
                model_name
            );
            return load_embedder_scorer(enc, model_name, weights_dir, use_cuda);
        }
        return Err(format!(
            "Neither split models ({}_encoder.pt, {}_scorer.pt) nor combined model ({}.pt) found in {}",
            model_name, model_name, model_name, weights_dir.display()
        ));
    }

    let cross_attention = scorer_name == "attention" || scorer_name == "transformer";
    // Default hidden_dim — ideally read from model metadata, but for now use 256 (GCN) or 64 (sentence/features)
    let hidden_dim = match enc {
        "gcn" | "gat" | "graphsage" => 256,
        _ => 64,
    };

    match enc {
        "gcn" | "gat" | "graphsage" => {
            let encoder = crate::selection::GcnEncoder::new(&encoder_path, hidden_dim, use_cuda)?;
            let scorer = crate::selection::TorchScriptScorer::new(&scorer_path, hidden_dim, cross_attention, use_cuda)?;
            Ok((Box::new(encoder), Box::new(scorer)))
        }
        "sentence" => {
            let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", model_name));
            let emb = crate::selection::load_sentence_embedder(&encoder_path, &tokenizer_path, use_cuda)?;
            let scorer = crate::selection::TorchScriptScorer::new(&scorer_path, hidden_dim, cross_attention, use_cuda)?;
            Ok((Box::new(emb), Box::new(scorer)))
        }
        "features" => {
            // Features encoder: load TorchScript model that takes [N, 9] → [N, hidden_dim]
            let encoder = crate::selection::GcnEncoder::new(&encoder_path, hidden_dim, use_cuda)?;
            let scorer = crate::selection::TorchScriptScorer::new(&scorer_path, hidden_dim, cross_attention, use_cuda)?;
            Ok((Box::new(encoder), Box::new(scorer)))
        }
        _ => Err(format!("Unknown encoder for split path: '{}'", enc)),
    }
}
