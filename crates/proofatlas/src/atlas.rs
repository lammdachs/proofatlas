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
        /// Whether CUDA is available (used for per-model device decisions in processors).
        use_cuda: bool,
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
            SinkKind::Pipeline { handle, encoder, scorer, use_cuda } => {
                use crate::selection::pipeline::processors;

                let int = Arc::new(interner.clone());
                let temp = self.temperature;
                let h = handle.clone();
                let cuda = *use_cuda;

                // Device placement per encoder type:
                // - GCN/features: lightweight → encoder on CPU, scorer follows use_cuda
                // - Sentence: heavy (MiniLM) → both follow use_cuda
                let is_lightweight_encoder = matches!(
                    encoder.as_str(),
                    "gcn" | "gat" | "graphsage" | "features"
                );
                let embed_cuda = if is_lightweight_encoder { false } else { cuda };
                let score_cuda = cuda;

                let processor: Box<dyn crate::selection::DataProcessor> = match (encoder.as_str(), scorer.as_str()) {
                    // GCN + MLP: cache scores
                    ("gcn" | "gat" | "graphsage", "mlp") => {
                        Box::new(processors::GcnScoreProcessor::new(h, temp, embed_cuda))
                    }
                    // GCN + attention/transformer: cache embeddings
                    ("gcn" | "gat" | "graphsage", "attention" | "transformer") => {
                        Box::new(processors::GcnEmbeddingProcessor::new(h, temp, embed_cuda, score_cuda))
                    }
                    // Sentence + MLP: cache scores
                    ("sentence", "mlp") => {
                        Box::new(processors::SentenceScoreProcessor::new(h, int, temp, embed_cuda))
                    }
                    // Sentence + attention/transformer: cache embeddings
                    ("sentence", "attention" | "transformer") => {
                        Box::new(processors::SentenceEmbeddingProcessor::new(h, int, temp, embed_cuda, score_cuda))
                    }
                    // Features + MLP: cache scores
                    ("features", "mlp") => {
                        Box::new(processors::FeaturesScoreProcessor::new(h, temp, embed_cuda))
                    }
                    // Features + attention/transformer: cache embeddings
                    ("features", "attention" | "transformer") => {
                        Box::new(processors::FeaturesEmbeddingProcessor::new(h, temp, embed_cuda, score_cuda))
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
                let is_mlp = scorer_name == "mlp";

                let weights_dir = std::path::PathBuf::from(
                    self.weights_path.as_deref().unwrap_or(".weights")
                );

                // Create model specs with lazy-loading factories.
                // The Backend loads each model on the device specified by the
                // first request (from the DataProcessor's embed_cuda/score_cuda).
                let specs = if is_mlp {
                    make_combined_spec(enc, scorer_name, &weights_dir)?
                } else {
                    make_split_specs(enc, scorer_name, &weights_dir)?
                };

                let backend = Backend::new(specs);
                let handle = backend.handle();

                Ok(ProofAtlas {
                    config: self.config,
                    include_dirs: self.include_dirs,
                    sink_kind: SinkKind::Pipeline {
                        handle,
                        encoder: enc.to_string(),
                        scorer: scorer_name.to_string(),
                        use_cuda: self.use_cuda,
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

// =============================================================================
// Model spec factories — create lazy-loading specs for the Backend
// =============================================================================

/// Create a ModelSpec for MLP configurations (combined embed+score model).
#[cfg(feature = "ml")]
fn make_combined_spec(
    enc: &str,
    scorer_name: &str,
    weights_dir: &std::path::Path,
) -> Result<Vec<crate::selection::pipeline::backend::ModelSpec>, String> {
    use crate::selection::pipeline::backend::ModelSpec;

    let model_name = format!("{}_{}", enc, scorer_name);
    let model_path = weights_dir.join(format!("{}.pt", model_name));
    if !model_path.exists() {
        return Err(format!("Model not found at {}", model_path.display()));
    }

    let enc = enc.to_string();
    let model_path_clone = model_path.clone();
    let weights_dir_owned = weights_dir.to_path_buf();

    Ok(vec![ModelSpec {
        model_id: "embed_score".to_string(),
        factory: Box::new(move |use_cuda| {
            let (embedder, scorer) = load_embedder_scorer(
                &enc, &model_path_clone, &weights_dir_owned, use_cuda,
            )?;
            Ok(Box::new(crate::selection::EmbedScoreModel::new(embedder, scorer)))
        }),
    }])
}

/// Create ModelSpecs for attention/transformer configurations (separate encoder + scorer).
#[cfg(feature = "ml")]
fn make_split_specs(
    enc: &str,
    scorer_name: &str,
    weights_dir: &std::path::Path,
) -> Result<Vec<crate::selection::pipeline::backend::ModelSpec>, String> {
    use crate::selection::pipeline::backend::ModelSpec;

    let model_name = format!("{}_{}", enc, scorer_name);
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
            return make_combined_spec(enc, scorer_name, weights_dir);
        }
        return Err(format!(
            "Neither split models ({}_encoder.pt, {}_scorer.pt) nor combined model ({}.pt) found in {}",
            model_name, model_name, model_name, weights_dir.display()
        ));
    }

    let cross_attention = scorer_name == "attention" || scorer_name == "transformer";
    let hidden_dim = match enc {
        "gcn" | "gat" | "graphsage" => 256,
        _ => 64,
    };

    // Encoder spec — loaded lazily with device from first "embed" request
    let enc_owned = enc.to_string();
    let encoder_path_clone = encoder_path.clone();
    let weights_dir_owned = weights_dir.to_path_buf();
    let model_name_clone = model_name.clone();
    let embed_spec = ModelSpec {
        model_id: "embed".to_string(),
        factory: Box::new(move |use_cuda| {
            let embedder = load_encoder(
                &enc_owned, &model_name_clone, &encoder_path_clone, &weights_dir_owned,
                hidden_dim, use_cuda,
            )?;
            Ok(Box::new(crate::selection::EmbedModel::new(embedder)))
        }),
    };

    // Scorer spec — loaded lazily with device from first "score_context" request
    let score_spec = ModelSpec {
        model_id: "score_context".to_string(),
        factory: Box::new(move |use_cuda| {
            let scorer = crate::selection::TorchScriptScorer::new(
                &scorer_path, hidden_dim, cross_attention, use_cuda,
            )?;
            Ok(Box::new(crate::selection::ContextScoreModel::new(Box::new(scorer))))
        }),
    };

    Ok(vec![embed_spec, score_spec])
}

// =============================================================================
// Model loading helpers
// =============================================================================

/// Load an encoder component for the split (attention/transformer) path.
#[cfg(feature = "ml")]
fn load_encoder(
    enc: &str,
    _model_name: &str,
    encoder_path: &std::path::Path,
    weights_dir: &std::path::Path,
    hidden_dim: usize,
    use_cuda: bool,
) -> Result<Box<dyn crate::selection::cached::ClauseEmbedder + Send>, String> {
    match enc {
        "gcn" | "gat" | "graphsage" => {
            let encoder = crate::selection::GcnEncoder::new(encoder_path, hidden_dim, use_cuda)?;
            Ok(Box::new(encoder))
        }
        "sentence" => {
            // Find tokenizer relative to encoder path
            let model_stem = encoder_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .strip_suffix("_encoder")
                .unwrap_or("");
            let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", model_stem));
            let emb = crate::selection::load_sentence_embedder(encoder_path, &tokenizer_path, use_cuda)?;
            Ok(Box::new(emb))
        }
        "features" => {
            let encoder = crate::selection::GcnEncoder::new(encoder_path, hidden_dim, use_cuda)?;
            Ok(Box::new(encoder))
        }
        _ => Err(format!("Unknown encoder for split path: '{}'", enc)),
    }
}

/// Load embedder + scorer for MLP configurations (combined model).
#[cfg(feature = "ml")]
fn load_embedder_scorer(
    enc: &str,
    model_path: &std::path::Path,
    weights_dir: &std::path::Path,
    use_cuda: bool,
) -> Result<(
    Box<dyn crate::selection::cached::ClauseEmbedder + Send>,
    Box<dyn crate::selection::cached::EmbeddingScorer + Send>,
), String> {
    match enc {
        "sentence" => {
            let model_stem = model_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", model_stem));
            let emb = crate::selection::load_sentence_embedder(model_path, &tokenizer_path, use_cuda)?;
            Ok((Box::new(emb), Box::new(crate::selection::PassThroughScorer) as _))
        }
        "gcn" | "gat" | "graphsage" => {
            let emb = crate::selection::load_gcn_embedder(model_path, use_cuda)?;
            Ok((Box::new(emb) as _, Box::new(crate::selection::GcnScorer) as _))
        }
        "features" => {
            let emb = crate::selection::load_gcn_embedder(model_path, use_cuda)?;
            Ok((Box::new(emb) as _, Box::new(crate::selection::GcnScorer) as _))
        }
        _ => Err(format!("Unknown encoder for MLP path: '{}'", enc)),
    }
}
