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
        /// If true, the preprocessor converts clauses to strings (sentence models).
        string_input: bool,
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
            SinkKind::Pipeline { handle, string_input } => {
                let preprocess: crate::selection::Preprocessor = if *string_input {
                    // Sentence path: convert clause→String using interner in data processing thread
                    let int = Arc::new(interner.clone());
                    Box::new(move |clause: Arc<crate::logic::Clause>| {
                        let s = clause.display(&int).to_string();
                        Box::new(s) as Box<dyn std::any::Any + Send>
                    })
                } else {
                    // GCN path: pass clause through unchanged
                    crate::selection::identity_preprocessor()
                };
                Ok(Box::new(crate::selection::create_pipeline_with_handle(
                    handle.clone(),
                    self.temperature,
                    "ml_pipeline".to_string(),
                    preprocess,
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

    /// Set the ML encoder type (e.g., "gcn", "gat", "graphsage", "sentence").
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
            Some(enc @ ("gcn" | "gat" | "graphsage" | "sentence")) => {
                let scorer_name = self.scorer.as_deref()
                    .ok_or_else(|| "scorer required when encoder is set".to_string())?;
                let model_name = format!("{}_{}", enc, scorer_name);

                let weights_dir = std::path::PathBuf::from(
                    self.weights_path.as_deref().unwrap_or(".weights")
                );

                let string_input = enc == "sentence";

                let (embedder, scorer_box): (
                    Box<dyn crate::selection::cached::ClauseEmbedder + Send>,
                    Box<dyn crate::selection::cached::EmbeddingScorer + Send>,
                ) = if string_input {
                    let model_path = weights_dir.join(format!("{}.pt", model_name));
                    let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", model_name));
                    if !model_path.exists() {
                        return Err(format!("Model not found at {}", model_path.display()));
                    }
                    let emb = crate::selection::load_sentence_embedder(
                        &model_path, &tokenizer_path, self.use_cuda,
                    )?;
                    (Box::new(emb), Box::new(crate::selection::PassThroughScorer) as _)
                } else {
                    let model_path = weights_dir.join(format!("{}.pt", model_name));
                    if !model_path.exists() {
                        return Err(format!("Model not found at {}", model_path.display()));
                    }
                    let emb = crate::selection::load_gcn_embedder(
                        &model_path, self.use_cuda,
                    )?;
                    (Box::new(emb) as _, Box::new(crate::selection::GcnScorer) as _)
                };

                let model = crate::selection::EmbedScoreModel::new(embedder, scorer_box);
                let backend = crate::selection::Backend::new(vec![Box::new(model)]);
                let handle = backend.handle();

                Ok(ProofAtlas {
                    config: self.config,
                    include_dirs: self.include_dirs,
                    sink_kind: SinkKind::Pipeline { handle, string_input },
                    temperature: self.temperature,
                    _backend: Some(backend),
                })
            }
            Some(other) => {
                #[cfg(feature = "ml")]
                let available = "None, 'gcn', 'gat', 'graphsage', or 'sentence'";
                #[cfg(not(feature = "ml"))]
                let available = "None (ML features not enabled)";
                Err(format!("Unknown encoder: '{}'. Use {}", other, available))
            }
        }
    }
}
