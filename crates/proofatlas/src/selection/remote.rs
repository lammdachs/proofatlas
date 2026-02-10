//! Remote clause selector that delegates scoring to a ScoringServer
//!
//! Communicates over a Unix domain socket. Not feature-gated — requires
//! no tch-rs dependency (pure socket I/O + serde).

use std::collections::HashSet;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use indexmap::IndexSet;

use super::clause::{ClauseSelector, SelectorStats};
use super::protocol::{
    read_message, write_message, InternedSymbols, ScoringRequest, ScoringResponse,
};
use crate::logic::{Clause, Interner};

/// Clause selector that delegates scoring to a remote ScoringServer.
///
/// Maintains a local set of cached clause indices (tracking which clauses
/// the server has already embedded) and applies softmax sampling locally.
pub struct RemoteSelector {
    stream: UnixStream,
    /// Clause indices the server already has embeddings for.
    cached_indices: HashSet<usize>,
    /// Indices of clauses moved to the processed set.
    processed_indices: Vec<usize>,
    /// Whether Init has been sent for the current problem.
    initialized: bool,
    /// Softmax temperature (τ=1.0 default).
    temperature: f32,
    /// LCG RNG state (same algorithm as CachingSelector).
    rng_state: u64,
    // Stats
    cache_hits: usize,
    cache_misses: usize,
    network_time: Duration,
}

impl RemoteSelector {
    /// Connect to a scoring server at the given socket path.
    pub fn connect(socket_path: &str) -> Result<Self, String> {
        let stream = UnixStream::connect(socket_path)
            .map_err(|e| format!("Failed to connect to scoring server at {}: {}", socket_path, e))?;

        // Set timeouts to avoid indefinite hangs
        stream
            .set_read_timeout(Some(Duration::from_secs(300)))
            .map_err(|e| format!("Failed to set read timeout: {}", e))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(300)))
            .map_err(|e| format!("Failed to set write timeout: {}", e))?;

        Ok(Self {
            stream,
            cached_indices: HashSet::new(),
            processed_indices: Vec::new(),
            initialized: false,
            temperature: 1.0,
            rng_state: 12345,
            cache_hits: 0,
            cache_misses: 0,
            network_time: Duration::ZERO,
        })
    }

    /// LCG random number generator (identical to CachingSelector).
    fn next_random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Send a request and receive a response, tracking network time.
    fn request(&mut self, req: &ScoringRequest) -> Result<ScoringResponse, String> {
        let t0 = Instant::now();
        write_message(&mut self.stream, req)
            .map_err(|e| format!("Failed to send request: {}", e))?;
        let resp: ScoringResponse = read_message(&mut self.stream)
            .map_err(|e| format!("Failed to read response: {}", e))?;
        self.network_time += t0.elapsed();
        Ok(resp)
    }
}

impl ClauseSelector for RemoteSelector {
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Clause]) -> Option<usize> {
        if unprocessed.is_empty() {
            return None;
        }

        if unprocessed.len() == 1 {
            return unprocessed.shift_remove_index(0);
        }

        // Find uncached clauses
        let uncached: Vec<(usize, Clause)> = unprocessed
            .iter()
            .copied()
            .filter(|idx| !self.cached_indices.contains(idx))
            .map(|idx| (idx, clauses[idx].clone()))
            .collect();

        // Track cache stats
        let cached_count = unprocessed.len() - uncached.len();
        self.cache_hits += cached_count;
        self.cache_misses += uncached.len();

        // Mark newly sent clauses as cached
        for &(idx, _) in &uncached {
            self.cached_indices.insert(idx);
        }

        // Build request
        let unprocessed_indices: Vec<usize> = unprocessed.iter().copied().collect();
        let req = ScoringRequest::Score {
            uncached,
            unprocessed_indices,
            processed_indices: self.processed_indices.clone(),
        };

        // Send and receive scores
        let scores = match self.request(&req) {
            Ok(ScoringResponse::Scores(s)) => s,
            Ok(ScoringResponse::Error(e)) => {
                eprintln!("RemoteSelector: server error: {}", e);
                return unprocessed.shift_remove_index(0);
            }
            Ok(_) => {
                eprintln!("RemoteSelector: unexpected response");
                return unprocessed.shift_remove_index(0);
            }
            Err(e) => {
                eprintln!("RemoteSelector: connection error: {}", e);
                return unprocessed.shift_remove_index(0);
            }
        };

        if scores.len() != unprocessed.len() {
            eprintln!(
                "RemoteSelector: score count mismatch ({} vs {})",
                scores.len(),
                unprocessed.len()
            );
            return unprocessed.shift_remove_index(0);
        }

        // Softmax sampling with temperature (identical to CachingSelector)
        let tau = self.temperature;
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f64> = scores
            .iter()
            .map(|&s| (((s - max_score) / tau) as f64).exp())
            .collect();
        let sum: f64 = exp_scores.iter().sum();
        let probs: Vec<f64> = exp_scores.iter().map(|&e| e / sum).collect();

        // Sample
        let r = self.next_random();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return unprocessed.shift_remove_index(i);
            }
        }

        unprocessed.pop()
    }

    fn name(&self) -> &str {
        "remote_selector"
    }

    fn reset(&mut self) {
        self.cached_indices.clear();
        self.processed_indices.clear();
        self.initialized = false;
        self.rng_state = 12345;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.network_time = Duration::ZERO;

        // Best-effort reset on server
        let _ = self.request(&ScoringRequest::Reset);
    }

    fn on_clause_processed(&mut self, clause_idx: usize) {
        self.processed_indices.push(clause_idx);
    }

    fn stats(&self) -> Option<SelectorStats> {
        Some(SelectorStats {
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            embed_time: self.network_time,
            score_time: Duration::ZERO,
        })
    }

    fn set_interner(&mut self, interner: Arc<Interner>) {
        let symbols = InternedSymbols::from_interner(&interner);
        match self.request(&ScoringRequest::Init {
            interner: symbols,
        }) {
            Ok(ScoringResponse::InitOk) => {
                self.initialized = true;
            }
            Ok(ScoringResponse::Error(e)) => {
                eprintln!("RemoteSelector: Init error: {}", e);
            }
            Ok(_) => {
                eprintln!("RemoteSelector: unexpected Init response");
            }
            Err(e) => {
                eprintln!("RemoteSelector: Init failed: {}", e);
            }
        }
    }
}

impl Drop for RemoteSelector {
    fn drop(&mut self) {
        // Best-effort shutdown
        let _ = write_message(&mut self.stream, &ScoringRequest::Shutdown);
    }
}

#[cfg(test)]
#[cfg(feature = "ml")]
mod tests {
    use super::*;
    use crate::logic::{Constant, Literal, PredicateSymbol, Term};
    use crate::selection::cached::{ClauseEmbedder, EmbeddingScorer};
    use crate::selection::server::ScoringServer;
    use std::thread;
    use std::time::Duration;

    /// Simple test embedder that returns clause length as embedding
    struct TestEmbedder;

    impl ClauseEmbedder for TestEmbedder {
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
            "test"
        }
    }

    /// Simple test scorer that returns embedding value as score
    struct TestScorer;

    impl EmbeddingScorer for TestScorer {
        fn score_batch(&self, embeddings: &[&[f32]]) -> Vec<f32> {
            embeddings.iter().map(|e| e[0]).collect()
        }

        fn name(&self) -> &str {
            "test"
        }
    }

    fn make_clause(num_literals: usize, interner: &mut Interner) -> Clause {
        let p = PredicateSymbol {
            id: interner.intern_predicate("P"),
            arity: 1,
        };
        let a = Term::Constant(Constant {
            id: interner.intern_constant("a"),
        });
        let literals: Vec<Literal> = (0..num_literals)
            .map(|_| Literal::positive(p.clone(), vec![a.clone()]))
            .collect();
        Clause::new(literals)
    }

    #[test]
    fn test_remote_selector_integration() {
        let socket_path = format!("/tmp/proofatlas-test-{}.sock", std::process::id());
        let socket_path_clone = socket_path.clone();

        // Start server in background
        let server = ScoringServer::new(
            Box::new(TestEmbedder),
            Box::new(TestScorer),
            socket_path_clone,
        );
        let _handle = server.spawn();

        // Wait for server to bind
        thread::sleep(Duration::from_millis(100));

        // Connect client
        let mut selector = RemoteSelector::connect(&socket_path).expect("connect failed");

        // Create test data
        let mut interner = Interner::new();
        let clauses = vec![
            make_clause(1, &mut interner),
            make_clause(2, &mut interner),
            make_clause(3, &mut interner),
        ];

        // Set interner
        selector.set_interner(Arc::new(interner));

        // Select a clause
        let mut unprocessed: IndexSet<usize> = (0..3).collect();
        let selected = selector.select(&mut unprocessed, &clauses);
        assert!(selected.is_some());
        assert_eq!(unprocessed.len(), 2);

        // Mark one as processed
        if let Some(idx) = selected {
            selector.on_clause_processed(idx);
        }

        // Select again
        let selected2 = selector.select(&mut unprocessed, &clauses);
        assert!(selected2.is_some());
        assert_eq!(unprocessed.len(), 1);

        // Check stats
        let stats = selector.stats().unwrap();
        assert!(stats.cache_hits + stats.cache_misses > 0);

        // Reset
        selector.reset();

        // Clean up
        let _ = std::fs::remove_file(&socket_path);
    }
}
