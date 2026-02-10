//! Remote clause selector that delegates scoring to a ScoringServer
//!
//! Communicates over a Unix domain socket. Not feature-gated — requires
//! no tch-rs dependency (pure socket I/O + serde).
//!
//! On connection errors (broken pipe), attempts to reconnect to the server
//! and re-initialize.  The server may still be alive even if this connection's
//! handler thread exited (e.g., because a previous worker on the same handler
//! was killed).  If reconnection fails repeatedly, the prover's timeout is
//! the safety net.

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
    /// Socket path for reconnection.
    socket_path: String,
    /// Interner symbols for re-initialization after reconnect.
    interner_symbols: Option<InternedSymbols>,
    /// Clause indices the server already has embeddings for.
    cached_indices: HashSet<usize>,
    /// Indices of clauses moved to the processed set.
    processed_indices: Vec<usize>,
    /// Softmax temperature (τ=1.0 default).
    temperature: f32,
    /// LCG RNG state (same algorithm as CachingSelector).
    rng_state: u64,
    /// Number of reconnections performed.
    reconnect_count: usize,
    /// Server permanently unavailable — select() returns None immediately.
    dead: bool,
    // Stats
    cache_hits: usize,
    cache_misses: usize,
    network_time: Duration,
}

/// Delay between retries, capped at this maximum.
const MAX_RETRY_DELAY: Duration = Duration::from_secs(2);

/// Maximum number of reconnection attempts before giving up.
const MAX_RECONNECTS: usize = 5;

impl RemoteSelector {
    /// Connect to a scoring server at the given socket path.
    ///
    /// Retries with exponential backoff for up to 15 seconds to handle the
    /// case where the server is restarting (e.g., after a crash).
    pub fn connect(socket_path: &str) -> Result<Self, String> {
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut delay = Duration::from_millis(500);
        let stream = loop {
            match Self::open_stream(socket_path) {
                Ok(s) => break s,
                Err(e) => {
                    if Instant::now() + delay > deadline {
                        return Err(e);
                    }
                    eprintln!(
                        "RemoteSelector: connect failed ({}), retrying in {:?}...",
                        e, delay
                    );
                    std::thread::sleep(delay);
                    delay = (delay * 2).min(Duration::from_secs(2));
                }
            }
        };

        Ok(Self {
            stream,
            socket_path: socket_path.to_string(),
            interner_symbols: None,
            cached_indices: HashSet::new(),
            processed_indices: Vec::new(),
            temperature: 1.0,
            rng_state: 12345,
            reconnect_count: 0,
            dead: false,
            cache_hits: 0,
            cache_misses: 0,
            network_time: Duration::ZERO,
        })
    }

    /// Open a new stream to the server with timeouts configured.
    fn open_stream(socket_path: &str) -> Result<UnixStream, String> {
        let stream = UnixStream::connect(socket_path)
            .map_err(|e| format!("Failed to connect to scoring server at {}: {}", socket_path, e))?;
        // Timeout must be long enough for mutex contention: with N workers
        // sharing one embedder, worst-case wait is N * embed_time. 120s handles
        // 8 workers with up to 15s per embed_batch. The prover's own timeout
        // (typically 600s) is the real safety net.
        stream
            .set_read_timeout(Some(Duration::from_secs(120)))
            .map_err(|e| format!("Failed to set read timeout: {}", e))?;
        stream
            .set_write_timeout(Some(Duration::from_secs(120)))
            .map_err(|e| format!("Failed to set write timeout: {}", e))?;
        Ok(stream)
    }

    /// Attempt to reconnect to the server and re-initialize.
    ///
    /// On success, replaces the stream, clears the embedding cache (the new
    /// handler thread has an empty cache), and re-sends Init if we have
    /// interner symbols.  Returns true on success.
    fn reconnect(&mut self) -> bool {
        self.reconnect_count += 1;
        if self.reconnect_count > MAX_RECONNECTS {
            eprintln!(
                "RemoteSelector: max reconnects ({}) exceeded, giving up",
                MAX_RECONNECTS
            );
            return false;
        }

        eprintln!(
            "RemoteSelector: reconnecting (attempt {}/{})",
            self.reconnect_count, MAX_RECONNECTS
        );

        match Self::open_stream(&self.socket_path) {
            Ok(new_stream) => {
                self.stream = new_stream;
                // New handler thread has empty cache
                self.cached_indices.clear();

                // Re-send Init if we have interner symbols
                if let Some(symbols) = &self.interner_symbols {
                    let req = ScoringRequest::Init {
                        interner: symbols.clone(),
                    };
                    match self.request(&req) {
                        Ok(ScoringResponse::InitOk) => {
                            eprintln!("RemoteSelector: reconnected and re-initialized");
                            true
                        }
                        Ok(resp) => {
                            eprintln!(
                                "RemoteSelector: reconnected but Init failed: {:?}",
                                resp
                            );
                            false
                        }
                        Err(e) => {
                            eprintln!(
                                "RemoteSelector: reconnected but Init request failed: {}",
                                e
                            );
                            false
                        }
                    }
                } else {
                    eprintln!("RemoteSelector: reconnected (no interner to re-send)");
                    true
                }
            }
            Err(e) => {
                eprintln!("RemoteSelector: reconnect failed: {}", e);
                false
            }
        }
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
        if unprocessed.is_empty() || self.dead {
            return None;
        }

        if unprocessed.len() == 1 {
            return unprocessed.shift_remove_index(0);
        }

        // Retry loop — on error, revert cache marks and retry with backoff.
        // On connection error, attempt reconnect before retrying.
        // The prover's timeout is the safety net for persistent failures.
        let mut retry_delay = Duration::from_millis(100);

        // Cap uncached clauses per request to prevent overwhelming the server
        // after a reconnect (when the entire unprocessed set is uncached).
        // The server returns a default score (0.0) for clauses not yet embedded.
        const MAX_UNCACHED_PER_REQUEST: usize = 512;

        let scores = loop {
            // Find uncached clauses (recomputed each attempt since cache
            // marks are reverted on error), capped to prevent large batches
            let uncached_indices: Vec<usize> = unprocessed
                .iter()
                .copied()
                .filter(|idx| !self.cached_indices.contains(idx))
                .take(MAX_UNCACHED_PER_REQUEST)
                .collect();

            // Tentatively mark as cached (reverted on error)
            for &idx in &uncached_indices {
                self.cached_indices.insert(idx);
            }

            let uncached: Vec<(usize, Clause)> = uncached_indices
                .iter()
                .map(|&idx| (idx, clauses[idx].clone()))
                .collect();

            let req = ScoringRequest::Score {
                uncached,
                unprocessed_indices: unprocessed.iter().copied().collect(),
                processed_indices: self.processed_indices.clone(),
            };

            match self.request(&req) {
                Ok(ScoringResponse::Scores(s)) if s.len() == unprocessed.len() => {
                    // Success — record cache stats and reset reconnect counter
                    self.cache_hits += unprocessed.len() - uncached_indices.len();
                    self.cache_misses += uncached_indices.len();
                    self.reconnect_count = 0;
                    break s;
                }
                Ok(ScoringResponse::Scores(s)) => {
                    eprintln!(
                        "RemoteSelector: score count mismatch ({} vs {}), retrying...",
                        s.len(),
                        unprocessed.len()
                    );
                }
                Ok(ScoringResponse::Error(e)) => {
                    eprintln!("RemoteSelector: server error: {}, retrying...", e);
                }
                Ok(_) => {
                    eprintln!("RemoteSelector: unexpected response, retrying...");
                }
                Err(e) => {
                    // Connection error (broken pipe, etc.) — the handler thread
                    // for this connection is dead, but the server may still be
                    // alive.  Attempt to reconnect.
                    eprintln!("RemoteSelector: connection error: {}", e);

                    // Revert cache marks before reconnect clears them
                    for &idx in &uncached_indices {
                        self.cached_indices.remove(&idx);
                    }

                    if !self.reconnect() {
                        eprintln!("RemoteSelector: cannot recover, giving up");
                        self.dead = true;
                        return None;
                    }
                    // Reconnect succeeded — retry immediately (cache already cleared)
                    retry_delay = Duration::from_millis(100);
                    continue;
                }
            }

            // Revert cache marks — server didn't embed these
            for &idx in &uncached_indices {
                self.cached_indices.remove(&idx);
            }

            std::thread::sleep(retry_delay);
            retry_delay = (retry_delay * 2).min(MAX_RETRY_DELAY);
        };

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
        self.rng_state = 12345;
        self.reconnect_count = 0;
        self.dead = false;
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
        // Store for re-initialization after reconnect
        self.interner_symbols = Some(symbols.clone());
        let req = ScoringRequest::Init { interner: symbols };

        let mut retry_delay = Duration::from_millis(100);
        loop {
            match self.request(&req) {
                Ok(ScoringResponse::InitOk) => {
                    return;
                }
                Ok(ScoringResponse::Error(e)) => {
                    eprintln!("RemoteSelector: Init error: {}, retrying...", e);
                }
                Ok(_) => {
                    eprintln!("RemoteSelector: unexpected Init response, retrying...");
                }
                Err(e) => {
                    eprintln!("RemoteSelector: Init connection error: {}", e);
                    if !self.reconnect() {
                        eprintln!("RemoteSelector: Init cannot recover, giving up");
                        self.dead = true;
                    }
                    return;
                }
            }
            std::thread::sleep(retry_delay);
            retry_delay = (retry_delay * 2).min(MAX_RETRY_DELAY);
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
