//! Scoring server for ML clause selectors
//!
//! Runs model inference in a dedicated process/thread. Workers connect
//! over Unix domain sockets and send clause data; the server returns scores.
//! Each connection maintains its own embedding cache.

#[cfg(feature = "ml")]
use std::collections::HashMap;
#[cfg(feature = "ml")]
use std::os::unix::net::UnixListener;
#[cfg(feature = "ml")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "ml")]
use std::thread;

#[cfg(feature = "ml")]
use super::cached::{ClauseEmbedder, EmbeddingScorer};
#[cfg(feature = "ml")]
use super::protocol::{
    read_message, write_message, ScoringRequest, ScoringResponse,
};
#[cfg(feature = "ml")]
use crate::logic::Interner;

/// A scoring server that accepts connections over a Unix domain socket.
///
/// Owns the embedder and scorer behind `Arc<Mutex<>>` since `tch::CModule`
/// is `!Sync`. Each connection gets its own thread and embedding cache.
#[cfg(feature = "ml")]
pub struct ScoringServer {
    embedder: Arc<Mutex<Box<dyn ClauseEmbedder>>>,
    scorer: Arc<Mutex<Box<dyn EmbeddingScorer>>>,
    socket_path: String,
}

#[cfg(feature = "ml")]
impl ScoringServer {
    /// Create a new scoring server.
    ///
    /// # Arguments
    /// * `embedder` - The clause embedder (e.g., GcnEmbedder, SentenceEmbedder)
    /// * `scorer` - The embedding scorer (e.g., GcnScorer, TorchScriptScorer)
    /// * `socket_path` - Path for the Unix domain socket
    pub fn new(
        embedder: Box<dyn ClauseEmbedder>,
        scorer: Box<dyn EmbeddingScorer>,
        socket_path: String,
    ) -> Self {
        Self {
            embedder: Arc::new(Mutex::new(embedder)),
            scorer: Arc::new(Mutex::new(scorer)),
            socket_path,
        }
    }

    /// Run the server, blocking the current thread.
    ///
    /// Binds to the socket path, accepts connections, and spawns a thread
    /// per connection. The server continues accepting even if individual
    /// connections fail.
    pub fn run(&self) {
        // Install SIGSEGV handler to capture backtrace before libtorch's handler eats it.
        // This runs AFTER model loading (CUDA init), so it overrides libtorch's handler.
        install_crash_handler();

        // Remove stale socket file
        let _ = std::fs::remove_file(&self.socket_path);

        let listener = match UnixListener::bind(&self.socket_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("ScoringServer: failed to bind {}: {}", self.socket_path, e);
                return;
            }
        };

        eprintln!("ScoringServer: listening on {}", self.socket_path);

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let embedder = Arc::clone(&self.embedder);
                    let scorer = Arc::clone(&self.scorer);
                    // Use 16 MiB stack (default is 2 MiB). libtorch's sparse
                    // tensor ops and GCN forward pass need deep stack frames;
                    // large clause batches can overflow the default.
                    let res = thread::Builder::new()
                        .stack_size(16 * 1024 * 1024)
                        .spawn(move || {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            handle_connection(stream, embedder, scorer)
                        })) {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => {
                                eprintln!("ScoringServer: connection error: {}", e);
                            }
                            Err(panic) => {
                                let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                                    (*s).to_string()
                                } else if let Some(s) = panic.downcast_ref::<String>() {
                                    s.clone()
                                } else {
                                    "unknown panic".to_string()
                                };
                                eprintln!("ScoringServer: connection handler panicked: {}", msg);
                            }
                        }
                    });
                    if let Err(e) = res {
                        eprintln!("ScoringServer: failed to spawn handler thread: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("ScoringServer: accept error: {}", e);
                }
            }
        }
    }

    /// Spawn the server in a background thread and return the join handle.
    pub fn spawn(self) -> thread::JoinHandle<()> {
        thread::spawn(move || self.run())
    }
}

/// Install a SIGSEGV/SIGBUS/SIGABRT handler that prints a native backtrace.
///
/// libtorch installs its own SIGSEGV handler during library loading and CUDA init.
/// This function overrides it so we can capture crash diagnostics.
/// Uses only async-signal-safe functions (write, backtrace, backtrace_symbols_fd).
#[cfg(feature = "ml")]
fn install_crash_handler() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return; // Already installed
    }

    unsafe extern "C" fn crash_handler(sig: libc::c_int) {
        // Write header (async-signal-safe: raw write to fd 2)
        let header = match sig {
            libc::SIGSEGV => b"\n=== ScoringServer CRASH: SIGSEGV (signal 11) ===\nBacktrace:\n" as &[u8],
            libc::SIGBUS => b"\n=== ScoringServer CRASH: SIGBUS (signal 7) ===\nBacktrace:\n",
            libc::SIGABRT => b"\n=== ScoringServer CRASH: SIGABRT (signal 6) ===\nBacktrace:\n",
            _ => b"\n=== ScoringServer CRASH: unknown signal ===\nBacktrace:\n",
        };
        libc::write(2, header.as_ptr() as *const libc::c_void, header.len());

        // Capture backtrace (async-signal-safe on Linux/glibc)
        let mut buffer = [std::ptr::null_mut::<libc::c_void>(); 128];
        let depth = libc::backtrace(buffer.as_mut_ptr(), 128);
        libc::backtrace_symbols_fd(buffer.as_ptr(), depth, 2);

        let footer = b"=== END CRASH ===\n";
        libc::write(2, footer.as_ptr() as *const libc::c_void, footer.len());

        // Re-raise with default handler to produce core dump / proper exit
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }

    unsafe {
        libc::signal(libc::SIGSEGV, crash_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGBUS, crash_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGABRT, crash_handler as *const () as libc::sighandler_t);
    }
    eprintln!("ScoringServer: crash handler installed (overrides libtorch signal handlers)");
}

/// Lock a mutex, recovering from poison (a prior thread panicked while holding it).
///
/// This is critical for the server: if one connection's inference call panics
/// (e.g., CUDA OOM), the mutex gets poisoned. Without recovery, ALL other
/// connections would fail on their next lock attempt, cascading the failure
/// across every worker.
#[cfg(feature = "ml")]
fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        eprintln!("ScoringServer: recovering from poisoned mutex");
        poisoned.into_inner()
    })
}

/// Handle a single client connection.
///
/// Maintains a per-connection embedding cache and interner. Processes
/// requests sequentially until Shutdown or a connection error.
#[cfg(feature = "ml")]
fn handle_connection(
    stream: std::os::unix::net::UnixStream,
    embedder: Arc<Mutex<Box<dyn ClauseEmbedder>>>,
    scorer: Arc<Mutex<Box<dyn EmbeddingScorer>>>,
) -> std::io::Result<()> {
    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream;

    let mut cache: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut _interner: Option<Interner> = None;

    loop {
        let request: ScoringRequest = match read_message(&mut reader) {
            Ok(req) => req,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };

        let response = match request {
            ScoringRequest::Init {
                interner: symbols,
            } => {
                let new_interner = symbols.to_interner();
                {
                    let mut emb = lock_or_recover(&embedder);
                    emb.set_interner(Arc::new(new_interner.clone()));
                }
                _interner = Some(new_interner);
                cache.clear();
                ScoringResponse::InitOk
            }

            ScoringRequest::Score {
                uncached,
                unprocessed_indices,
                processed_indices,
            } => {
                // Embed uncached clauses in chunks to cap per-call GPU memory.
                // Large problems (e.g., PUZ021-1.p) can accumulate thousands of
                // uncached clauses between selections.
                const MAX_EMBED_BATCH: usize = 512;

                let embed_result = if !uncached.is_empty() {
                    if uncached.len() > MAX_EMBED_BATCH {
                        eprintln!(
                            "ScoringServer: large batch ({} uncached), chunking",
                            uncached.len()
                        );
                    }
                    let clause_refs: Vec<&crate::logic::Clause> =
                        uncached.iter().map(|(_, c)| c).collect();
                    let mut all_embeddings = Vec::with_capacity(clause_refs.len());
                    let mut failed = false;
                    for chunk in clause_refs.chunks(MAX_EMBED_BATCH) {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let emb = lock_or_recover(&embedder);
                            emb.embed_batch(chunk)
                        })) {
                            Ok(embeddings) => all_embeddings.extend(embeddings),
                            Err(_) => {
                                failed = true;
                                break;
                            }
                        }
                    }
                    if failed {
                        Err(Box::new("embedder panic") as Box<dyn std::any::Any + Send>)
                    } else {
                        Ok(all_embeddings)
                    }
                } else {
                    Ok(vec![])
                };

                match embed_result {
                    Ok(embeddings) => {
                        for ((idx, _), embedding) in uncached.into_iter().zip(embeddings.into_iter()) {
                            cache.insert(idx, embedding);
                        }

                        // Collect embeddings for cached unprocessed clauses only
                        let cached_u: Vec<(usize, &[f32])> = unprocessed_indices
                            .iter()
                            .enumerate()
                            .filter_map(|(pos, idx)| {
                                cache.get(idx).map(|e| (pos, e.as_slice()))
                            })
                            .collect();

                        let p_embeddings: Vec<&[f32]> = processed_indices
                            .iter()
                            .filter_map(|idx| cache.get(idx).map(|e| e.as_slice()))
                            .collect();

                        let u_emb_refs: Vec<&[f32]> = cached_u.iter().map(|(_, e)| *e).collect();
                        let score_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let sc = lock_or_recover(&scorer);
                            if sc.uses_context() && !p_embeddings.is_empty() {
                                sc.score_with_context(&u_emb_refs, &p_embeddings)
                            } else {
                                sc.score_batch(&u_emb_refs)
                            }
                        }));

                        match score_result {
                            Ok(cached_scores) => {
                                // Build full score vector: cached clauses get their
                                // scores, uncached clauses get 0.0 (neutral for softmax).
                                let num_unprocessed = unprocessed_indices.len();
                                let mut scores = vec![0.0f32; num_unprocessed];
                                for ((pos, _), score) in cached_u.iter().zip(cached_scores.iter()) {
                                    scores[*pos] = *score;
                                }
                                ScoringResponse::Scores(scores)
                            }
                            Err(_) => {
                                eprintln!("ScoringServer: scorer panicked, returning error");
                                ScoringResponse::Error("scorer panic".into())
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!("ScoringServer: embedder panicked, returning error");
                        ScoringResponse::Error("embedder panic".into())
                    }
                }
            }

            ScoringRequest::Reset => {
                cache.clear();
                _interner = None;
                ScoringResponse::ResetOk
            }

            ScoringRequest::Shutdown => {
                write_message(&mut writer, &ScoringResponse::ResetOk)?;
                return Ok(());
            }
        };

        write_message(&mut writer, &response)?;
    }
}
