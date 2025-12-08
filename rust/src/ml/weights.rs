//! Model weight downloading and caching.
//!
//! Weights are searched in order:
//! 1. Project-local `.weights/` directory (for development)
//! 2. Global cache `~/.cache/proofatlas/models/` (for pip installs)
//!
//! If not found and the "download" feature is enabled, weights are downloaded
//! from GitHub Releases to the global cache.

use std::fs;
use std::io;
use std::path::PathBuf;
use thiserror::Error;

#[cfg(feature = "download")]
use sha2::{Digest, Sha256};
#[cfg(feature = "download")]
use std::fs::File;
#[cfg(feature = "download")]
use std::io::{BufReader, Read, Write};
#[cfg(feature = "download")]
use std::path::Path;

/// Base URL for downloading model weights from GitHub Releases
#[cfg(feature = "download")]
const GITHUB_RELEASE_URL: &str =
    "https://github.com/lexpk/proofatlas/releases/download";

/// Default version tag for model downloads
#[cfg(feature = "download")]
const DEFAULT_VERSION: &str = "v0.2.0";

#[derive(Error, Debug)]
pub enum WeightError {
    #[error("Model not found: {0}")]
    NotFound(String),

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("No cache directory available")]
    NoCacheDir,

    #[error("Download feature not enabled")]
    DownloadDisabled,
}

/// Model metadata for downloading and verification
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name (e.g., "gcn_selector")
    pub name: String,
    /// Expected SHA256 hash of the file (hex string), or None to skip verification
    pub sha256: Option<String>,
    /// Version tag for GitHub release
    #[cfg(feature = "download")]
    pub version: String,
}

impl ModelInfo {
    #[cfg(feature = "download")]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sha256: None,
            version: DEFAULT_VERSION.to_string(),
        }
    }

    #[cfg(not(feature = "download"))]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sha256: None,
        }
    }

    #[cfg(feature = "download")]
    pub fn with_version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }

    pub fn with_sha256(mut self, hash: &str) -> Self {
        self.sha256 = Some(hash.to_string());
        self
    }

    /// Get the download URL for this model
    #[cfg(feature = "download")]
    pub fn download_url(&self) -> String {
        format!(
            "{}/{}/{}.safetensors",
            GITHUB_RELEASE_URL, self.version, self.name
        )
    }

    /// Get the filename for this model
    pub fn filename(&self) -> String {
        format!("{}.safetensors", self.name)
    }
}

/// Get the global cache directory for model weights
#[cfg(feature = "download")]
pub fn global_cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|p| p.join("proofatlas").join("models"))
}

#[cfg(not(feature = "download"))]
pub fn global_cache_dir() -> Option<PathBuf> {
    // Without dirs crate, try common locations
    if let Ok(home) = std::env::var("HOME") {
        Some(PathBuf::from(home).join(".cache").join("proofatlas").join("models"))
    } else {
        None
    }
}

/// Get the project-local weights directory
pub fn local_weights_dir() -> PathBuf {
    PathBuf::from(".weights")
}

/// Find a model weight file, checking local and global cache.
/// If the exact name is not found, looks for the latest iteration (e.g., gcn_iter_5.safetensors).
pub fn find_model(name: &str) -> Option<PathBuf> {
    let filename = format!("{}.safetensors", name);

    // 1. Check project-local .weights/ (development)
    let local = local_weights_dir().join(&filename);
    if local.exists() {
        return Some(local);
    }

    // 2. Check global cache (pip installs)
    if let Some(cache_dir) = global_cache_dir() {
        let global = cache_dir.join(&filename);
        if global.exists() {
            return Some(global);
        }
    }

    // 3. Look for latest iteration (e.g., gcn_iter_5.safetensors)
    if let Some(path) = find_latest_iteration(name, &local_weights_dir()) {
        return Some(path);
    }

    if let Some(cache_dir) = global_cache_dir() {
        if let Some(path) = find_latest_iteration(name, &cache_dir) {
            return Some(path);
        }
    }

    None
}

/// Find the latest iteration of a model (e.g., gcn_iter_5.safetensors for "gcn")
fn find_latest_iteration(name: &str, dir: &Path) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }

    let prefix = format!("{}_iter_", name);
    let mut latest_iter: Option<(u32, PathBuf)> = None;

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if stem.starts_with(&prefix) {
                    // Extract iteration number
                    if let Ok(iter_num) = stem[prefix.len()..].parse::<u32>() {
                        match &latest_iter {
                            None => latest_iter = Some((iter_num, path)),
                            Some((best, _)) if iter_num > *best => {
                                latest_iter = Some((iter_num, path))
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    latest_iter.map(|(_, path)| path)
}

/// Get a model, downloading if necessary (requires "download" feature)
#[cfg(feature = "download")]
pub fn get_model(info: &ModelInfo) -> Result<PathBuf, WeightError> {
    // Check if already available
    if let Some(path) = find_model(&info.name) {
        // Verify hash if provided
        if let Some(expected_hash) = &info.sha256 {
            let actual_hash = file_sha256(&path)?;
            if &actual_hash != expected_hash {
                eprintln!(
                    "Warning: Local model {} has different hash, re-downloading",
                    info.name
                );
                // Fall through to download
            } else {
                return Ok(path);
            }
        } else {
            return Ok(path);
        }
    }

    // Download to global cache
    download_model(info)
}

/// Get a model (without download support)
#[cfg(not(feature = "download"))]
pub fn get_model(info: &ModelInfo) -> Result<PathBuf, WeightError> {
    find_model(&info.name).ok_or_else(|| WeightError::NotFound(info.name.clone()))
}

/// Download a model to the global cache
#[cfg(feature = "download")]
pub fn download_model(info: &ModelInfo) -> Result<PathBuf, WeightError> {
    let cache_dir = global_cache_dir().ok_or(WeightError::NoCacheDir)?;
    fs::create_dir_all(&cache_dir)?;

    let dest_path = cache_dir.join(info.filename());
    let url = info.download_url();

    eprintln!("Downloading model: {}", info.name);
    eprintln!("  URL: {}", url);
    eprintln!("  Destination: {}", dest_path.display());

    // Download with progress
    let response = ureq::get(&url)
        .call()
        .map_err(|e| WeightError::DownloadFailed(e.to_string()))?;

    let content_length = response
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    // Write to temporary file first
    let temp_path = dest_path.with_extension("safetensors.tmp");
    let mut file = File::create(&temp_path)?;
    let mut reader = response.into_reader();

    let mut buffer = [0u8; 8192];
    let mut downloaded: u64 = 0;
    let mut hasher = Sha256::new();

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        file.write_all(&buffer[..bytes_read])?;
        hasher.update(&buffer[..bytes_read]);
        downloaded += bytes_read as u64;

        // Progress indicator
        if let Some(total) = content_length {
            let percent = (downloaded * 100) / total;
            eprint!("\r  Progress: {}% ({}/{})", percent, downloaded, total);
        } else {
            eprint!("\r  Downloaded: {} bytes", downloaded);
        }
    }
    eprintln!(); // newline after progress

    file.flush()?;
    drop(file);

    // Verify hash if provided
    let actual_hash = format!("{:x}", hasher.finalize());
    if let Some(expected_hash) = &info.sha256 {
        if &actual_hash != expected_hash {
            fs::remove_file(&temp_path)?;
            return Err(WeightError::HashMismatch {
                expected: expected_hash.clone(),
                actual: actual_hash,
            });
        }
    }

    // Move to final location
    fs::rename(&temp_path, &dest_path)?;
    eprintln!("  SHA256: {}", actual_hash);
    eprintln!("  Done!");

    Ok(dest_path)
}

#[cfg(not(feature = "download"))]
pub fn download_model(_info: &ModelInfo) -> Result<PathBuf, WeightError> {
    Err(WeightError::DownloadDisabled)
}

/// Calculate SHA256 hash of a file
#[cfg(feature = "download")]
pub fn file_sha256(path: &Path) -> Result<String, WeightError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();

    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// List all models in the cache directories
pub fn list_cached_models() -> Vec<(PathBuf, String)> {
    let mut models = Vec::new();

    // Check local weights
    if let Ok(entries) = fs::read_dir(local_weights_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    let name = name.to_string();
                    models.push((path, name));
                }
            }
        }
    }

    // Check global cache
    if let Some(cache_dir) = global_cache_dir() {
        if let Ok(entries) = fs::read_dir(cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "safetensors") {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        let name = name.to_string();
                        // Avoid duplicates
                        if !models.iter().any(|(_, n)| n == &name) {
                            models.push((path, name));
                        }
                    }
                }
            }
        }
    }

    models
}

/// Clear the global model cache
pub fn clear_cache() -> Result<(), WeightError> {
    if let Some(cache_dir) = global_cache_dir() {
        if cache_dir.exists() {
            fs::remove_dir_all(&cache_dir)?;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "download"))]
mod tests {
    use super::*;

    #[test]
    fn test_model_info() {
        let info = ModelInfo::new("gcn_selector").with_version("v0.2.0");
        assert_eq!(info.filename(), "gcn_selector.safetensors");
        assert_eq!(
            info.download_url(),
            "https://github.com/lexpk/proofatlas/releases/download/v0.2.0/gcn_selector.safetensors"
        );
    }

    #[test]
    fn test_cache_dir() {
        let cache = global_cache_dir();
        assert!(cache.is_some());
        let path = cache.unwrap();
        assert!(path.ends_with("proofatlas/models"));
    }
}
