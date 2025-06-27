//! Error types for ProofAtlas

use thiserror::Error;
use std::io;

#[derive(Error, Debug)]
pub enum ProofAtlasError {
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Include file not found: {0}")]
    IncludeNotFound(String),
    
    #[error("Circular include detected: {0}")]
    CircularInclude(String),
    
    #[error("Depth limit exceeded")]
    DepthLimitExceeded,
}

pub type Result<T> = std::result::Result<T, ProofAtlasError>;