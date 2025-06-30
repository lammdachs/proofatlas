//! TPTP file format handler
//! 
//! This module provides high-level APIs for working with TPTP files

use crate::core::Problem;
use crate::parsing::{self, tptp_parser::ParseError};
use std::path::Path;

type Result<T> = std::result::Result<T, ParseError>;

/// TPTP format handler
pub struct TPTPFormat {
    include_path: Option<String>,
}

impl TPTPFormat {
    pub fn new() -> Self {
        TPTPFormat { include_path: None }
    }
    
    pub fn with_include_path(include_path: String) -> Self {
        TPTPFormat { include_path: Some(include_path) }
    }
    
    /// Parse a TPTP file
    pub fn parse_file(&self, file_path: &Path) -> Result<Problem> {
        let path_str = file_path.to_str()
            .ok_or_else(|| ParseError::ParseError(
                "Invalid file path".to_string()
            ))?;
        
        parsing::tptp_parser::parse_file(path_str, self.include_path.as_deref())
    }
    
    /// Parse TPTP content from string
    pub fn parse_string(&self, content: &str) -> Result<Problem> {
        parsing::tptp_parser::parse_string(content)
    }
    
    /// Quick prescan to estimate complexity
    pub fn prescan_file(&self, file_path: &Path) -> Result<(usize, bool)> {
        parsing::prescan::prescan_file(file_path, 3, &mut std::collections::HashSet::new())
            .map_err(|e| ParseError::ParseError(format!("Prescan error: {:?}", e)))
    }
}

impl Default for TPTPFormat {
    fn default() -> Self {
        Self::new()
    }
}