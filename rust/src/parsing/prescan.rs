//! Fast pre-scanning for TPTP files to estimate complexity

use std::fs;
use std::path::Path;
use std::collections::HashSet;
use std::io;

/// Quickly estimate the number of literals in a TPTP file without full parsing
pub fn prescan_file(
    file_path: &Path,
    max_depth: i32,
    visited: &mut HashSet<String>,
) -> io::Result<(usize, bool)> {
    // Avoid circular includes
    let canonical_path = file_path.canonicalize()?;
    let path_str = canonical_path.display().to_string();
    
    if visited.contains(&path_str) {
        return Err(io::Error::new(io::ErrorKind::Other, format!("Circular include: {}", path_str)));
    }
    
    if max_depth <= 0 {
        return Err(io::Error::new(io::ErrorKind::Other, "Depth limit exceeded"));
    }
    
    visited.insert(path_str);
    
    // Read file content
    let content = fs::read_to_string(file_path)?;
    
    let mut cnf_literals = 0;
    let mut fof_count = 0;
    let mut include_literals = 0;
    
    // Quick scan for CNF clauses
    for line in content.lines() {
        let trimmed = line.trim();
        
        // Skip comments
        if trimmed.starts_with('%') || trimmed.is_empty() {
            continue;
        }
        
        // Count CNF clauses
        if trimmed.starts_with("cnf(") {
            // Simple heuristic: count pipes (|) for disjunctions
            cnf_literals += line.matches('|').count() + 1;
        }
        
        // Count FOF formulas
        if trimmed.starts_with("fof(") {
            fof_count += 1;
        }
        
        // Handle includes
        if trimmed.starts_with("include(") {
            if let Some(include_path) = extract_include_path(trimmed) {
                let resolved_path = resolve_include_path(file_path, &include_path);
                if let Ok(path) = resolved_path {
                    if path.exists() {
                        if let Ok((inc_lits, _)) = prescan_file(&path, max_depth - 1, visited) {
                            include_literals += inc_lits;
                        }
                    }
                }
            }
        }
    }
    
    // Estimate total literals
    // FOF formulas typically generate multiple clauses when converted to CNF
    let estimated_total = cnf_literals + include_literals + (fof_count * 10);
    
    // Result is exact only if no FOF formulas and we scanned all includes
    let is_exact = fof_count == 0 && max_depth > 1;
    
    Ok((estimated_total, is_exact))
}

/// Extract include file path from include statement
fn extract_include_path(line: &str) -> Option<String> {
    // Match include('path') or include("path")
    let start_single = line.find("include('");
    let start_double = line.find("include(\"");
    
    let (start, quote_char) = if let Some(pos) = start_single {
        (pos + 9, '\'')
    } else if let Some(pos) = start_double {
        (pos + 9, '"')
    } else {
        return None;
    };
    
    let remaining = &line[start..];
    if let Some(end) = remaining.find(quote_char) {
        Some(remaining[..end].to_string())
    } else {
        None
    }
}

/// Resolve include path relative to current file
fn resolve_include_path(current_file: &Path, include_path: &str) -> io::Result<std::path::PathBuf> {
    let parent = current_file.parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid file path"))?;
    
    // Try relative to current file first
    let relative_path = parent.join(include_path);
    if relative_path.exists() {
        return Ok(relative_path);
    }
    
    // Try with TPTP_PATH environment variable
    if let Ok(tptp_path) = std::env::var("TPTP_PATH") {
        let tptp_resolved = Path::new(&tptp_path).join(include_path);
        if tptp_resolved.exists() {
            return Ok(tptp_resolved);
        }
    }
    
    Err(io::Error::new(io::ErrorKind::NotFound, format!("Include not found: {}", include_path)))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_include_path() {
        assert_eq!(
            extract_include_path("include('Axioms/SET003+0.ax')."),
            Some("Axioms/SET003+0.ax".to_string())
        );
        
        assert_eq!(
            extract_include_path("include(\"test.ax\")."),
            Some("test.ax".to_string())
        );
        
        assert_eq!(extract_include_path("not an include"), None);
    }
}