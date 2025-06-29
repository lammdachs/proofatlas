//! Symbol table for string interning

use std::collections::HashMap;

/// Symbol table for efficient string storage and lookup
#[derive(Debug)]
pub struct SymbolTable {
    /// Symbol strings indexed by ID
    symbols: Vec<String>,
    /// Map from symbol string to ID
    symbol_to_id: HashMap<String, u32>,
}

impl SymbolTable {
    /// Create a new empty symbol table
    pub fn new() -> Self {
        SymbolTable {
            symbols: Vec::new(),
            symbol_to_id: HashMap::new(),
        }
    }
    
    /// Intern a symbol, returning its ID
    pub fn intern(&mut self, symbol: &str) -> u32 {
        if let Some(&id) = self.symbol_to_id.get(symbol) {
            id
        } else {
            let id = self.symbols.len() as u32;
            self.symbols.push(symbol.to_string());
            self.symbol_to_id.insert(symbol.to_string(), id);
            id
        }
    }
    
    /// Get a symbol by ID
    pub fn get(&self, id: u32) -> Option<&str> {
        self.symbols.get(id as usize).map(|s| s.as_str())
    }
    
    /// Get the ID of a symbol if it exists
    pub fn get_id(&self, symbol: &str) -> Option<u32> {
        self.symbol_to_id.get(symbol).copied()
    }
    
    /// Number of symbols in the table
    pub fn len(&self) -> usize {
        self.symbols.len()
    }
    
    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

#[cfg(test)]
#[path = "symbol_table_tests.rs"]
mod tests;