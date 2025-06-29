//! Comprehensive tests for symbol table

#[cfg(test)]
mod tests {
    use super::super::symbol_table::*;
    
    #[test]
    fn test_symbol_table_creation() {
        let table = SymbolTable::new();
        
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
        assert_eq!(table.get(0), None);
        assert_eq!(table.get_id("test"), None);
    }
    
    #[test]
    fn test_intern_single_symbol() {
        let mut table = SymbolTable::new();
        
        let id = table.intern("hello");
        assert_eq!(id, 0);
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
        
        // Should return same ID for same string
        let id2 = table.intern("hello");
        assert_eq!(id2, 0);
        assert_eq!(table.len(), 1); // No new symbol added
    }
    
    #[test]
    fn test_intern_multiple_symbols() {
        let mut table = SymbolTable::new();
        
        let id1 = table.intern("foo");
        let id2 = table.intern("bar");
        let id3 = table.intern("baz");
        
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(table.len(), 3);
        
        // Test retrieval
        assert_eq!(table.get(0), Some("foo"));
        assert_eq!(table.get(1), Some("bar"));
        assert_eq!(table.get(2), Some("baz"));
        assert_eq!(table.get(3), None);
    }
    
    #[test]
    fn test_get_id() {
        let mut table = SymbolTable::new();
        
        table.intern("alpha");
        table.intern("beta");
        table.intern("gamma");
        
        assert_eq!(table.get_id("alpha"), Some(0));
        assert_eq!(table.get_id("beta"), Some(1));
        assert_eq!(table.get_id("gamma"), Some(2));
        assert_eq!(table.get_id("delta"), None);
    }
    
    #[test]
    fn test_case_sensitivity() {
        let mut table = SymbolTable::new();
        
        let id1 = table.intern("Test");
        let id2 = table.intern("test");
        let id3 = table.intern("TEST");
        
        // Should be different symbols
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(table.len(), 3);
    }
    
    #[test]
    fn test_empty_string() {
        let mut table = SymbolTable::new();
        
        let id = table.intern("");
        assert_eq!(id, 0);
        assert_eq!(table.get(0), Some(""));
        assert_eq!(table.get_id(""), Some(0));
    }
    
    #[test]
    fn test_special_characters() {
        let mut table = SymbolTable::new();
        
        let symbols = vec![
            "hello world",
            "test-symbol",
            "symbol_with_underscore",
            "123numeric",
            "special!@#$%",
            "unicode_café",
            "emoji_🦀",
        ];
        
        let mut ids = Vec::new();
        for symbol in &symbols {
            ids.push(table.intern(symbol));
        }
        
        // All should have unique IDs
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(*id, i as u32);
        }
        
        // All should be retrievable
        for (i, symbol) in symbols.iter().enumerate() {
            assert_eq!(table.get(i as u32), Some(*symbol));
            assert_eq!(table.get_id(symbol), Some(i as u32));
        }
    }
    
    #[test]
    fn test_large_symbol_table() {
        let mut table = SymbolTable::new();
        
        // Add many symbols
        for i in 0..1000 {
            let symbol = format!("symbol_{}", i);
            let id = table.intern(&symbol);
            assert_eq!(id, i);
        }
        
        assert_eq!(table.len(), 1000);
        
        // Verify some random symbols
        assert_eq!(table.get(42), Some("symbol_42"));
        assert_eq!(table.get(999), Some("symbol_999"));
        assert_eq!(table.get_id("symbol_500"), Some(500));
    }
    
    #[test]
    fn test_symbol_ordering() {
        let mut table = SymbolTable::new();
        
        // Symbols should get IDs in order of first insertion
        let symbols = vec!["z", "a", "m", "b"];
        let expected_ids = vec![0, 1, 2, 3];
        
        for (symbol, expected_id) in symbols.iter().zip(expected_ids.iter()) {
            let id = table.intern(symbol);
            assert_eq!(id, *expected_id);
        }
        
        // Re-interning should preserve original IDs
        assert_eq!(table.intern("z"), 0);
        assert_eq!(table.intern("a"), 1);
    }
    
    #[test]
    fn test_symbol_table_consistency() {
        let mut table = SymbolTable::new();
        
        // Add symbols
        let symbols = vec!["P", "Q", "R", "f", "g", "a", "b", "X", "Y"];
        for symbol in &symbols {
            table.intern(symbol);
        }
        
        // Verify bidirectional mapping
        for (i, symbol) in symbols.iter().enumerate() {
            let id = i as u32;
            assert_eq!(table.get(id), Some(*symbol));
            assert_eq!(table.get_id(symbol), Some(id));
            
            // Re-interning should give same ID
            assert_eq!(table.intern(symbol), id);
        }
    }
    
    #[test]
    fn test_memory_efficiency() {
        let mut table = SymbolTable::new();
        
        // Intern same symbol many times
        for _ in 0..1000 {
            table.intern("repeated_symbol");
        }
        
        // Should only store one copy
        assert_eq!(table.len(), 1);
        
        // Add more unique symbols
        for i in 0..100 {
            table.intern(&format!("unique_{}", i));
        }
        
        assert_eq!(table.len(), 101);
    }
    
    #[test]
    fn test_typical_logic_symbols() {
        let mut table = SymbolTable::new();
        
        // Common symbols in theorem proving
        let logic_symbols = vec![
            "=",           // Equality
            "!=",          // Inequality  
            "true",        // Boolean constants
            "false",
            "forall",      // Quantifiers
            "exists",
            "and",         // Connectives
            "or",
            "not",
            "implies",
            "$i",          // TPTP types
            "$o",
            "$int",
            "$rat",
            "select",      // Array operations
            "store",
        ];
        
        for (i, symbol) in logic_symbols.iter().enumerate() {
            let id = table.intern(symbol);
            assert_eq!(id, i as u32);
        }
        
        // Common pattern: predicate and function names
        let pred_names = vec!["P", "Q", "R", "parent", "ancestor", "member"];
        let func_names = vec!["f", "g", "h", "plus", "times", "successor"];
        let const_names = vec!["a", "b", "c", "zero", "one"];
        let var_names = vec!["X", "Y", "Z", "X1", "X2", "Y1"];
        
        let base = logic_symbols.len() as u32;
        
        for (i, name) in pred_names.iter().enumerate() {
            assert_eq!(table.intern(name), base + i as u32);
        }
        
        // All symbols should be distinct
        assert_eq!(table.len(), logic_symbols.len() + pred_names.len());
    }
}