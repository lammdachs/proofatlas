//! Tests for symbol table

#[cfg(test)]
mod tests {
    use super::super::*;
    
    #[test]
    fn test_symbol_table_new() {
        let table = SymbolTable::new();
        // New table should be empty
        assert_eq!(table.get(0), None);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }
    
    #[test]
    fn test_intern_and_get() {
        let mut table = SymbolTable::new();
        
        // Intern some symbols
        let id1 = table.intern("foo");
        let id2 = table.intern("bar");
        let id3 = table.intern("foo"); // Should return same ID
        
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        
        // Get them back
        assert_eq!(table.get(id1), Some("foo"));
        assert_eq!(table.get(id2), Some("bar"));
        assert_eq!(table.get(id3), Some("foo"));
        
        // Non-existent ID
        assert_eq!(table.get(999), None);
    }
    
    #[test]
    fn test_empty_string_special_case() {
        let mut table = SymbolTable::new();
        
        // Empty string should always be ID 0
        let empty_id = table.intern("");
        assert_eq!(empty_id, 0);
        assert_eq!(table.get(0), Some(""));
        
        // Interning empty string again should return 0
        let empty_id2 = table.intern("");
        assert_eq!(empty_id2, 0);
    }
    
    #[test]
    fn test_intern_many_symbols() {
        let mut table = SymbolTable::new();
        let mut ids = Vec::new();
        
        // Intern many symbols
        for i in 0..100 {
            let symbol = format!("symbol_{}", i);
            let id = table.intern(&symbol);
            ids.push((id, symbol));
        }
        
        // Verify all can be retrieved
        for (id, symbol) in ids {
            assert_eq!(table.get(id), Some(symbol.as_str()));
        }
    }
    
    #[test]
    fn test_intern_unicode() {
        let mut table = SymbolTable::new();
        
        let id1 = table.intern("α");
        let id2 = table.intern("β");
        let id3 = table.intern("∀");
        let id4 = table.intern("∃");
        
        assert_eq!(table.get(id1), Some("α"));
        assert_eq!(table.get(id2), Some("β"));
        assert_eq!(table.get(id3), Some("∀"));
        assert_eq!(table.get(id4), Some("∃"));
    }
    
    #[test]
    fn test_case_sensitivity() {
        let mut table = SymbolTable::new();
        
        let id1 = table.intern("Foo");
        let id2 = table.intern("foo");
        let id3 = table.intern("FOO");
        
        // Should be different IDs
        assert_ne!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(id2, id3);
        
        assert_eq!(table.get(id1), Some("Foo"));
        assert_eq!(table.get(id2), Some("foo"));
        assert_eq!(table.get(id3), Some("FOO"));
    }
    
    #[test]
    fn test_whitespace_preservation() {
        let mut table = SymbolTable::new();
        
        let id1 = table.intern("foo bar");
        let id2 = table.intern("foo  bar");
        let id3 = table.intern(" foo bar ");
        
        // Should be different IDs
        assert_ne!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(id2, id3);
        
        assert_eq!(table.get(id1), Some("foo bar"));
        assert_eq!(table.get(id2), Some("foo  bar"));
        assert_eq!(table.get(id3), Some(" foo bar "));
    }
}