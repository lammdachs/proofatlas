//! Symbol interning for efficient memory usage and comparison
//!
//! This module provides interned symbol IDs that replace String-based symbol names.
//! Benefits:
//! - O(1) comparison and hashing (u32 vs String)
//! - Reduced memory usage (4 bytes vs 24+ bytes per symbol)
//! - Copy semantics (no heap allocation on clone)
//!
//! Each symbol type has its own ID type for type safety:
//! - `VariableId` for variables
//! - `ConstantId` for constants
//! - `FunctionId` for function symbols
//! - `PredicateId` for predicate symbols

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;

/// ID for an interned variable name
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VariableId(pub(crate) u32);

/// ID for an interned constant name
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantId(pub(crate) u32);

/// ID for an interned function symbol name
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionId(pub(crate) u32);

/// ID for an interned predicate symbol name
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PredicateId(pub(crate) u32);

impl VariableId {
    /// Get the raw ID value (for debugging/serialization)
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl ConstantId {
    /// Get the raw ID value (for debugging/serialization)
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Create a ConstantId from a raw u32 (for error handling)
    pub fn from_raw(id: u32) -> Self {
        ConstantId(id)
    }
}

impl FunctionId {
    /// Get the raw ID value (for debugging/serialization)
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl PredicateId {
    /// Get the raw ID value (for debugging/serialization)
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Internal string arena for a single symbol type
#[derive(Debug, Clone, Default)]
struct StringArena {
    /// Interned strings, indexed by ID
    strings: Vec<String>,
    /// Lookup table from string to ID
    lookup: HashMap<String, u32>,
}

impl StringArena {
    fn new() -> Self {
        StringArena {
            strings: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Intern a string, returning its ID (get-or-create)
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.lookup.get(name) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.strings.push(name.to_string());
        self.lookup.insert(name.to_string(), id);
        id
    }

    /// Resolve an ID to its string
    fn resolve(&self, id: u32) -> &str {
        &self.strings[id as usize]
    }

    /// Check if a string is already interned
    fn contains(&self, name: &str) -> bool {
        self.lookup.contains_key(name)
    }

    /// Get the ID for an already-interned string (returns None if not found)
    fn get(&self, name: &str) -> Option<u32> {
        self.lookup.get(name).copied()
    }

    /// Number of interned strings
    fn len(&self) -> usize {
        self.strings.len()
    }
}

/// Symbol interner for first-order logic
///
/// Stores all symbol names in separate arenas for variables, constants,
/// functions, and predicates. Pass through problem context rather than
/// using global state for WASM compatibility.
#[derive(Debug, Clone, Default)]
pub struct Interner {
    variables: StringArena,
    constants: StringArena,
    functions: StringArena,
    predicates: StringArena,
}

impl Interner {
    /// Create a new empty interner
    pub fn new() -> Self {
        Interner {
            variables: StringArena::new(),
            constants: StringArena::new(),
            functions: StringArena::new(),
            predicates: StringArena::new(),
        }
    }

    // === Variable interning ===

    /// Intern a variable name, returning its ID (get-or-create)
    pub fn intern_variable(&mut self, name: &str) -> VariableId {
        VariableId(self.variables.intern(name))
    }

    /// Resolve a variable ID to its name
    pub fn resolve_variable(&self, id: VariableId) -> &str {
        self.variables.resolve(id.0)
    }

    /// Check if a variable name is already interned
    pub fn contains_variable(&self, name: &str) -> bool {
        self.variables.contains(name)
    }

    /// Get the ID for an already-interned variable (returns None if not found)
    pub fn get_variable(&self, name: &str) -> Option<VariableId> {
        self.variables.get(name).map(VariableId)
    }

    /// Number of interned variables
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    // === Constant interning ===

    /// Intern a constant name, returning its ID (get-or-create)
    pub fn intern_constant(&mut self, name: &str) -> ConstantId {
        ConstantId(self.constants.intern(name))
    }

    /// Resolve a constant ID to its name
    pub fn resolve_constant(&self, id: ConstantId) -> &str {
        self.constants.resolve(id.0)
    }

    /// Check if a constant name is already interned
    pub fn contains_constant(&self, name: &str) -> bool {
        self.constants.contains(name)
    }

    /// Get the ID for an already-interned constant (returns None if not found)
    pub fn get_constant(&self, name: &str) -> Option<ConstantId> {
        self.constants.get(name).map(ConstantId)
    }

    /// Number of interned constants
    pub fn constant_count(&self) -> usize {
        self.constants.len()
    }

    // === Function interning ===

    /// Intern a function name, returning its ID (get-or-create)
    pub fn intern_function(&mut self, name: &str) -> FunctionId {
        FunctionId(self.functions.intern(name))
    }

    /// Resolve a function ID to its name
    pub fn resolve_function(&self, id: FunctionId) -> &str {
        self.functions.resolve(id.0)
    }

    /// Check if a function name is already interned
    pub fn contains_function(&self, name: &str) -> bool {
        self.functions.contains(name)
    }

    /// Get the ID for an already-interned function (returns None if not found)
    pub fn get_function(&self, name: &str) -> Option<FunctionId> {
        self.functions.get(name).map(FunctionId)
    }

    /// Number of interned functions
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    // === Predicate interning ===

    /// Intern a predicate name, returning its ID (get-or-create)
    pub fn intern_predicate(&mut self, name: &str) -> PredicateId {
        PredicateId(self.predicates.intern(name))
    }

    /// Resolve a predicate ID to its name
    pub fn resolve_predicate(&self, id: PredicateId) -> &str {
        self.predicates.resolve(id.0)
    }

    /// Check if a predicate name is already interned
    pub fn contains_predicate(&self, name: &str) -> bool {
        self.predicates.contains(name)
    }

    /// Get the ID for an already-interned predicate (returns None if not found)
    pub fn get_predicate(&self, name: &str) -> Option<PredicateId> {
        self.predicates.get(name).map(PredicateId)
    }

    /// Number of interned predicates
    pub fn predicate_count(&self) -> usize {
        self.predicates.len()
    }

    // === Statistics ===

    /// Total number of interned symbols
    pub fn total_symbols(&self) -> usize {
        self.variable_count() + self.constant_count() + self.function_count() + self.predicate_count()
    }
}

// === Display implementations for debugging ===

impl fmt::Display for VariableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V{}", self.0)
    }
}

impl fmt::Display for ConstantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}", self.0)
    }
}

impl fmt::Display for FunctionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F{}", self.0)
    }
}

impl fmt::Display for PredicateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.0)
    }
}

// === Serde implementations ===
// These serialize IDs as u32 for compact storage
// Full string resolution happens via WithInterner wrappers in json.rs

impl Serialize for VariableId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VariableId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u32::deserialize(deserializer).map(VariableId)
    }
}

impl Serialize for ConstantId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ConstantId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u32::deserialize(deserializer).map(ConstantId)
    }
}

impl Serialize for FunctionId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FunctionId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u32::deserialize(deserializer).map(FunctionId)
    }
}

impl Serialize for PredicateId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PredicateId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u32::deserialize(deserializer).map(PredicateId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_interning() {
        let mut interner = Interner::new();

        let x1 = interner.intern_variable("X");
        let x2 = interner.intern_variable("X");
        let y = interner.intern_variable("Y");

        // Same name should return same ID
        assert_eq!(x1, x2);

        // Different names should return different IDs
        assert_ne!(x1, y);

        // Resolution should work
        assert_eq!(interner.resolve_variable(x1), "X");
        assert_eq!(interner.resolve_variable(y), "Y");

        // Count should be 2
        assert_eq!(interner.variable_count(), 2);
    }

    #[test]
    fn test_constant_interning() {
        let mut interner = Interner::new();

        let a = interner.intern_constant("a");
        let b = interner.intern_constant("b");
        let a2 = interner.intern_constant("a");

        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(interner.resolve_constant(a), "a");
        assert_eq!(interner.constant_count(), 2);
    }

    #[test]
    fn test_function_interning() {
        let mut interner = Interner::new();

        let f = interner.intern_function("f");
        let g = interner.intern_function("g");
        let f2 = interner.intern_function("f");

        assert_eq!(f, f2);
        assert_ne!(f, g);
        assert_eq!(interner.resolve_function(f), "f");
        assert_eq!(interner.function_count(), 2);
    }

    #[test]
    fn test_predicate_interning() {
        let mut interner = Interner::new();

        let p = interner.intern_predicate("p");
        let q = interner.intern_predicate("q");
        let p2 = interner.intern_predicate("p");

        assert_eq!(p, p2);
        assert_ne!(p, q);
        assert_eq!(interner.resolve_predicate(p), "p");
        assert_eq!(interner.predicate_count(), 2);
    }

    #[test]
    fn test_separate_namespaces() {
        let mut interner = Interner::new();

        // Same name in different namespaces should have different IDs
        // (though the types prevent mixing them)
        let v = interner.intern_variable("x");
        let c = interner.intern_constant("x");
        let f = interner.intern_function("x");
        let p = interner.intern_predicate("x");

        // All should resolve to "x"
        assert_eq!(interner.resolve_variable(v), "x");
        assert_eq!(interner.resolve_constant(c), "x");
        assert_eq!(interner.resolve_function(f), "x");
        assert_eq!(interner.resolve_predicate(p), "x");

        // Each namespace has 1 entry
        assert_eq!(interner.variable_count(), 1);
        assert_eq!(interner.constant_count(), 1);
        assert_eq!(interner.function_count(), 1);
        assert_eq!(interner.predicate_count(), 1);
        assert_eq!(interner.total_symbols(), 4);
    }

    #[test]
    fn test_contains_and_get() {
        let mut interner = Interner::new();

        assert!(!interner.contains_variable("X"));
        assert!(interner.get_variable("X").is_none());

        let x = interner.intern_variable("X");

        assert!(interner.contains_variable("X"));
        assert_eq!(interner.get_variable("X"), Some(x));
        assert!(!interner.contains_variable("Y"));
    }

    #[test]
    fn test_id_copy_and_hash() {
        use std::collections::HashSet;

        let mut interner = Interner::new();
        let x = interner.intern_variable("X");
        let y = interner.intern_variable("Y");

        // Test Copy
        let x_copy = x;
        assert_eq!(x, x_copy);

        // Test Hash
        let mut set = HashSet::new();
        set.insert(x);
        set.insert(y);
        set.insert(x); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_id_ordering() {
        let mut interner = Interner::new();
        let x = interner.intern_variable("X");
        let y = interner.intern_variable("Y");

        // First interned should have lower ID
        assert!(x < y);
    }

    #[test]
    fn test_clone_interner() {
        let mut interner = Interner::new();
        let x = interner.intern_variable("X");

        let interner2 = interner.clone();
        assert_eq!(interner2.resolve_variable(x), "X");
        assert_eq!(interner2.variable_count(), 1);
    }
}
