# Changelog

All notable changes to the ProofAtlas Rust implementation will be documented in this file.

## [Unreleased]

### Major Change - Box<[T]> Arrays for Zero-Copy (2024)

#### Changed
- **Replaced Vec<T> with Box<[T]>** throughout ArrayProblem for stable memory addresses
- All inference rules updated to use array indexing instead of push/insert operations
- Parser now returns ArrayProblem with pre-allocated capacity
- Added capacity tracking with CapacityError for safe overflow handling

#### Added
- **PyArrayProblem** wrapper with freeze mechanism for safe array access
- `parse_string_to_array()` and `parse_file_to_array()` with user-specified capacity
- Pre-allocation constructor: `ArrayProblem::with_capacity(max_nodes, max_clauses, max_edges)`
- Comprehensive capacity checking in all array operations

#### Removed
- Redundant zero-copy implementations (zero_copy_array.rs, zero_copy_bindings.rs)
- Debug files (superposition_debug.rs)
- Misplaced test files (test_basic_parser.rs, test_superposition.rs)

#### Technical Details
- Enums stored as primitive types (u8) for numpy compatibility
- CSR edge format with proper offset updates
- All copy functions return Result<(), CapacityError>
- Arrays exposed to Python as numpy arrays (currently copies, zero-copy planned)

### Bug Fixes and Improvements

#### Fixed
- **Variable Sharing**: Variables within clauses now properly share the same node (e.g., X in P(X,X))
- **Parser**: Implemented variable tracking with HashMap to ensure consistent variable nodes within clauses
- **Subsumption**: Fixed to use proper unification instead of just checking predicate symbols
- **Saturation Loop**: Fixed order of operations - given clause now added to processed list after generating inferences

#### Added
- **Comprehensive Failure Tests**: Added 26 tests covering all ways inference rules can fail:
  - Resolution failure scenarios (5 tests)
  - Factoring failure scenarios (5 tests)
  - Superposition failure scenarios (5 tests)
  - Equality resolution failure scenarios (4 tests)
  - Equality factoring failure scenarios (5 tests)
  - Edge cases and complex scenarios (2 tests)
- **Variable Sharing Tests**: Added 7 tests for proper variable scoping and unification

#### Improved
- **Code Quality**: Removed unused imports and marked unused functions with #[allow(dead_code)]
- **Test Coverage**: All 112 tests now pass cleanly without warnings
- **Documentation**: Documented that occurs check is not implemented for performance reasons

### Major Refactoring - Array-Based Representation

#### Changed
- **Complete architectural overhaul** from traditional AST representation to array-based CSR format
- Restructured codebase into three main modules: `core/`, `rules/`, and `saturation/`
- Removed traditional `core` module with AST-based types (Term, Literal, Clause, Problem)
- All formulas now use `ArrayProblem` with CSR (Compressed Sparse Row) representation

#### Added
- **Core Module**
  - `ArrayProblem` - Main data structure using CSR format
  - `SymbolTable` - String interning for efficient symbol management
  - `ArrayBuilder` - Builder pattern for constructing problems
  - Comprehensive test suite (27 tests) for core components
  
- **Rules Module** 
  - Modular inference rule implementations
  - `InferenceResult` common type for rule results
  - Individual files for each rule: resolution, factoring, superposition, etc.
  
- **Saturation Module**
  - Given-clause saturation loop implementation
  - Literal selection strategies (5 different strategies)
  - Array-based unification algorithm
  - Subsumption checking (linear scan, discrimination trees planned)

#### Improved
- **Performance**: CSR format provides better cache locality
- **Memory usage**: More compact representation
- **Modularity**: Clear separation between core/rules/saturation
- **Testing**: Comprehensive unit tests for all modules

#### Removed
- Traditional AST types (Term, Literal, Clause, Problem)
- Old `algorithms` module structure
- Redundant substitution tracking in InferenceResult

### Infrastructure

#### Added
- Comprehensive documentation in README files
- Module-level documentation with examples
- Test organization documentation

#### Changed
- Updated Python bindings to work with array representation
- Parser now converts directly to array format

## [0.1.0] - Initial Release

- Initial TPTP parser implementation
- Basic Python bindings
- Traditional AST-based representation (now removed)