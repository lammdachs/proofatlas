//! Index data structures for saturation-based theorem proving.
//!
//! Each rule owns its own index instance. Rules that need an index hold it
//! as a field and implement lifecycle methods (on_add, on_transfer, on_delete,
//! on_activate) to maintain it.
//!
//! ## Concrete Index Types
//!
//! - `SubsumptionChecker`: Literal discrimination tree for subsumption filtering
//! - `DiscriminationTree`: Trie index on rewrite rule LHS terms for demodulation
//! - `SelectedLiteralIndex`: Selected literal index for generating inference candidate filtering

pub mod disc_tree;
pub mod discrimination_tree;
pub mod selected_literals;
pub mod subsumption;

pub use discrimination_tree::DiscriminationTree;
pub use selected_literals::SelectedLiteralIndex;
pub use subsumption::SubsumptionChecker;
