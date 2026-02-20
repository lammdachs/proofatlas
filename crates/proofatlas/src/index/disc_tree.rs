//! Shared discrimination tree primitives.
//!
//! Provides the trie data structures and traversal algorithms used by both the
//! demodulation tree (`DiscriminationTree`) and the subsumption literal tree
//! (`LiteralDiscTree`). Terms are flattened into preorder key sequences; the
//! trie indexes these sequences and supports generalization and instance retrieval.

use crate::logic::{ConstantId, FunctionId, Term, VariableId};
use std::collections::HashMap;

// =============================================================================
// Flat key for preorder term representation
// =============================================================================

/// Key type for a single node in the flattened preorder traversal of a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlatKey {
    /// Function symbol with arity (arity needed for skip counts during Star traversal)
    Func(FunctionId, u8),
    /// Constant symbol
    Const(ConstantId),
    /// Wildcard: variable in the indexed term (matches anything during retrieval)
    Star,
    /// Concrete variable: variable in a query term (only Star matches it)
    Var(VariableId),
}

// =============================================================================
// Trie node
// =============================================================================

/// A node in a discrimination tree trie.
#[derive(Debug, Default)]
pub struct DiscTreeNode {
    /// Children keyed by flat key
    pub children: HashMap<FlatKey, DiscTreeNode>,
    /// Clause indices stored at this node (leaf entries)
    pub entries: Vec<usize>,
}

// =============================================================================
// Flattening functions
// =============================================================================

/// Flatten a term for insertion: preorder traversal, variables become Star.
pub fn flatten_insert(term: &Term, keys: &mut Vec<FlatKey>) {
    match term {
        Term::Variable(_) => keys.push(FlatKey::Star),
        Term::Constant(c) => keys.push(FlatKey::Const(c.id)),
        Term::Function(f, args) => {
            keys.push(FlatKey::Func(f.id, f.arity));
            for arg in args {
                flatten_insert(arg, keys);
            }
        }
    }
}

/// Flatten a term for generalization queries: variables become Var(id).
///
/// Used when the query contains variables that should NOT match concrete trie
/// symbols — only Star in the trie matches a Var in the query.
pub fn flatten_query_vars_concrete(term: &Term, keys: &mut Vec<FlatKey>) {
    match term {
        Term::Variable(v) => keys.push(FlatKey::Var(v.id)),
        Term::Constant(c) => keys.push(FlatKey::Const(c.id)),
        Term::Function(f, args) => {
            keys.push(FlatKey::Func(f.id, f.arity));
            for arg in args {
                flatten_query_vars_concrete(arg, keys);
            }
        }
    }
}

// =============================================================================
// Subterm size
// =============================================================================

/// Count the number of flat keys a subterm occupies starting at `pos`.
pub fn subterm_size(keys: &[FlatKey], pos: usize) -> usize {
    if pos >= keys.len() {
        return 0;
    }
    match keys[pos] {
        FlatKey::Star | FlatKey::Const(_) | FlatKey::Var(_) => 1,
        FlatKey::Func(_, arity) => {
            let mut size = 1;
            for _ in 0..arity {
                size += subterm_size(keys, pos + size);
            }
            size
        }
    }
}

// =============================================================================
// Trie insertion
// =============================================================================

/// Insert an entry into a trie node following the given key path.
pub fn trie_insert(node: &mut DiscTreeNode, keys: &[FlatKey], entry: usize) {
    let mut current = node;
    for &key in keys {
        current = current.children.entry(key).or_default();
    }
    current.entries.push(entry);
}

// =============================================================================
// Generalization retrieval
// =============================================================================

/// Find entries in the trie that are MORE GENERAL than the query.
///
/// A trie Star matches any query key (including Var). A query Var only matches
/// a trie Star (not concrete symbols). This implements one-way matching where
/// only trie-side variables can be substituted.
pub fn retrieve_generalizations(
    node: &DiscTreeNode,
    query: &[FlatKey],
    pos: usize,
    results: &mut Vec<usize>,
) {
    if pos == query.len() {
        results.extend_from_slice(&node.entries);
        return;
    }

    let key = query[pos];

    // Branch 1: exact match on the query key
    if let Some(child) = node.children.get(&key) {
        retrieve_generalizations(child, query, pos + 1, results);
    }

    // Branch 2: Star in the trie generalizes any non-Star query key.
    // For Var(X) in query: Var(X) != Star is true, so Star branch is followed
    // (correct — a trie variable generalizes any query variable).
    // For Star in query: Star == Star, so we skip this branch to avoid
    // double-counting (the exact match above already handles it).
    if key != FlatKey::Star {
        if let Some(star_child) = node.children.get(&FlatKey::Star) {
            let skip = subterm_size(query, pos);
            retrieve_generalizations(star_child, query, pos + skip, results);
        }
    }
}

// =============================================================================
// Instance retrieval
// =============================================================================

/// Find entries in the trie that are INSTANCES of the query.
///
/// A query Star matches any trie path (the query variable can be instantiated
/// to whatever the trie contains). A trie Star also matches any query key
/// (trivially — instantiate the trie variable to the query term). This is used
/// for backward subsumption: the subsumer's variables are Stars in the query,
/// and we want to find indexed clauses whose literals are instances.
pub fn retrieve_instances(
    node: &DiscTreeNode,
    query: &[FlatKey],
    pos: usize,
    results: &mut Vec<usize>,
) {
    if pos == query.len() {
        results.extend_from_slice(&node.entries);
        return;
    }

    match query[pos] {
        FlatKey::Star => {
            // Query wildcard: follow ALL trie children (query var matches anything)
            for (key, child) in &node.children {
                match key {
                    FlatKey::Const(_) | FlatKey::Star => {
                        retrieve_instances(child, query, pos + 1, results);
                    }
                    FlatKey::Func(_, arity) => {
                        skip_trie_args_and_continue(
                            child,
                            *arity as usize,
                            query,
                            pos + 1,
                            results,
                        );
                    }
                    FlatKey::Var(_) => {} // never in trie
                }
            }
        }
        key @ (FlatKey::Const(_) | FlatKey::Func(_, _)) => {
            // Exact match
            if let Some(child) = node.children.get(&key) {
                retrieve_instances(child, query, pos + 1, results);
            }
            // Star in trie (trie variable instantiated to this concrete symbol)
            if let Some(star_child) = node.children.get(&FlatKey::Star) {
                let skip = subterm_size(query, pos);
                retrieve_instances(star_child, query, pos + skip, results);
            }
        }
        FlatKey::Var(_) => {} // unreachable in instance queries (subsumer vars are Star)
    }
}

/// Helper for instance retrieval: when a query Star matches Func(f, n) in the
/// trie, traverse all trie paths forming a complete n-ary subterm, then resume
/// matching the query.
fn skip_trie_args_and_continue(
    node: &DiscTreeNode,
    remaining_args: usize,
    query: &[FlatKey],
    qpos: usize,
    results: &mut Vec<usize>,
) {
    if remaining_args == 0 {
        retrieve_instances(node, query, qpos, results);
        return;
    }
    for (key, child) in &node.children {
        match key {
            FlatKey::Const(_) | FlatKey::Star => {
                skip_trie_args_and_continue(child, remaining_args - 1, query, qpos, results);
            }
            FlatKey::Func(_, arity) => {
                skip_trie_args_and_continue(
                    child,
                    remaining_args - 1 + *arity as usize,
                    query,
                    qpos,
                    results,
                );
            }
            FlatKey::Var(_) => {} // never in trie
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, Interner, Variable};

    struct TestCtx {
        interner: Interner,
    }

    impl TestCtx {
        fn new() -> Self {
            TestCtx {
                interner: Interner::new(),
            }
        }

        fn var(&mut self, name: &str) -> Term {
            let id = self.interner.intern_variable(name);
            Term::Variable(Variable::new(id))
        }

        fn const_(&mut self, name: &str) -> Term {
            let id = self.interner.intern_constant(name);
            Term::Constant(Constant::new(id))
        }

        fn func(&mut self, name: &str, args: Vec<Term>) -> Term {
            let id = self.interner.intern_function(name);
            Term::Function(FunctionSymbol::new(id, args.len() as u8), args)
        }
    }

    #[test]
    fn test_flatten_insert_constant() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let mut keys = Vec::new();
        flatten_insert(&a, &mut keys);
        assert_eq!(keys.len(), 1);
        assert!(matches!(keys[0], FlatKey::Const(_)));
    }

    #[test]
    fn test_flatten_insert_variable_becomes_star() {
        let mut ctx = TestCtx::new();
        let x = ctx.var("X");
        let mut keys = Vec::new();
        flatten_insert(&x, &mut keys);
        assert_eq!(keys, vec![FlatKey::Star]);
    }

    #[test]
    fn test_flatten_insert_function() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let x = ctx.var("X");
        let fa_x = ctx.func("f", vec![a, x]);
        let mut keys = Vec::new();
        flatten_insert(&fa_x, &mut keys);
        assert_eq!(keys.len(), 3);
        assert!(matches!(keys[0], FlatKey::Func(_, 2)));
        assert!(matches!(keys[1], FlatKey::Const(_)));
        assert_eq!(keys[2], FlatKey::Star);
    }

    #[test]
    fn test_flatten_query_vars_concrete() {
        let mut ctx = TestCtx::new();
        let x = ctx.var("X");
        let a = ctx.const_("a");
        let fx = ctx.func("f", vec![x, a]);
        let mut keys = Vec::new();
        flatten_query_vars_concrete(&fx, &mut keys);
        assert_eq!(keys.len(), 3);
        assert!(matches!(keys[0], FlatKey::Func(_, 2)));
        assert!(matches!(keys[1], FlatKey::Var(_)));
        assert!(matches!(keys[2], FlatKey::Const(_)));
    }

    #[test]
    fn test_subterm_size_constant() {
        assert_eq!(subterm_size(&[FlatKey::Const(ConstantId(0))], 0), 1);
    }

    #[test]
    fn test_subterm_size_star() {
        assert_eq!(subterm_size(&[FlatKey::Star], 0), 1);
    }

    #[test]
    fn test_subterm_size_var() {
        assert_eq!(subterm_size(&[FlatKey::Var(VariableId(0))], 0), 1);
    }

    #[test]
    fn test_subterm_size_function() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let fab = ctx.func("f", vec![a, b]);
        let mut keys = Vec::new();
        flatten_insert(&fab, &mut keys);
        // f(a, b) -> [Func(f,2), Const(a), Const(b)] -> size 3
        assert_eq!(subterm_size(&keys, 0), 3);
    }

    #[test]
    fn test_subterm_size_nested() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let ga = ctx.func("g", vec![a]);
        let f_ga_b = ctx.func("f", vec![ga, b]);
        let mut keys = Vec::new();
        flatten_insert(&f_ga_b, &mut keys);
        // f(g(a), b) -> [Func(f,2), Func(g,1), Const(a), Const(b)] -> size 4
        assert_eq!(subterm_size(&keys, 0), 4);
        // g(a) subterm at pos 1 -> size 2
        assert_eq!(subterm_size(&keys, 1), 2);
    }

    #[test]
    fn test_trie_insert_and_retrieve_generalizations_ground() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let mut root = DiscTreeNode::default();

        let mut keys = Vec::new();
        flatten_insert(&a, &mut keys);
        trie_insert(&mut root, &keys, 0);

        // Query for same constant
        let mut results = Vec::new();
        retrieve_generalizations(&root, &keys, 0, &mut results);
        assert_eq!(results, vec![0]);

        // Query for different constant
        let b = ctx.const_("b");
        let mut keys_b = Vec::new();
        flatten_insert(&b, &mut keys_b);
        let mut results = Vec::new();
        retrieve_generalizations(&root, &keys_b, 0, &mut results);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_generalizations_star_matches_concrete() {
        let mut ctx = TestCtx::new();
        let x = ctx.var("X");
        let fx = ctx.func("f", vec![x]);

        let mut root = DiscTreeNode::default();
        let mut keys = Vec::new();
        flatten_insert(&fx, &mut keys);
        trie_insert(&mut root, &keys, 0);

        // Query f(a) — Star in trie should match constant a
        let a = ctx.const_("a");
        let fa = ctx.func("f", vec![a]);
        let mut query_keys = Vec::new();
        flatten_insert(&fa, &mut query_keys);
        let mut results = Vec::new();
        retrieve_generalizations(&root, &query_keys, 0, &mut results);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn test_retrieve_generalizations_var_only_matches_star() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let fa = ctx.func("f", vec![a]);

        let mut root = DiscTreeNode::default();
        let mut keys = Vec::new();
        flatten_insert(&fa, &mut keys);
        trie_insert(&mut root, &keys, 0);

        // Query f(X) with Var — Var should NOT match Const(a) in trie
        let x = ctx.var("X");
        let fx = ctx.func("f", vec![x]);
        let mut query_keys = Vec::new();
        flatten_query_vars_concrete(&fx, &mut query_keys);
        let mut results = Vec::new();
        retrieve_generalizations(&root, &query_keys, 0, &mut results);
        assert!(results.is_empty(), "Var in query should not match concrete trie key");

        // But if trie has f(Star), Var should match Star
        let y = ctx.var("Y");
        let fy = ctx.func("f", vec![y]);
        let mut star_keys = Vec::new();
        flatten_insert(&fy, &mut star_keys);
        trie_insert(&mut root, &star_keys, 1);

        let mut results = Vec::new();
        retrieve_generalizations(&root, &query_keys, 0, &mut results);
        assert_eq!(results, vec![1], "Var in query should match Star in trie");
    }

    #[test]
    fn test_retrieve_instances_ground() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let fa = ctx.func("f", vec![a]);

        let mut root = DiscTreeNode::default();
        let mut keys = Vec::new();
        flatten_insert(&fa, &mut keys);
        trie_insert(&mut root, &keys, 0);

        // Query f(a) with Star for variable — exact match is an instance
        let mut query_keys = Vec::new();
        flatten_insert(&fa, &mut query_keys);
        let mut results = Vec::new();
        retrieve_instances(&root, &query_keys, 0, &mut results);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn test_retrieve_instances_star_in_query_matches_all() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let fa = ctx.func("f", vec![a]);
        let fb = ctx.func("f", vec![b]);

        let mut root = DiscTreeNode::default();
        let mut keys_a = Vec::new();
        flatten_insert(&fa, &mut keys_a);
        trie_insert(&mut root, &keys_a, 0);

        let mut keys_b = Vec::new();
        flatten_insert(&fb, &mut keys_b);
        trie_insert(&mut root, &keys_b, 1);

        // Query f(Star) — should match both f(a) and f(b)
        let x = ctx.var("X");
        let fx = ctx.func("f", vec![x]);
        let mut query_keys = Vec::new();
        flatten_insert(&fx, &mut query_keys); // vars become Star
        let mut results = Vec::new();
        retrieve_instances(&root, &query_keys, 0, &mut results);
        results.sort();
        assert_eq!(results, vec![0, 1]);
    }

    #[test]
    fn test_retrieve_instances_star_in_trie_matches_query() {
        let mut ctx = TestCtx::new();
        // Trie contains f(Star) (from f(X))
        let x = ctx.var("X");
        let fx = ctx.func("f", vec![x]);
        let mut root = DiscTreeNode::default();
        let mut keys = Vec::new();
        flatten_insert(&fx, &mut keys);
        trie_insert(&mut root, &keys, 0);

        // Query f(a) — trie Star should match (trie var instantiated to a)
        let a = ctx.const_("a");
        let fa = ctx.func("f", vec![a]);
        let mut query_keys = Vec::new();
        flatten_insert(&fa, &mut query_keys);
        let mut results = Vec::new();
        retrieve_instances(&root, &query_keys, 0, &mut results);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn test_retrieve_instances_nested_function() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let ga = ctx.func("g", vec![a.clone()]);
        let f_ga_b = ctx.func("f", vec![ga, b]);

        let mut root = DiscTreeNode::default();
        let mut keys = Vec::new();
        flatten_insert(&f_ga_b, &mut keys);
        trie_insert(&mut root, &keys, 0);

        // Query f(Star, Star) — should match f(g(a), b)
        let x = ctx.var("X");
        let y = ctx.var("Y");
        let f_xy = ctx.func("f", vec![x, y]);
        let mut query_keys = Vec::new();
        flatten_insert(&f_xy, &mut query_keys); // [Func(f,2), Star, Star]
        let mut results = Vec::new();
        retrieve_instances(&root, &query_keys, 0, &mut results);
        assert_eq!(results, vec![0]);
    }

    #[test]
    fn test_skip_trie_args_nested() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let c = ctx.const_("c");
        let ga = ctx.func("g", vec![a]);
        let h_ga_b = ctx.func("h", vec![ga, b]);
        let f_h_c = ctx.func("f", vec![h_ga_b, c.clone()]);

        let mut root = DiscTreeNode::default();
        let mut keys = Vec::new();
        flatten_insert(&f_h_c, &mut keys);
        trie_insert(&mut root, &keys, 0);

        // Query f(Star, c) — Star should match entire h(g(a), b) subtree
        let x = ctx.var("X");
        let f_x_c = ctx.func("f", vec![x, c.clone()]);
        let mut query_keys = Vec::new();
        flatten_insert(&f_x_c, &mut query_keys);
        let mut results = Vec::new();
        retrieve_instances(&root, &query_keys, 0, &mut results);
        assert_eq!(results, vec![0]);
    }
}
