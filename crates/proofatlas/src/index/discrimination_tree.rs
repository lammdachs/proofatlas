//! Discrimination tree index for efficient demodulation candidate retrieval.
//!
//! A discrimination tree indexes rewrite rule LHS terms in a trie keyed by
//! preorder term traversal. Given a query subterm, it returns only structurally
//! compatible rewrite rules in O(|term|) time instead of scanning all unit
//! equalities O(k).
//!
//! Both sides of each unit equality are inserted (orientation is checked later
//! by `demodulate()` via KBO). Deletion is lazy via an `active` set.

use crate::index::{Index, IndexKind};
use crate::logic::{Clause, ConstantId, FunctionId, Interner, PredicateId, Term};
use std::any::Any;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Flat key for preorder term representation
// =============================================================================

/// Key type for a single node in the flattened preorder traversal of a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FlatKey {
    /// Function symbol with arity (arity needed for skip counts during Star traversal)
    Func(FunctionId, u8),
    /// Constant symbol
    Const(ConstantId),
    /// Wildcard: any variable in the indexed term (pattern side)
    Star,
}

// =============================================================================
// Trie node
// =============================================================================

#[derive(Debug, Default)]
struct DiscTreeNode {
    /// Children keyed by flat key
    children: HashMap<FlatKey, DiscTreeNode>,
    /// Clause indices stored at this node (leaf entries)
    entries: Vec<usize>,
}

// =============================================================================
// DiscriminationTree
// =============================================================================

/// Discrimination tree for indexing rewrite rule LHS terms.
///
/// Supports insert (on transfer of unit equalities) and retrieve (given a
/// query subterm, returns candidate clause indices). Deletion is lazy: an
/// `active` set tracks which clause indices are still valid.
#[derive(Debug)]
pub struct DiscriminationTree {
    root: DiscTreeNode,
    /// Predicate ID for equality (None if "=" not interned)
    equality_pred_id: Option<PredicateId>,
    /// Active clause indices (lazy deletion: only return these)
    active: HashSet<usize>,
}

impl DiscriminationTree {
    pub fn new(interner: &Interner) -> Self {
        DiscriminationTree {
            root: DiscTreeNode::default(),
            equality_pred_id: interner.get_predicate("="),
            active: HashSet::new(),
        }
    }

    /// Check if a clause is a unit positive equality.
    fn is_unit_equality(&self, clause: &Clause) -> bool {
        if clause.literals.len() != 1 || !clause.literals[0].polarity {
            return false;
        }
        if let Some(eq_pred_id) = self.equality_pred_id {
            clause.literals[0].predicate.id == eq_pred_id
                && clause.literals[0].predicate.arity == 2
        } else {
            false
        }
    }

    /// Insert a term into the trie, associating it with the given clause index.
    fn insert(&mut self, clause_idx: usize, term: &Term) {
        let mut keys = Vec::new();
        Self::flatten_insert(term, &mut keys);

        let mut node = &mut self.root;
        for key in keys {
            node = node.children.entry(key).or_default();
        }
        node.entries.push(clause_idx);
    }

    /// Flatten a term for insertion: preorder traversal, variables become Star.
    fn flatten_insert(term: &Term, keys: &mut Vec<FlatKey>) {
        match term {
            Term::Variable(_) => keys.push(FlatKey::Star),
            Term::Constant(c) => keys.push(FlatKey::Const(c.id)),
            Term::Function(f, args) => {
                keys.push(FlatKey::Func(f.id, f.arity));
                for arg in args {
                    Self::flatten_insert(arg, keys);
                }
            }
        }
    }

    /// Retrieve candidate clause indices for a query term.
    ///
    /// Traverses the trie following exact key matches and Star branches.
    /// When a Star is encountered in the trie, the corresponding subterm in
    /// the query is skipped (using `subterm_size` to count how many keys to
    /// skip).
    fn retrieve(&self, term: &Term) -> Vec<usize> {
        let mut query_keys = Vec::new();
        Self::flatten_query(term, &mut query_keys);

        let mut results = Vec::new();
        self.retrieve_recursive(&self.root, &query_keys, 0, &mut results);
        results
    }

    /// Flatten a term for querying: preorder traversal, variables become
    /// a query-side representation (we just use the same flattening, but
    /// variables in the query can match Star in the trie).
    fn flatten_query(term: &Term, keys: &mut Vec<FlatKey>) {
        // For the query side, we flatten identically. Variables in the query
        // can match any path in the trie (but for demodulation, the query is
        // a ground or partially-ground subterm of the clause being simplified,
        // and the pattern in the trie has Stars for variables). We handle
        // query-side variables by letting them match Star nodes.
        Self::flatten_insert(term, keys);
    }

    /// Recursive retrieval: traverse the trie matching against query keys.
    fn retrieve_recursive(
        &self,
        node: &DiscTreeNode,
        query_keys: &[FlatKey],
        pos: usize,
        results: &mut Vec<usize>,
    ) {
        if pos == query_keys.len() {
            // Reached end of query — collect active entries
            for &idx in &node.entries {
                if self.active.contains(&idx) {
                    results.push(idx);
                }
            }
            return;
        }

        let key = query_keys[pos];

        // Branch 1: exact match on the query key
        if let Some(child) = node.children.get(&key) {
            self.retrieve_recursive(child, query_keys, pos + 1, results);
        }

        // Branch 2: Star branch in the trie (pattern variable matches any query subterm).
        // For one-way matching, only pattern variables can be substituted.
        // A query variable can only be matched by a pattern Star (covered here
        // since Star-in-trie matches ANY query key including query Stars).
        // Query variables must NOT match concrete trie nodes — that would be
        // unification, not matching.
        if key != FlatKey::Star {
            if let Some(star_child) = node.children.get(&FlatKey::Star) {
                let skip = Self::subterm_size(query_keys, pos);
                self.retrieve_recursive(star_child, query_keys, pos + skip, results);
            }
        }
    }

    /// Compute the number of flat keys a subterm occupies starting at `pos`.
    fn subterm_size(keys: &[FlatKey], pos: usize) -> usize {
        if pos >= keys.len() {
            return 0;
        }
        match keys[pos] {
            FlatKey::Star | FlatKey::Const(_) => 1,
            FlatKey::Func(_, arity) => {
                let mut size = 1;
                for _ in 0..arity {
                    size += Self::subterm_size(keys, pos + size);
                }
                size
            }
        }
    }

    /// Retrieve all candidate clause indices for demodulating a clause.
    ///
    /// For every subterm in the clause, queries the trie and returns the
    /// deduplicated set of candidate rule indices.
    pub fn retrieve_clause_candidates(&self, clause: &Clause) -> HashSet<usize> {
        let mut candidates = HashSet::new();
        for lit in &clause.literals {
            for arg in &lit.args {
                self.collect_subterm_candidates(arg, &mut candidates);
            }
        }
        candidates
    }

    /// Recursively query the trie for every subterm.
    fn collect_subterm_candidates(&self, term: &Term, candidates: &mut HashSet<usize>) {
        // Query this term
        let results = self.retrieve(term);
        candidates.extend(results);

        // Recurse into subterms
        if let Term::Function(_, args) = term {
            for arg in args {
                self.collect_subterm_candidates(arg, candidates);
            }
        }
    }
}

// =============================================================================
// Index trait implementation
// =============================================================================

impl Index for DiscriminationTree {
    fn kind(&self) -> IndexKind {
        IndexKind::DiscriminationTree
    }

    fn on_add(&mut self, _idx: usize, _clause: &Clause) {
        // No-op: unit equalities are only tracked after transfer
    }

    fn on_transfer(&mut self, idx: usize, clause: &Clause) {
        if !self.is_unit_equality(clause) {
            return;
        }

        // Insert both sides of the equality into the trie
        let args = &clause.literals[0].args;
        if args.len() == 2 {
            self.insert(idx, &args[0]);
            self.insert(idx, &args[1]);
            self.active.insert(idx);
        }
    }

    fn on_delete(&mut self, idx: usize, _clause: &Clause) {
        self.active.remove(&idx);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logic::{Constant, FunctionSymbol, Literal, PredicateSymbol, Variable};

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

        fn unit_eq(&mut self, lhs: Term, rhs: Term) -> Clause {
            let eq_id = self.interner.intern_predicate("=");
            Clause::new(vec![Literal::positive(
                PredicateSymbol::new(eq_id, 2),
                vec![lhs, rhs],
            )])
        }
    }

    #[test]
    fn test_insert_and_retrieve_constant() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let eq = ctx.unit_eq(a.clone(), b.clone());

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &eq);

        // Query for `a` should find clause 0
        let results = tree.retrieve(&a);
        assert!(results.contains(&0), "Should find clause via LHS");

        // Query for `b` should also find clause 0
        let results = tree.retrieve(&b);
        assert!(results.contains(&0), "Should find clause via RHS");

        // Query for a different constant should find nothing
        let c = ctx.const_("c");
        let results = tree.retrieve(&c);
        assert!(results.is_empty(), "Should not find unrelated constant");
    }

    #[test]
    fn test_insert_and_retrieve_function() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let fa = ctx.func("f", vec![a.clone()]);
        let eq = ctx.unit_eq(fa.clone(), b.clone());

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &eq);

        // Query for f(a) should match
        let results = tree.retrieve(&fa);
        assert!(results.contains(&0));

        // Query for f(b) should not match (different argument)
        let fb = ctx.func("f", vec![b.clone()]);
        let results = tree.retrieve(&fb);
        assert!(results.is_empty());
    }

    #[test]
    fn test_star_matches_any_subterm() {
        let mut ctx = TestCtx::new();
        let x = ctx.var("X");
        let b = ctx.const_("b");
        let fx = ctx.func("f", vec![x.clone()]);
        // Rule: f(X) = b
        let eq = ctx.unit_eq(fx.clone(), b.clone());

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &eq);

        // Query for f(a) should match (Star matches a)
        let a = ctx.const_("a");
        let fa = ctx.func("f", vec![a.clone()]);
        let results = tree.retrieve(&fa);
        assert!(results.contains(&0), "Star should match constant");

        // Query for f(g(a)) should match (Star matches g(a))
        let ga = ctx.func("g", vec![a.clone()]);
        let fga = ctx.func("f", vec![ga.clone()]);
        let results = tree.retrieve(&fga);
        assert!(results.contains(&0), "Star should match function term");
    }

    #[test]
    fn test_lazy_deletion() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let eq = ctx.unit_eq(a.clone(), b.clone());

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &eq);

        let results = tree.retrieve(&a);
        assert!(results.contains(&0));

        // Delete clause 0
        tree.on_delete(0, &eq);

        let results = tree.retrieve(&a);
        assert!(results.is_empty(), "Deleted clause should not appear");
    }

    #[test]
    fn test_multiple_rules() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let c = ctx.const_("c");
        let fa = ctx.func("f", vec![a.clone()]);
        let fb = ctx.func("f", vec![b.clone()]);

        // Rule 0: f(a) = b
        let eq0 = ctx.unit_eq(fa.clone(), b.clone());
        // Rule 1: f(b) = c
        let eq1 = ctx.unit_eq(fb.clone(), c.clone());

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &eq0);
        tree.on_transfer(1, &eq1);

        // Query f(a) should only find rule 0
        let results = tree.retrieve(&fa);
        assert!(results.contains(&0));
        assert!(!results.contains(&1));

        // Query f(b) should only find rule 1
        let results = tree.retrieve(&fb);
        assert!(!results.contains(&0));
        assert!(results.contains(&1));
    }

    #[test]
    fn test_clause_candidates() {
        let mut ctx = TestCtx::new();
        let a = ctx.const_("a");
        let b = ctx.const_("b");
        let fa = ctx.func("f", vec![a.clone()]);

        // Rule: f(a) = b
        let eq = ctx.unit_eq(fa.clone(), b.clone());

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &eq);

        // Target clause: P(f(a), b) — should find rule 0 via subterm f(a) and via b
        let p_id = ctx.interner.intern_predicate("P");
        let target = Clause::new(vec![Literal::positive(
            PredicateSymbol::new(p_id, 2),
            vec![fa.clone(), b.clone()],
        )]);

        let candidates = tree.retrieve_clause_candidates(&target);
        assert!(candidates.contains(&0));
    }

    #[test]
    fn test_nested_function() {
        let mut ctx = TestCtx::new();
        let x = ctx.var("X");
        let y = ctx.var("Y");
        let b = ctx.const_("b");
        let fxy = ctx.func("f", vec![x.clone(), y.clone()]);
        // Rule: f(X, Y) = b
        let eq = ctx.unit_eq(fxy.clone(), b.clone());

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &eq);

        // Query f(a, g(c)) should match (Stars match a and g(c))
        let a = ctx.const_("a");
        let c = ctx.const_("c");
        let gc = ctx.func("g", vec![c.clone()]);
        let query = ctx.func("f", vec![a.clone(), gc.clone()]);
        let results = tree.retrieve(&query);
        assert!(results.contains(&0), "Stars should match any subterms");
    }

    #[test]
    fn test_non_unit_equality_ignored() {
        let mut ctx = TestCtx::new();
        // Multi-literal clause (not a unit equality)
        let p_id = ctx.interner.intern_predicate("P");
        let a = ctx.const_("a");
        let clause = Clause::new(vec![
            Literal::positive(PredicateSymbol::new(p_id, 1), vec![a.clone()]),
            Literal::positive(PredicateSymbol::new(p_id, 1), vec![a.clone()]),
        ]);

        let mut tree = DiscriminationTree::new(&ctx.interner);
        tree.on_transfer(0, &clause);

        // Should not have been inserted
        assert!(tree.active.is_empty());
    }
}
