#!/usr/bin/env python3
"""
Custom problem creation example.

This example shows how to create theorem proving problems
programmatically using the ProofAtlas API.
"""

from proofatlas.core import (
    Variable, Constant, Function, Predicate,
    Term, Literal, Clause, Problem
)


def create_propositional_problem():
    """Create a simple propositional logic problem."""
    print("=== Propositional Logic Problem ===\n")
    
    # Create propositional predicates (0-ary)
    P = Predicate("P", 0)
    Q = Predicate("Q", 0)
    R = Predicate("R", 0)
    
    # Create clauses for: (P ∨ Q), (¬P ∨ R), (¬Q ∨ R) ⊢ R
    clause1 = Clause(Literal(P(), True), Literal(Q(), True))      # P ∨ Q
    clause2 = Clause(Literal(P(), False), Literal(R(), True))     # ¬P ∨ R
    clause3 = Clause(Literal(Q(), False), Literal(R(), True))     # ¬Q ∨ R
    clause4 = Clause(Literal(R(), False))                         # ¬R (negated goal)
    
    problem = Problem(clause1, clause2, clause3, clause4)
    
    print("Problem clauses:")
    for i, clause in enumerate(problem.clauses):
        print(f"  {i}: {clause}")
    
    return problem


def create_first_order_problem():
    """Create a first-order logic problem with quantifiers."""
    print("\n=== First-Order Logic Problem ===\n")
    
    # Constants and variables
    a = Constant("a")
    b = Constant("b")
    X = Variable("X")
    Y = Variable("Y")
    
    # Predicates
    Human = Predicate("Human", 1)
    Mortal = Predicate("Mortal", 1)
    Parent = Predicate("Parent", 2)
    Ancestor = Predicate("Ancestor", 2)
    
    # Create clauses for a classic syllogism:
    # All humans are mortal: ∀X. Human(X) → Mortal(X)
    # Socrates is human: Human(socrates)
    # Therefore: Mortal(socrates)
    
    socrates = Constant("socrates")
    
    # ¬Human(X) ∨ Mortal(X)
    clause1 = Clause(
        Literal(Human(X), False),
        Literal(Mortal(X), True)
    )
    
    # Human(socrates)
    clause2 = Clause(Literal(Human(socrates), True))
    
    # ¬Mortal(socrates) (negated goal)
    clause3 = Clause(Literal(Mortal(socrates), False))
    
    problem = Problem(clause1, clause2, clause3)
    
    print("Syllogism problem:")
    for i, clause in enumerate(problem.clauses):
        print(f"  {i}: {clause}")
    
    # Another example with binary predicates
    print("\n\nParent-Ancestor problem:")
    
    # Parent(X,Y) ∧ Ancestor(Y,Z) → Ancestor(X,Z)
    # Represented as: ¬Parent(X,Y) ∨ ¬Ancestor(Y,Z) ∨ Ancestor(X,Z)
    Z = Variable("Z")
    transitivity = Clause(
        Literal(Parent(X, Y), False),
        Literal(Ancestor(Y, Z), False),
        Literal(Ancestor(X, Z), True)
    )
    
    # Base case: Parent(X,Y) → Ancestor(X,Y)
    # Represented as: ¬Parent(X,Y) ∨ Ancestor(X,Y)
    base_case = Clause(
        Literal(Parent(X, Y), False),
        Literal(Ancestor(X, Y), True)
    )
    
    # Facts
    parent_fact = Clause(Literal(Parent(a, b), True))
    
    # Query: Is 'a' an ancestor of 'b'?
    # Negated: ¬Ancestor(a,b)
    query = Clause(Literal(Ancestor(a, b), False))
    
    problem2 = Problem(transitivity, base_case, parent_fact, query)
    
    for i, clause in enumerate(problem2.clauses):
        print(f"  {i}: {clause}")
    
    return problem, problem2


def create_equality_problem():
    """Create a problem involving equality and functions."""
    print("\n=== Equality and Functions Problem ===\n")
    
    # Variables and constants
    X = Variable("X")
    Y = Variable("Y")
    a = Constant("a")
    b = Constant("b")
    
    # Function symbols
    f = Function("f", 1)
    g = Function("g", 2)
    
    # Equality predicate (using a regular predicate for now)
    Eq = Predicate("=", 2)
    P = Predicate("P", 1)
    
    # Create clauses demonstrating function terms
    
    # f(a) = b
    clause1 = Clause(Literal(Eq(f(a), b), True))
    
    # ∀X. f(X) = g(X, X)
    # Represented as: ¬Eq(f(X), g(X,X)) ∨ ⊥ (always true)
    # For this example: Eq(f(X), g(X,X))
    clause2 = Clause(Literal(Eq(f(X), g(X, X)), True))
    
    # P(f(a))
    clause3 = Clause(Literal(P(f(a)), True))
    
    # ¬P(b) (to derive contradiction)
    clause4 = Clause(Literal(P(b), False))
    
    problem = Problem(clause1, clause2, clause3, clause4)
    
    print("Equality problem:")
    for i, clause in enumerate(problem.clauses):
        print(f"  {i}: {clause}")
    
    return problem


def demonstrate_problem_analysis():
    """Show how to analyze problem structure."""
    print("\n=== Problem Analysis ===\n")
    
    # Create a complex problem
    X = Variable("X")
    Y = Variable("Y")
    a = Constant("a")
    b = Constant("b")
    f = Function("f", 2)
    P = Predicate("P", 1)
    Q = Predicate("Q", 2)
    
    clauses = [
        Clause(Literal(P(X), True), Literal(Q(X, f(X, a)), False)),
        Clause(Literal(P(a), True)),
        Clause(Literal(P(b), False), Literal(Q(b, f(a, b)), True)),
        Clause(Literal(Q(Y, f(a, Y)), False))
    ]
    
    problem = Problem(*clauses)
    
    print(f"Problem with {len(problem.clauses)} clauses")
    print(f"Symbols in problem:")
    print(f"  Constants: {[str(c) for c in problem.constants]}")
    print(f"  Variables: {[str(v) for v in problem.variables]}")
    print(f"  Functions: {[(str(f), f.arity) for f in problem.functions]}")
    print(f"  Predicates: {[(str(p), p.arity) for p in problem.predicates]}")
    
    print("\nClause analysis:")
    for i, clause in enumerate(problem.clauses):
        print(f"  Clause {i}: {clause}")
        print(f"    Literals: {len(clause.literals)}")
        print(f"    Variables: {[str(v) for v in clause.variables]}")
        print(f"    Positive: {[str(lit) for lit in clause.literals if lit.polarity]}")
        print(f"    Negative: {[str(lit) for lit in clause.literals if not lit.polarity]}")


def main():
    """Main example function."""
    # Create different types of problems
    prop_problem = create_propositional_problem()
    
    fo_problem1, fo_problem2 = create_first_order_problem()
    
    eq_problem = create_equality_problem()
    
    # Analyze problem structure
    demonstrate_problem_analysis()
    
    # Save a problem
    print("\n=== Saving Problem ===")
    from proofatlas.core import save_problem
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        save_problem(prop_problem, f.name)
        print(f"Saved problem to: {f.name}")
        
        # Load it back
        from proofatlas.core import load_problem
        loaded = load_problem(f.name)
        print(f"Loaded problem with {len(loaded.clauses)} clauses")
        
        # Clean up
        import os
        os.unlink(f.name)


if __name__ == "__main__":
    main()