"""
ProofAtlas: A modular framework for automated theorem proving.

ProofAtlas provides a clean, extensible implementation of saturation-based
theorem proving with the given clause algorithm. It includes:

- First-order logic representations
- Inference rules (resolution, factoring, subsumption)
- Given clause saturation loops
- Clause selection strategies
- Proof tracking and visualization
- TPTP format support

Basic usage:
    >>> from proofatlas import *
    >>> # Create a simple problem
    >>> P = Predicate("P", 0)
    >>> clause1 = Clause(Literal(P(), True))
    >>> clause2 = Clause(Literal(P(), False))
    >>> problem = Problem(clause1, clause2)
    >>> 
    >>> # Run saturation
    >>> proof = prove(problem)
    >>> print(f"Proof found: {proof.final_state.contains_empty_clause}")
"""

__version__ = "0.1.0"

# Core logic structures
from proofatlas.core import (
    # Symbols
    Variable, Constant, Function, Predicate,
    # Terms and formulas
    Term, Literal, Clause, Problem,
    # Serialization
    save_problem, load_problem
)

# Unification
from proofatlas.core.unification import (
    Substitution, unify, unify_terms, rename_variables
)

# Proof structures
from proofatlas.proofs import (
    ProofState, Proof, ProofStep,
    save_proof, load_proof
)

# Inference rules
from proofatlas.rules import (
    Rule, RuleApplication,
    ResolutionRule, FactoringRule, SubsumptionRule
)

# Saturation loops
from proofatlas.loops import (
    Loop, BasicLoop, get_loop
)

# Clause selectors
from proofatlas.selectors import (
    Selector, RandomSelector, get_selector
)

# File formats
from proofatlas.fileformats import (
    get_format_handler
)

# Configuration
from proofatlas.utils.config import get_config


def prove(problem: Problem, 
          loop: str = "basic",
          selector: str = "fifo",
          max_steps: int = 1000,
          **kwargs) -> Proof:
    """
    Attempt to prove a problem using saturation.
    
    Args:
        problem: The problem to prove
        loop: Name of the loop to use (default: "basic")
        selector: Name of the selector to use (default: "fifo")
        max_steps: Maximum number of saturation steps
        **kwargs: Additional arguments for loop/selector
        
    Returns:
        Proof object containing the proof search trace
    """
    # Create initial state
    initial_state = ProofState(
        processed=[],
        unprocessed=list(problem.clauses)
    )
    
    # Create proof
    proof = Proof(initial_state)
    
    # Get loop and selector
    loop_instance = get_loop(loop, **kwargs)
    
    # TODO: Implement selector registry and FIFO selector
    # For now, use simple FIFO selection
    
    # Run saturation
    steps = 0
    while proof.final_state.unprocessed and steps < max_steps:
        # Select clause (FIFO for now)
        given_clause_idx = 0
        
        # Apply one step
        proof = loop_instance.step(proof, given_clause=given_clause_idx)
        
        # Check for proof
        if proof.final_state.contains_empty_clause:
            break
            
        steps += 1
    
    return proof


__all__ = [
    # Version
    "__version__",
    
    # Core logic
    "Variable", "Constant", "Function", "Predicate",
    "Term", "Literal", "Clause", "Problem",
    "save_problem", "load_problem",
    
    # Unification
    "Substitution", "unify", "unify_terms", "rename_variables",
    
    # Proofs
    "ProofState", "Proof", "ProofStep",
    "save_proof", "load_proof",
    
    # Rules
    "Rule", "RuleApplication",
    "ResolutionRule", "FactoringRule", "SubsumptionRule",
    
    # Loops
    "Loop", "BasicLoop", "get_loop",
    
    # Selectors
    "Selector", "RandomSelector", "get_selector",
    
    # File formats
    "get_format_handler",
    
    # Configuration
    "get_config",
    
    # High-level API
    "prove"
]