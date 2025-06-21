"""Basic given clause loop implementation."""

from typing import List, Tuple

from proofatlas.proofs import Proof
from proofatlas.proofs.state import ProofState
from proofatlas.core.logic import Clause
from proofatlas.rules import ResolutionRule, FactoringRule, RuleApplication
from .base import Loop


class BasicLoop(Loop):
    """Basic implementation of the given clause algorithm."""
    
    def __init__(self,
                 max_clause_size: int = 100,
                 forward_simplify: bool = True,
                 backward_simplify: bool = True):
        """
        Initialize the basic loop.
        
        Args:
            max_clause_size: Maximum clause size to keep
            forward_simplify: Apply forward simplification
            backward_simplify: Apply backward simplification
        """
        self.max_clause_size = max_clause_size
        self.forward_simplify = forward_simplify
        self.backward_simplify = backward_simplify
    
    def step(self, proof: Proof, given_clause: int) -> Proof:
        """
        Execute one step of the given clause loop.
        
        Args:
            proof: Current proof
            given_clause: Index of the clause to process from unprocessed
            
        Returns:
            Updated proof with new state
        """
        current_state = proof.final_state
        
        # Validate given clause index
        if given_clause < 0 or given_clause >= len(current_state.unprocessed):
            raise ValueError(f"Invalid clause index: {given_clause}")
        
        # Get the selected clause
        selected = current_state.unprocessed[given_clause]
        
        # Create new state with clause moved to processed
        new_processed = list(current_state.processed) + [selected]
        new_unprocessed = [c for i, c in enumerate(current_state.unprocessed) if i != given_clause]
        
        # Generate new clauses and track rule applications
        new_clauses, rule_applications = self._generate_clauses_with_rules(selected, new_processed[:-1])
        
        # Apply simplification
        if self.forward_simplify:
            new_clauses = self._forward_simplify(new_clauses, new_processed, new_unprocessed)
        
        if self.backward_simplify:
            new_processed, new_unprocessed = self._backward_simplify(
                new_clauses, new_processed, new_unprocessed
            )
        
        # Add non-redundant new clauses
        kept_clauses = []
        for clause in new_clauses:
            if not self._is_redundant(clause, new_processed, new_unprocessed):
                new_unprocessed.append(clause)
                kept_clauses.append(clause)
        
        # Create new state
        new_state = ProofState(processed=new_processed, unprocessed=new_unprocessed)
        
        # Add step to proof with applied rules
        metadata = {
            'rule': 'given_clause',
            'selected_clause': selected,
            'new_clauses': kept_clauses,
            'num_generated': len(new_clauses),
            'num_kept': len(kept_clauses)
        }
        
        proof.add_step(new_state, selected_clause=given_clause, 
                      applied_rules=rule_applications, **metadata)
        
        return proof
    
    def _generate_clauses(self, given_clause: Clause, processed: List[Clause]) -> List[Clause]:
        """Generate new clauses from the given clause."""
        new_clauses, _ = self._generate_clauses_with_rules(given_clause, processed)
        return new_clauses
    
    def _generate_clauses_with_rules(self, given_clause: Clause, processed: List[Clause]) -> Tuple[List[Clause], List[RuleApplication]]:
        """Generate new clauses from the given clause and track rule applications."""
        new_clauses = []
        rule_applications = []
        
        # Create a temporary state with given_clause in processed for rule application
        temp_processed = processed + [given_clause]
        temp_state = ProofState(processed=temp_processed, unprocessed=[])
        
        # Apply resolution between given clause and each processed clause
        resolution = ResolutionRule()
        given_idx = len(processed)  # Index of given clause in temp_processed
        
        for i, proc_clause in enumerate(processed):
            result = resolution.apply(temp_state, [i, given_idx])
            if result:
                new_clauses.extend(result.generated_clauses)
                rule_applications.append(result)
        
        # Apply factoring to the given clause
        factoring = FactoringRule()
        result = factoring.apply(temp_state, [given_idx])
        if result:
            new_clauses.extend(result.generated_clauses)
            rule_applications.append(result)
        
        # Filter by size
        filtered_clauses = []
        filtered_applications = []
        clause_idx = 0
        
        for app in rule_applications:
            app_clauses = []
            for clause in app.generated_clauses:
                if len(clause.literals) <= self.max_clause_size:
                    filtered_clauses.append(clause)
                    app_clauses.append(clause)
            
            # Only keep the rule application if it generated at least one kept clause
            if app_clauses:
                # Create a new RuleApplication with only the kept clauses
                filtered_app = RuleApplication(
                    rule_name=app.rule_name,
                    parents=app.parents,
                    generated_clauses=app_clauses,
                    deleted_clause_indices=app.deleted_clause_indices,
                    metadata=app.metadata
                )
                filtered_applications.append(filtered_app)
        
        return filtered_clauses, filtered_applications
    
    def _forward_simplify(self, clauses: List[Clause], 
                         processed: List[Clause],
                         unprocessed: List[Clause]) -> List[Clause]:
        """Apply forward simplification."""
        simplified = []
        all_existing = processed + unprocessed
        
        for clause in clauses:
            # Skip tautologies
            if self.is_tautology(clause):
                continue
            
            # Skip if subsumed by existing clauses
            if self.is_subsumed(clause, all_existing):
                continue
            
            simplified.append(clause)
        
        return simplified
    
    def _backward_simplify(self, new_clauses: List[Clause],
                          processed: List[Clause], 
                          unprocessed: List[Clause]) -> tuple:
        """Apply backward simplification."""
        # TODO: Remove clauses subsumed by new clauses
        # For now, just return the original lists
        # processed = remove_subsumed(processed, new_clauses)
        # unprocessed = remove_subsumed(unprocessed, new_clauses)
        
        return processed, unprocessed
    
    def _is_redundant(self, clause: Clause, 
                     processed: List[Clause],
                     unprocessed: List[Clause]) -> bool:
        """Check if a clause is redundant."""
        # Check for exact duplicates
        all_clauses = processed + unprocessed
        for existing in all_clauses:
            if self._clauses_equal(clause, existing):
                return True
        return False
    
    def _clauses_equal(self, c1: Clause, c2: Clause) -> bool:
        """Check if two clauses are equal."""
        if len(c1.literals) != len(c2.literals):
            return False
        
        # Simple string comparison for now
        # A more sophisticated implementation would use proper equality
        return str(c1) == str(c2)