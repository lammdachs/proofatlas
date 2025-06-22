
"""Basic given clause loop implementation.

This module implements a complete saturation loop for theorem proving using
the given clause algorithm. The loop applies resolution and factoring rules,
performs redundancy elimination, and maintains a complete proof history.
"""

from typing import List, Tuple

from proofatlas.proofs import Proof
from proofatlas.proofs.proof import ProofStep
from proofatlas.proofs.state import ProofState
from proofatlas.core.logic import Clause
from proofatlas.rules import ResolutionRule, FactoringRule, RuleApplication
from .base import Loop


class BasicLoop(Loop):
    """Basic implementation of the given clause algorithm.
    
    This loop implements a single step of saturation-based theorem proving:
    1. Moves the given clause from unprocessed to processed
    2. Applies resolution between given clause and all processed clauses
    3. Applies factoring to the given clause
    4. Filters redundant clauses (tautologies, subsumed, duplicates)
    5. Adds non-redundant clauses to unprocessed
    
    The loop tracks all rule applications but only records those that
    produce non-redundant clauses in the final proof.
    """
    
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
        
        # First, try to factor the selected clause
        factoring = FactoringRule()
        temp_state = ProofState(processed=[selected], unprocessed=[])
        factor_result = factoring.apply(temp_state, [0])
        
        # Determine what to add to processed
        # If factoring succeeded and produced a simpler clause, use it instead
        clause_to_process = selected
        factoring_app = None
        
        if factor_result and factor_result.generated_clauses:
            # Check if any factor is simpler (fewer literals)
            for factor in factor_result.generated_clauses:
                if len(factor.literals) < len(selected.literals):
                    clause_to_process = factor
                    factoring_app = RuleApplication(
                        rule_name="factoring",
                        parents=[],  # No parents - applied to given clause
                        generated_clauses=[factor],
                        deleted_clause_indices=[],
                        metadata={'simplified_given_clause': True}
                    )
                    break
        
        # Create new state with clause moved to processed
        new_processed = list(current_state.processed) + [clause_to_process]
        new_unprocessed = [c for i, c in enumerate(current_state.unprocessed) if i != given_clause]
        
        # Generate new clauses via resolution
        # Pass the NEW processed list (which includes the clause we're adding)
        new_clauses, rule_applications = self._generate_resolution_clauses(clause_to_process, new_processed)
        
        # Add factoring application if we factored
        if factoring_app:
            rule_applications.insert(0, factoring_app)
        
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
        
        # Update rule applications to only include kept clauses
        final_rule_applications = []
        for app in rule_applications:
            # Special handling for factoring which was already added to processed
            if app.metadata.get('simplified_given_clause'):
                # Factoring already applied, just keep the record
                final_rule_applications.append(app)
            else:
                # Filter generated clauses to only those that were kept
                kept_from_app = [c for c in app.generated_clauses if c in kept_clauses]
                
                # Only include the rule application if it contributed kept clauses
                if kept_from_app:
                    final_app = RuleApplication(
                        rule_name=app.rule_name,
                        parents=app.parents,
                        generated_clauses=kept_from_app,
                        deleted_clause_indices=app.deleted_clause_indices,
                        metadata=app.metadata
                    )
                    final_rule_applications.append(final_app)
        
        # Create new state
        new_state = ProofState(processed=new_processed, unprocessed=new_unprocessed)
        
        # Add step with current state BEFORE processing, the selected clause,
        # and the rules that WILL BE applied
        proof.add_step(current_state, selected_clause=given_clause, 
                      applied_rules=final_rule_applications)
        
        # The Proof class will have added an extra step with the same state and no selection
        # We need to update that last step to have the NEW state
        proof.steps[-1] = ProofStep(new_state, selected_clause=None, applied_rules=[])
        
        return proof
    
    
    def _generate_clauses(self, given_clause: Clause, processed: List[Clause]) -> List[Clause]:
        """Generate new clauses from the given clause."""
        new_clauses, _ = self._generate_resolution_clauses(given_clause, processed)
        return new_clauses
    
    def _generate_resolution_clauses(self, given_clause: Clause, processed: List[Clause]) -> Tuple[List[Clause], List[RuleApplication]]:
        """Generate new clauses via resolution with previously processed clauses.
        
        Args:
            given_clause: The clause being processed (already at end of processed list)
            processed: List of ALL processed clauses (including given clause at the end)
        """
        new_clauses = []
        rule_applications = []
        
        # The state for rule application is just the processed list
        # (given clause is already at the end)
        temp_state = ProofState(processed=processed, unprocessed=[])
        
        # Apply resolution between given clause and each OTHER processed clause
        resolution = ResolutionRule()
        given_idx = len(processed) - 1  # Index of given clause (last in processed)
        
        # Only resolve with clauses BEFORE the given clause
        for i in range(len(processed) - 1):
            result = resolution.apply(temp_state, [i, given_idx])
            if result:
                new_clauses.extend(result.generated_clauses)
                # Modify the rule application to only show the processed clause index
                # The given clause is implicit (it's the one being processed)
                modified_result = RuleApplication(
                    rule_name=result.rule_name,
                    parents=[i],  # Only the index of the already-processed clause
                    generated_clauses=result.generated_clauses,
                    deleted_clause_indices=result.deleted_clause_indices,
                    metadata={**result.metadata, 'with_given_clause': True}
                )
                rule_applications.append(modified_result)
        
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
        """Apply forward simplification to remove redundant new clauses.
        
        Filters out:
        - Tautologies (clauses with complementary literals)
        - Clauses subsumed by existing clauses
        
        Args:
            clauses: New clauses to simplify
            processed: Currently processed clauses
            unprocessed: Currently unprocessed clauses
            
        Returns:
            List of clauses that passed all redundancy checks
        """
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
        """Apply backward simplification (NOT IMPLEMENTED).
        
        This would remove existing clauses that are subsumed by newly
        generated clauses. For example, if we generate P(a) and we
        already have P(a) ∨ Q(b), the latter would be removed.
        
        Currently returns the original lists unchanged.
        
        Args:
            new_clauses: Newly generated clauses
            processed: Currently processed clauses
            unprocessed: Currently unprocessed clauses
            
        Returns:
            Tuple of (processed, unprocessed) - currently unchanged
        """
        # TODO: Remove clauses subsumed by new clauses
        # For now, just return the original lists
        # processed = remove_subsumed(processed, new_clauses)
        # unprocessed = remove_subsumed(unprocessed, new_clauses)
        
        return processed, unprocessed
    
    def _is_redundant(self, clause: Clause, 
                     processed: List[Clause],
                     unprocessed: List[Clause]) -> bool:
        """Check if a clause is redundant (exact duplicate).
        
        This only checks for exact duplicates. Subsumption and tautology
        checking are done separately in _forward_simplify().
        
        Args:
            clause: Clause to check
            processed: Currently processed clauses
            unprocessed: Currently unprocessed clauses
            
        Returns:
            True if clause already exists in either set
        """
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