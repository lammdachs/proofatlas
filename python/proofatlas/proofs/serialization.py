"""JSON serialization for proof objects."""

import json
import os
from pathlib import Path
from typing import Union

from proofatlas.core.serialization import CoreJSONEncoder, decode_core_object
from proofatlas.rules import RuleApplication

# Import the appropriate implementations based on USE_RUST
USE_RUST = os.environ.get('PROOFATLAS_USE_RUST', 'false').lower() == 'true'

if USE_RUST:
    try:
        from .proof_rust import Proof, ProofStep
        from .state_rust import ProofState
    except ImportError:
        from .proof import Proof, ProofStep
        from .state import ProofState
else:
    from .proof import Proof, ProofStep
    from .state import ProofState


class ProofJSONEncoder(CoreJSONEncoder):
    """JSON encoder for proof objects."""
    
    def default(self, obj):
        # ProofState - check by class name to support both Python and Rust implementations
        if obj.__class__.__name__ == "ProofState":
            return {
                "_type": "ProofState",
                "processed": obj.processed,
                "unprocessed": obj.unprocessed
            }
        
        # RuleApplication
        elif isinstance(obj, RuleApplication):
            return {
                "_type": "RuleApplication",
                "rule_name": obj.rule_name,
                "parents": obj.parents,
                "generated_clauses": obj.generated_clauses,
                "deleted_clause_indices": obj.deleted_clause_indices,
                "metadata": obj.metadata
            }
        
        # ProofStep - check by class name
        elif obj.__class__.__name__ == "ProofStep":
            return {
                "_type": "ProofStep",
                "state": obj.state,
                "selected_clause": obj.selected_clause,
                "applied_rules": obj.applied_rules,
                "metadata": obj.metadata
            }
        
        # Proof - check by class name
        elif obj.__class__.__name__ == "Proof":
            return {
                "_type": "Proof",
                "steps": obj.steps
            }
        
        # Fall back to parent encoder
        return super().default(obj)


class ProofJSONDecoder(json.JSONDecoder):
    """JSON decoder for proof objects."""
    
    def __init__(self):
        super().__init__(object_hook=self.object_hook)
    
    def object_hook(self, obj):
        # First try core decoder
        result = decode_core_object(obj)
        if result is not obj:
            return result
        
        # ProofState
        if obj.get("_type") == "ProofState":
            return ProofState(
                processed=obj["processed"],
                unprocessed=obj["unprocessed"]
            )
        
        # RuleApplication
        elif obj.get("_type") == "RuleApplication":
            return RuleApplication(
                rule_name=obj["rule_name"],
                parents=obj["parents"],
                generated_clauses=obj.get("generated_clauses", []),
                deleted_clause_indices=obj.get("deleted_clause_indices", []),
                metadata=obj.get("metadata", {})
            )
        
        # ProofStep
        elif obj.get("_type") == "ProofStep":
            return ProofStep(
                state=obj["state"],
                selected_clause=obj.get("selected_clause"),
                applied_rules=obj.get("applied_rules", []),
                metadata=obj.get("metadata", {})
            )
        
        # Proof
        elif obj.get("_type") == "Proof":
            # Create proof from first step
            first_step = obj["steps"][0]
            proof = Proof(first_step.state)
            
            # If first step has selection, replace it
            if first_step.selected_clause is not None:
                proof.add_step(
                    first_step.state,
                    selected_clause=first_step.selected_clause,
                    applied_rules=first_step.applied_rules,
                    **first_step.metadata
                )
            
            # Add remaining steps
            for step in obj["steps"][1:]:
                if step.selected_clause is None:
                    # Final step
                    proof.finalize(step.state)
                else:
                    proof.add_step(
                        step.state,
                        selected_clause=step.selected_clause,
                        applied_rules=step.applied_rules,
                        **step.metadata
                    )
            return proof
        
        return obj


# Convenience functions
def proof_to_json(proof: Proof, indent: int = 2) -> str:
    """Convert a proof to JSON string."""
    return json.dumps(proof, cls=ProofJSONEncoder, indent=indent)


def proof_from_json(json_str: str) -> Proof:
    """Convert JSON string to a proof."""
    return json.loads(json_str, cls=ProofJSONDecoder)


def save_proof(proof: Proof, filepath: Union[str, Path]) -> None:
    """Save a proof to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(proof, f, cls=ProofJSONEncoder, indent=2)


def load_proof(filepath: Union[str, Path]) -> Proof:
    """Load a proof from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f, cls=ProofJSONDecoder)