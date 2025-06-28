"""JSON serialization for core objects."""

import json
from pathlib import Path
from typing import Dict, Any, Union

from .logic import (
    Variable, Constant, Function, Predicate,
    Term, Literal, Clause, Problem
)


class CoreJSONEncoder(json.JSONEncoder):
    """JSON encoder for core theorem proving objects."""
    
    def default(self, obj):
        # Variables
        if isinstance(obj, Variable):
            return {
                "_type": "Variable",
                "name": obj.name
            }
        
        # Constants and Functions
        elif isinstance(obj, Constant):
            return {
                "_type": "Constant", 
                "name": obj.name
            }
        
        elif isinstance(obj, Function):
            return {
                "_type": "Function",
                "name": obj.name,
                "arity": obj.arity
            }
        
        # Predicates
        elif isinstance(obj, Predicate):
            return {
                "_type": "Predicate",
                "name": obj.name,
                "arity": obj.arity
            }
        
        # Terms (compound)
        elif isinstance(obj, Term) and hasattr(obj, 'symbol') and hasattr(obj, 'args'):
            return {
                "_type": "Term",
                "symbol": obj.symbol,
                "args": obj.args
            }
        
        # Literals
        elif isinstance(obj, Literal):
            return {
                "_type": "Literal",
                "predicate": obj.predicate,
                "polarity": obj.polarity
            }
        
        # Clauses
        elif isinstance(obj, Clause):
            return {
                "_type": "Clause",
                "literals": list(obj.literals)
            }
        
        # Problems
        elif isinstance(obj, Problem):
            return {
                "_type": "Problem",
                "clauses": list(obj.clauses)
            }
        
        return super().default(obj)


def decode_core_object(dct: Dict[str, Any]) -> Any:
    """Decode a JSON dictionary back to core objects."""
    if "_type" not in dct:
        return dct
    
    obj_type = dct["_type"]
    
    # Variables
    if obj_type == "Variable":
        return Variable(dct["name"])
    
    # Constants
    elif obj_type == "Constant":
        return Constant(dct["name"])
    
    # Functions
    elif obj_type == "Function":
        return Function(dct["name"], dct["arity"])
    
    # Predicates
    elif obj_type == "Predicate":
        return Predicate(dct["name"], dct["arity"])
    
    # Terms
    elif obj_type == "Term":
        symbol = dct["symbol"]
        args = dct["args"]
        # Apply the symbol to arguments
        if hasattr(symbol, '__call__'):
            return symbol(*args)
        return symbol
    
    # Literals
    elif obj_type == "Literal":
        return Literal(dct["predicate"], dct["polarity"])
    
    # Clauses
    elif obj_type == "Clause":
        return Clause(*dct["literals"])
    
    # Problems
    elif obj_type == "Problem":
        return Problem(*dct["clauses"])
    
    return dct


# Convenience functions

def problem_to_json(problem: Problem, indent: int = 2) -> str:
    """Convert a Problem to JSON string."""
    return json.dumps(problem, cls=CoreJSONEncoder, indent=indent)


def problem_from_json(json_str: str) -> Problem:
    """Create a Problem from JSON string."""
    return json.loads(json_str, object_hook=decode_core_object)


def save_problem(problem: Problem, file_path: Union[str, Path]) -> None:
    """Save a Problem to a JSON file."""
    file_path = Path(file_path)
    with open(file_path, 'w') as f:
        json.dump(problem, f, cls=CoreJSONEncoder, indent=2)


def load_problem(file_path: Union[str, Path]) -> Problem:
    """Load a Problem from a JSON file."""
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        return json.load(f, object_hook=decode_core_object)