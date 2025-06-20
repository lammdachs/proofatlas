"""TPTP file format handler."""

from pathlib import Path
from typing import List, Optional

from proofatlas.core.logic import Clause, Problem
from .tptp_parser.parser import read_file, read_string
from .base import FileFormat


class TPTPFormat(FileFormat):
    """Handler for TPTP file format."""
    
    def parse_file(self, file_path: Path, max_size: Optional[int] = None) -> Problem:
        """Parse a TPTP file and return a Problem object."""
        include_path = str(file_path.parent)
        return read_file(str(file_path), include_path=include_path, max_size=max_size)
    
    def parse_string(self, content: str) -> Problem:
        """Parse a TPTP string and return a Problem object."""
        return read_string(content)
    
    def to_cnf(self, problem: Problem) -> List[Clause]:
        """Convert a Problem to CNF clauses."""
        return problem.clauses
    
    def write_cnf(self, clauses: List[Clause], output_path: Path) -> None:
        """Write CNF clauses to a file in TPTP format."""
        with open(output_path, 'w') as f:
            for i, clause in enumerate(clauses):
                f.write(f"cnf(clause_{i}, plain, {self._clause_to_tptp(clause)}).\n")
    
    def _clause_to_tptp(self, clause: Clause) -> str:
        """Convert a clause to TPTP format string."""
        if not clause.literals:
            return "$false"
        
        literals = []
        for lit in clause.literals:
            lit_str = self._literal_to_tptp(lit)
            literals.append(lit_str)
        
        if len(literals) == 1:
            return literals[0]
        return f"({' | '.join(literals)})"
    
    def _literal_to_tptp(self, literal) -> str:
        """Convert a literal to TPTP format string."""
        if literal.negated:
            return f"~{self._atom_to_tptp(literal.atom)}"
        return self._atom_to_tptp(literal.atom)
    
    def _atom_to_tptp(self, atom) -> str:
        """Convert an atom to TPTP format string."""
        if atom.predicate.name == 'eq':
            return f"{self._term_to_tptp(atom.args[0])} = {self._term_to_tptp(atom.args[1])}"
        
        if atom.predicate.arity == 0:
            return atom.predicate.name
        
        args = ', '.join(self._term_to_tptp(arg) for arg in atom.args)
        return f"{atom.predicate.name}({args})"
    
    def _term_to_tptp(self, term) -> str:
        """Convert a term to TPTP format string."""
        if hasattr(term, 'name') and hasattr(term, 'arity'):
            if term.arity == 0:
                return term.name
            return term.name
        
        if hasattr(term, 'symbol') and hasattr(term, 'args'):
            if term.symbol.arity == 0:
                return term.symbol.name
            args = ', '.join(self._term_to_tptp(arg) for arg in term.args)
            return f"{term.symbol.name}({args})"
        
        return str(term)
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['.p', '.tptp', '.ax']