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
    
    def write_file(self, problem: Problem, file_path: Path, **kwargs) -> None:
        """Write a Problem to a file in TPTP format."""
        with open(file_path, 'w') as f:
            f.write(self.format_problem(problem, **kwargs))
    
    def format_problem(self, problem: Problem, **kwargs) -> str:
        """Format a Problem as a TPTP string."""
        lines = []
        for i, clause in enumerate(problem.clauses):
            lines.append(f"cnf(clause_{i}, plain, {self._clause_to_tptp(clause)}).")
        return '\n'.join(lines)
    
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
        if not literal.polarity:
            return f"~{self._atom_to_tptp(literal.predicate)}"
        return self._atom_to_tptp(literal.predicate)
    
    def _atom_to_tptp(self, atom) -> str:
        """Convert an atom (Term) to TPTP format string."""
        # Check if it's an equality predicate
        if hasattr(atom.symbol, 'name') and atom.symbol.name == '=':
            return f"{self._term_to_tptp(atom.args[0])} = {self._term_to_tptp(atom.args[1])}"
        
        # Propositional (0-ary predicate)
        if atom.symbol.arity == 0:
            return atom.symbol.name
        
        # Predicate with arguments
        args = ', '.join(self._term_to_tptp(arg) for arg in atom.args)
        return f"{atom.symbol.name}({args})"
    
    def _term_to_tptp(self, term) -> str:
        """Convert a term to TPTP format string."""
        # Check if it's a Variable or Constant (has name attribute directly)
        if hasattr(term, 'name') and not hasattr(term, 'args'):
            return term.name
        
        # It's a compound term with symbol and args
        if hasattr(term, 'symbol') and hasattr(term, 'args'):
            if term.symbol.arity == 0 or not term.args:
                return term.symbol.name
            args = ', '.join(self._term_to_tptp(arg) for arg in term.args)
            return f"{term.symbol.name}({args})"
        
        return str(term)
    
    @property
    def name(self) -> str:
        """Return the name of this file format."""
        return 'tptp'
    
    @property
    def extensions(self) -> List[str]:
        """Return list of file extensions this format handles."""
        return ['.p', '.tptp', '.ax']