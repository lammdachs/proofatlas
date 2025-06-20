"""Token-based data format for proof states."""

from typing import List, Dict, Any
import torch

from proofatlas.core.fol.logic import Clause, Literal, Term
from proofatlas.models.transformer.tokenizer import ProofTokenizer, TokenConfig
from .base import DataFormat, ProofState


class TokenFormat(DataFormat):
    """Convert proof states to token sequences."""
    
    def __init__(self, token_config: TokenConfig = None, max_length: int = 512):
        self.max_length = max_length
        self.token_config = token_config or TokenConfig()
        self.tokenizer = ProofTokenizer(self.token_config, max_length)
    
    def encode_state(self, proof_state: ProofState) -> Dict[str, torch.Tensor]:
        """Encode a proof state into token sequences."""
        # Encode processed clauses with special marker
        processed_tokens = []
        for clause in proof_state.processed:
            tokens = self._clause_to_tokens(clause)
            processed_tokens.extend([self.tokenizer.config.processed_token] + tokens)
        
        # Encode unprocessed clauses with special marker
        unprocessed_tokens = []
        for clause in proof_state.unprocessed:
            tokens = self._clause_to_tokens(clause)
            unprocessed_tokens.extend([self.tokenizer.config.unprocessed_token] + tokens)
        
        # Combine and pad
        all_tokens = (
            [self.tokenizer.config.cls_token] +
            processed_tokens +
            [self.tokenizer.config.sep_token] +
            unprocessed_tokens +
            [self.tokenizer.config.eos_token]
        )
        
        # Truncate if necessary
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length-1] + [self.tokenizer.config.eos_token]
        
        # Pad to max length
        attention_mask = [1] * len(all_tokens)
        padding_length = self.max_length - len(all_tokens)
        all_tokens.extend([self.tokenizer.config.pad_token] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(all_tokens),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    def encode_clauses(self, clauses: List[Clause]) -> Dict[str, torch.Tensor]:
        """Encode a list of clauses into token sequences."""
        all_tokens = [self.tokenizer.config.cls_token]
        
        for i, clause in enumerate(clauses):
            if i > 0:
                all_tokens.append(self.tokenizer.config.sep_token)
            tokens = self._clause_to_tokens(clause)
            all_tokens.extend(tokens)
        
        all_tokens.append(self.tokenizer.config.eos_token)
        
        # Truncate if necessary
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length-1] + [self.tokenizer.config.eos_token]
        
        # Pad to max length
        attention_mask = [1] * len(all_tokens)
        padding_length = self.max_length - len(all_tokens)
        all_tokens.extend([self.tokenizer.config.pad_token] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(all_tokens),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    def encode_clause(self, clause: Clause) -> Dict[str, torch.Tensor]:
        """Encode a single clause into token sequence."""
        tokens = (
            [self.tokenizer.config.cls_token] +
            self._clause_to_tokens(clause) +
            [self.tokenizer.config.eos_token]
        )
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.tokenizer.config.eos_token]
        
        # Pad to max length
        attention_mask = [1] * len(tokens)
        padding_length = self.max_length - len(tokens)
        tokens.extend([self.tokenizer.config.pad_token] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(tokens),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    def _clause_to_tokens(self, clause: Clause) -> List[int]:
        """Convert a clause to a list of token IDs."""
        tokens = []
        
        for i, literal in enumerate(clause.literals):
            if i > 0:
                tokens.append(self.tokenizer.config.or_token)
            
            lit_tokens = self._literal_to_tokens(literal)
            tokens.extend(lit_tokens)
        
        return tokens
    
    def _literal_to_tokens(self, literal: Literal) -> List[int]:
        """Convert a literal to a list of token IDs."""
        tokens = []
        
        if literal.negated:
            tokens.append(self.tokenizer.config.not_token)
        
        # Add predicate token
        pred_token = self.tokenizer.get_symbol_token(
            literal.atom.predicate.name,
            literal.atom.predicate.arity
        )
        tokens.append(pred_token)
        
        # Add argument tokens
        if literal.atom.args:
            tokens.append(self.tokenizer.config.lparen_token)
            for i, arg in enumerate(literal.atom.args):
                if i > 0:
                    tokens.append(self.tokenizer.config.comma_token)
                arg_tokens = self._term_to_tokens(arg)
                tokens.extend(arg_tokens)
            tokens.append(self.tokenizer.config.rparen_token)
        
        return tokens
    
    def _term_to_tokens(self, term: Term) -> List[int]:
        """Convert a term to a list of token IDs."""
        if hasattr(term, 'symbol') and hasattr(term, 'args'):
            # Function term
            tokens = [self.tokenizer.get_symbol_token(term.symbol.name, term.symbol.arity)]
            
            if term.args:
                tokens.append(self.tokenizer.config.lparen_token)
                for i, arg in enumerate(term.args):
                    if i > 0:
                        tokens.append(self.tokenizer.config.comma_token)
                    arg_tokens = self._term_to_tokens(arg)
                    tokens.extend(arg_tokens)
                tokens.append(self.tokenizer.config.rparen_token)
            
            return tokens
        else:
            # Variable or constant
            if hasattr(term, 'name'):
                return [self.tokenizer.get_symbol_token(term.name, 0)]
            return [self.tokenizer.config.unk_token]
    
    @property
    def format_name(self) -> str:
        return "token"