"""
File format handlers for converting various theorem proving file formats to TPTP CNF.
"""

from .base import FileFormat
from .tptp import TPTPFormat
from .registry import FileFormatRegistry, get_format_handler

__all__ = ['FileFormat', 'TPTPFormat', 'FileFormatRegistry', 'get_format_handler']