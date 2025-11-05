"""
Core utilities for document conversion
"""

from .metadata import format_metadata_header, generate_metadata
from .output import create_page_separator, format_markdown_output, save_markdown_file
from .text_processor import (
    apply_bidi_algorithm,
    detect_language,
    get_ocr_lang_code,
    normalize_text_for_llm,
)

__all__ = [
    "normalize_text_for_llm",
    "detect_language",
    "get_ocr_lang_code",
    "apply_bidi_algorithm",
    "generate_metadata",
    "format_metadata_header",
    "format_markdown_output",
    "create_page_separator",
    "save_markdown_file",
]
