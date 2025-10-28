"""
Core utilities for document conversion
"""

from .text_processor import (
    normalize_text_for_llm,
    detect_language,
    get_ocr_lang_code,
    apply_bidi_algorithm
)

from .metadata import (
    generate_metadata,
    format_metadata_header
)

from .output import (
    format_markdown_output,
    create_page_separator,
    save_markdown_file
)

__all__ = [
    'normalize_text_for_llm',
    'detect_language',
    'get_ocr_lang_code',
    'apply_bidi_algorithm',
    'generate_metadata',
    'format_metadata_header',
    'format_markdown_output',
    'create_page_separator',
    'save_markdown_file',
]
