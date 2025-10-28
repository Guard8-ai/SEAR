"""
Document Converter - Universal document to markdown conversion library

Supports:
- PDF files (with OCR fallback)
- Microsoft Word (.docx) files
- Smart text extraction and normalization
- Multi-language support (Hebrew, English, mixed)
- RAG-optimized output with metadata
"""

__version__ = "1.0.0"

from .converters import (
    BaseConverter,
    PDFConverter,
    DOCXConverter,
    get_converter_for_file,
    DOCX_AVAILABLE
)

from .core import (
    normalize_text_for_llm,
    detect_language,
    apply_bidi_algorithm
)

__all__ = [
    # Converters
    'BaseConverter',
    'PDFConverter',
    'DOCXConverter',
    'get_converter_for_file',
    'convert_document',
    'DOCX_AVAILABLE',

    # Utilities
    'normalize_text_for_llm',
    'detect_language',
    'apply_bidi_algorithm',

    # Version
    '__version__',
]


def convert_document(
    file_path: str,
    output_dir: str = "converted_md",
    force_ocr: bool = False,
    lang: str = None,
    dpi: int = 300,
    apply_bidi: bool = True,
    normalize: bool = True,
    **kwargs
):
    """
    Convert a document to markdown (convenience function)

    This is the main public API for converting documents. It automatically
    detects the file format and uses the appropriate converter.

    Args:
        file_path: Path to document file
        output_dir: Output directory for markdown files (default: "converted_md")
        force_ocr: Force OCR even if text layer exists (PDF only, default: False)
        lang: Language override for OCR - 'heb', 'eng', or 'heb+eng' (PDF only)
        dpi: DPI for OCR image conversion (PDF only, default: 300)
        apply_bidi: Apply BiDi algorithm for RTL text (default: True)
        normalize: Normalize text to reduce token waste (default: True)
        **kwargs: Additional converter-specific parameters

    Returns:
        Tuple of (success, elapsed_time, metadata_dict)

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> # Convert a PDF
        >>> convert_document('document.pdf')
        (True, 2.5, {'pages': 10, 'language': 'english', ...})

        >>> # Convert a Word document
        >>> convert_document('report.docx', output_dir='output')
        (True, 0.8, {'pages': 5, 'language': 'hebrew', ...})

        >>> # Force OCR on a PDF
        >>> convert_document('scanned.pdf', force_ocr=True, lang='heb+eng')
        (True, 45.2, {'pages': 8, 'language': 'hebrew', ...})

        >>> # Import as a library
        >>> from doc_converter import convert_document
        >>> success, time, meta = convert_document('file.pdf')
        >>> if success:
        ...     print(f"Converted {meta['pages']} pages in {time:.2f}s")
    """
    # Build converter kwargs
    converter_kwargs = {
        'apply_bidi': apply_bidi,
        'normalize': normalize
    }

    # Add PDF-specific options if it's a PDF
    if file_path.lower().endswith('.pdf'):
        converter_kwargs.update({
            'force_ocr': force_ocr,
            'dpi': dpi,
            'lang': lang
        })

    # Add any additional kwargs
    converter_kwargs.update(kwargs)

    # Get appropriate converter and convert
    converter = get_converter_for_file(file_path, **converter_kwargs)
    return converter.convert(file_path, output_dir=output_dir)
