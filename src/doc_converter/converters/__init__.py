"""
Document converters for various file formats
"""

from .base import BaseConverter
from .pdf import PDFConverter

# Optional converters (may not be available if dependencies not installed)
try:
    from .docx import DOCXConverter

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    DOCXConverter = None

__all__ = [
    "BaseConverter",
    "PDFConverter",
    "DOCXConverter",
    "DOCX_AVAILABLE",
]


def get_converter_for_file(file_path: str, **kwargs) -> BaseConverter:
    """
    Get the appropriate converter for a given file

    Args:
        file_path: Path to file
        **kwargs: Arguments to pass to converter constructor

    Returns:
        Appropriate converter instance

    Raises:
        ValueError: If no converter supports the file format
    """
    import os

    _, ext = os.path.splitext(file_path.lower())

    # Try each converter
    if PDFConverter.supports_format(file_path):
        return PDFConverter(**kwargs)

    if DOCX_AVAILABLE and DOCXConverter.supports_format(file_path):
        return DOCXConverter(**kwargs)

    # No converter found
    raise ValueError(
        f"No converter available for file format: {ext}\n"
        f"Supported formats: .pdf" + (", .docx" if DOCX_AVAILABLE else "")
    )
