#!/usr/bin/env python3
"""
Metadata Generation for Document Conversion
Handles creation and formatting of document metadata
"""

import os
from datetime import datetime
from typing import Any


def generate_metadata(
    source_file: str,
    pages: int,
    language: str,
    detected_languages: list,
    method: str,
    characters: int,
    **kwargs,
) -> dict[str, Any]:
    """
    Generate metadata dictionary for converted documents

    Args:
        source_file: Original file name
        pages: Number of pages/sections in document
        language: Primary detected language
        detected_languages: List of all detected languages
        method: Extraction method used (text_extraction, ocr, etc.)
        characters: Total character count
        **kwargs: Additional metadata fields

    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "source": os.path.basename(source_file),
        "pages": pages,
        "language": language,
        "detected_languages": detected_languages,
        "converted": datetime.now().isoformat(),
        "method": method,
        "characters": characters,
    }

    # Add any additional metadata
    metadata.update(kwargs)

    return metadata


def format_metadata_header(metadata: dict[str, Any]) -> str:
    """
    Format metadata as YAML front matter for markdown files

    Args:
        metadata: Dictionary of metadata key-value pairs

    Returns:
        Formatted YAML front matter string
    """
    lines = ["---"]

    for key, value in metadata.items():
        if isinstance(value, list):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {value}")

    lines.append("---\n")

    return "\n".join(lines)
