#!/usr/bin/env python3
"""
Output Formatting for Document Conversion
Handles markdown generation and file writing
"""

import os
from typing import Dict, Any
from .metadata import format_metadata_header


def format_markdown_output(
    source_filename: str,
    content: str,
    metadata: Dict[str, Any],
    include_title: bool = True
) -> str:
    """
    Format document content as markdown with metadata header

    Args:
        source_filename: Original file name for title
        content: Main document content
        metadata: Document metadata dictionary
        include_title: Whether to include # Title header

    Returns:
        Formatted markdown string
    """
    output_parts = []

    # Add metadata header
    output_parts.append(format_metadata_header(metadata))

    # Add title
    if include_title:
        output_parts.append(f"# {source_filename}\n")

    # Add content
    output_parts.append(content)

    return '\n'.join(output_parts)


def create_page_separator(page_number: int) -> str:
    """
    Create a page separator marker for RAG pipelines

    Args:
        page_number: The page number to display

    Returns:
        Formatted page separator string
    """
    return f"\n\n---\n**Page {page_number}**\n---\n\n"


def save_markdown_file(
    content: str,
    output_dir: str,
    source_filename: str
) -> str:
    """
    Save markdown content to file with safe filename

    Args:
        content: Markdown content to save
        output_dir: Output directory path
        source_filename: Original source filename

    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate safe filename
    base_name = os.path.basename(source_filename)
    safe_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_', '.')).strip()

    # Remove original extension and add .md
    safe_name = os.path.splitext(safe_name)[0] + '.md'

    output_path = os.path.join(output_dir, safe_name)

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path
