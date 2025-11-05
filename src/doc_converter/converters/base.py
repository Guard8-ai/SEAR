#!/usr/bin/env python3
"""
Base Converter Abstract Class
Defines the interface for all document converters
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional


class BaseConverter(ABC):
    """
    Abstract base class for document converters

    All converters must implement the extract() method and define
    supported_extensions class attribute.
    """

    supported_extensions: List[str] = []

    def __init__(self, apply_bidi: bool = True, normalize: bool = True):
        """
        Initialize converter

        Args:
            apply_bidi: Whether to apply BiDi algorithm for RTL text
            normalize: Whether to normalize text (remove niqqud, excess whitespace, etc.)
        """
        self.apply_bidi = apply_bidi
        self.normalize = normalize

    @classmethod
    def supports_format(cls, file_path: str) -> bool:
        """
        Check if this converter supports the given file format

        Args:
            file_path: Path to file to check

        Returns:
            True if file extension is supported
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in cls.supported_extensions

    @abstractmethod
    def extract(self, file_path: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from document

        Args:
            file_path: Path to document file
            **kwargs: Additional converter-specific parameters

        Returns:
            Tuple of (extracted_text, metadata_dict)

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If extraction fails
        """
        pass

    def convert(
        self,
        file_path: str,
        output_dir: str = "converted_md",
        **kwargs
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Convert document to markdown with metadata

        Args:
            file_path: Path to source document
            output_dir: Output directory for markdown files
            **kwargs: Additional converter-specific parameters

        Returns:
            Tuple of (success, elapsed_time, metadata)
        """
        from ..core import (
            save_markdown_file,
            format_markdown_output
        )

        start_time = time.time()
        base_name = os.path.basename(file_path)

        print(f"\n{'='*70}")
        print(f"Processing: {base_name}")
        print(f"{'='*70}")

        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return False, 0, {}

        try:
            # Extract text and metadata
            text, metadata = self.extract(file_path, **kwargs)

            if not text or not text.strip():
                print("Warning: No text extracted from document")
                return False, time.time() - start_time, {}

            # Format markdown output
            markdown_content = format_markdown_output(
                source_filename=base_name,
                content=text,
                metadata=metadata
            )

            # Save to file
            output_path = save_markdown_file(
                content=markdown_content,
                output_dir=output_dir,
                source_filename=file_path
            )

            elapsed = time.time() - start_time

            print(f"\nSuccess!")
            print(f"  Output: {output_path}")
            print(f"  Method: {metadata.get('method', 'unknown')}")
            print(f"  Language: {metadata.get('language', 'unknown')}")
            print(f"  Pages: {metadata.get('pages', 0)}")
            print(f"  Characters: {len(text):,}")
            print(f"  Time: {elapsed:.2f}s")

            return True, elapsed, metadata

        except Exception as e:
            print(f"\nError processing document: {e}")
            import traceback
            traceback.print_exc()
            return False, time.time() - start_time, {}

    def validate_file(self, file_path: str) -> None:
        """
        Validate that file exists and is supported

        Args:
            file_path: Path to file to validate

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports_format(file_path):
            raise ValueError(
                f"Unsupported file format. Supported extensions: {self.supported_extensions}"
            )
