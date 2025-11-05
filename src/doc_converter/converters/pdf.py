#!/usr/bin/env python3
"""
PDF Converter
Handles PDF to text conversion with OCR fallback
"""

from typing import Any, Optional

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from ..core import (
    apply_bidi_algorithm,
    create_page_separator,
    detect_language,
    generate_metadata,
    get_ocr_lang_code,
    normalize_text_for_llm,
)
from .base import BaseConverter


class PDFConverter(BaseConverter):
    """
    PDF document converter with smart text extraction and OCR fallback
    """

    supported_extensions = [".pdf"]

    def __init__(
        self,
        apply_bidi: bool = True,
        normalize: bool = True,
        force_ocr: bool = False,
        dpi: int = 300,
        lang: Optional[str] = None,
    ):
        """
        Initialize PDF converter

        Args:
            apply_bidi: Apply BiDi algorithm for RTL text
            normalize: Normalize text (remove niqqud, etc.)
            force_ocr: Force OCR even if text layer exists
            dpi: DPI for OCR image conversion
            lang: Language override for OCR (heb, eng, heb+eng)
        """
        super().__init__(apply_bidi=apply_bidi, normalize=normalize)
        self.force_ocr = force_ocr
        self.dpi = dpi
        self.lang = lang

    def _check_has_text_layer(self, pdf) -> tuple[bool, int]:
        """
        Check if PDF has a text layer by counting characters

        Returns:
            Tuple of (has_text, total_char_count)
        """
        total_chars = 0
        for page in pdf.pages:
            total_chars += len(page.chars)

        return total_chars > 0, total_chars

    def _extract_text_fast(self, pdf) -> tuple[str, int]:
        """
        Fast text extraction using pdfplumber

        Returns:
            Tuple of (extracted_text, page_count)
        """
        all_text = ""
        total_pages = len(pdf.pages)

        print("Using fast text extraction (PDF has text layer)")
        print(f"Total pages: {total_pages}")

        for page_num, page in enumerate(pdf.pages, 1):
            if page_num % 10 == 0 or page_num == 1:
                print(f"  Processing page {page_num}/{total_pages}...")

            text = page.extract_text()
            if text:
                # Normalize text to reduce token waste
                if self.normalize:
                    text = normalize_text_for_llm(text)

                # Apply BiDi algorithm for RTL text
                if self.apply_bidi:
                    text = apply_bidi_algorithm(text)

                # Add page separator
                all_text += create_page_separator(page_num)
                all_text += text + "\n\n"

        return all_text, total_pages

    def _extract_text_ocr(self, pdf_path: str) -> tuple[str, int, str]:
        """
        OCR-based text extraction using Tesseract

        Returns:
            Tuple of (extracted_text, page_count, detected_language)
        """
        print("Using OCR extraction (no text layer or forced OCR)")
        print(f"Converting PDF to images at {self.dpi} DPI...")

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=self.dpi)
        total_pages = len(images)

        print(f"Running OCR on {total_pages} pages...")

        all_text = ""
        sample_text = ""  # For language detection

        for page_num, image in enumerate(images, 1):
            if page_num % 5 == 0 or page_num == 1:
                print(f"  OCR page {page_num}/{total_pages}...")

            # First page: detect language if not specified
            ocr_lang = self.lang
            if page_num == 1 and not self.lang:
                # Quick OCR for language detection
                sample = pytesseract.image_to_string(image, lang="heb+eng")
                _, detected = detect_language(sample)
                ocr_lang = get_ocr_lang_code(detected)
                print(f"  Detected languages: {detected}, using OCR lang: {ocr_lang}")

            # Run OCR
            text = pytesseract.image_to_string(
                image,
                lang=ocr_lang or "eng",
                config="--psm 1",  # Automatic page segmentation with OSD
            )

            if text and text.strip():
                # Store sample for final language detection
                if page_num == 1:
                    sample_text = text

                # Normalize text
                if self.normalize:
                    text = normalize_text_for_llm(text)

                # Apply BiDi algorithm for RTL text
                if self.apply_bidi:
                    text = apply_bidi_algorithm(text)

                # Add page separator
                all_text += create_page_separator(page_num)
                all_text += text + "\n\n"

        # Detect final language from sample
        primary_lang, detected_langs = detect_language(sample_text or all_text[:1000])

        return all_text, total_pages, primary_lang

    def extract(self, file_path: str, **kwargs) -> tuple[str, dict[str, Any]]:
        """
        Extract text and metadata from PDF

        Args:
            file_path: Path to PDF file
            **kwargs: Additional parameters (ignored for compatibility)

        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        self.validate_file(file_path)

        # Step 1: PDF Analysis
        with pdfplumber.open(file_path) as pdf:
            has_text, char_count = self._check_has_text_layer(pdf)

            print("PDF Analysis:")
            print(f"  Pages: {len(pdf.pages)}")
            print(f"  Has text layer: {has_text} (char count: {char_count:,})")
            print(f"  Force OCR: {self.force_ocr}")
            print()

            # Step 2: Choose extraction method
            if has_text and not self.force_ocr:
                # Fast path: Direct text extraction
                all_text, total_pages = self._extract_text_fast(pdf)

                # Detect language from extracted text
                primary_lang, detected_langs = detect_language(all_text[:5000])
                method = "text_extraction"
            else:
                # Slow path: OCR
                all_text, total_pages, primary_lang = self._extract_text_ocr(file_path)
                _, detected_langs = detect_language(all_text[:5000])
                method = "ocr"

        # Generate metadata
        metadata = generate_metadata(
            source_file=file_path,
            pages=total_pages,
            language=primary_lang,
            detected_languages=detected_langs,
            method=method,
            characters=len(all_text),
        )

        return all_text, metadata
