#!/usr/bin/env python3
"""
Text Processing Utilities for Document Conversion
Handles text normalization, language detection, and BiDi processing
"""

import re
import unicodedata
from typing import Tuple, List, Optional
from bidi import algorithm as bidi_algorithm


def normalize_text_for_llm(text: str, remove_niqqud: bool = True,
                           normalize_whitespace: bool = True) -> str:
    """
    Normalize text to reduce token waste and cognitive load on LLMs

    Removes:
    - Hebrew niqqud (vowel points): ְ ֱ ֲ ֳ ִ ֵ ֶ ַ ָ ֹ ֺ ֻ ּ ֽ ־ ׀ ׃ ׄ ׅ ׆
    - Other Hebrew diacritics and cantillation marks
    - Soft hyphens and zero-width characters
    - Excessive whitespace and formatting
    - Control characters
    """
    if not text:
        return text

    # Remove Hebrew niqqud and diacritics (U+0591 to U+05C7)
    if remove_niqqud:
        # Hebrew vowel points, cantillation marks, and other diacritics
        text = re.sub(r'[\u0591-\u05C7]', '', text)
        # Additional Hebrew marks
        text = re.sub(r'[\u05F3\u05F4]', '', text)  # Geresh and Gershayim marks

    # Remove soft hyphens
    text = text.replace('\u00AD', '')

    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C'
                   or char in '\n\t')

    if normalize_whitespace:
        # Normalize multiple spaces to single space
        text = re.sub(r' {2,}', ' ', text)

        # Remove trailing whitespace from each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        # Reduce excessive blank lines (more than 2 consecutive) to 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove spaces before punctuation
        text = re.sub(r' +([.,;:!?])', r'\1', text)

    # Remove bullet points and list markers that don't add semantic value
    # Keep the structure but simplify markers
    text = re.sub(r'^[\s]*[•·∙●○◦▪▫■□▬▭▮▯◆◇◈◊][\s]+', '- ', text, flags=re.MULTILINE)

    # Normalize Unicode to NFC form (canonical composition)
    text = unicodedata.normalize('NFC', text)

    return text.strip()


def detect_language(text: str) -> Tuple[str, List[str]]:
    """
    Detect languages in text
    Returns: (primary_language, list_of_detected_languages)
    """
    hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = hebrew_chars + english_chars

    detected = []
    primary = "unknown"

    if total_alpha > 0:
        hebrew_ratio = hebrew_chars / total_alpha
        english_ratio = english_chars / total_alpha

        if hebrew_ratio > 0.1:
            detected.append("hebrew")
        if english_ratio > 0.1:
            detected.append("english")

        # Determine primary language
        if hebrew_ratio > english_ratio:
            primary = "hebrew"
        elif english_ratio > 0:
            primary = "english"

    return primary, detected if detected else ["unknown"]


def get_ocr_lang_code(detected_languages: List[str], lang_override: Optional[str] = None) -> str:
    """
    Convert detected languages to Tesseract language codes
    """
    if lang_override:
        return lang_override

    if "hebrew" in detected_languages and "english" in detected_languages:
        return "heb+eng"
    elif "hebrew" in detected_languages:
        return "heb"
    elif "english" in detected_languages:
        return "eng"
    else:
        return "eng"  # Default to English


def apply_bidi_algorithm(text: str) -> str:
    """
    Apply BiDi algorithm for RTL (Right-to-Left) text
    """
    return bidi_algorithm.get_display(text)
