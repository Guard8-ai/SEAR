#!/usr/bin/env python3
"""
Unified CLI for Document Conversion
Supports PDF, DOCX, and other document formats
"""

import argparse
import glob
import sys
import time

from ..converters import DOCX_AVAILABLE, get_converter_for_file


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Convert documents to Markdown with smart text extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PDFs with automatic detection
  doc-converter documents/*.pdf

  # Convert Word documents
  doc-converter documents/*.docx

  # Convert mixed formats
  doc-converter documents/report.pdf documents/memo.docx

  # Force OCR on PDFs
  doc-converter --force-ocr scanned.pdf

  # Specify language for OCR
  doc-converter --lang heb hebrew_doc.pdf

  # Custom output directory
  doc-converter --output-dir output/ documents/*

  # Disable text normalization
  doc-converter --no-normalize document.pdf
        """,
    )

    parser.add_argument("files", nargs="+", help="Document files to convert")

    # Output options
    parser.add_argument(
        "--output-dir", default="converted_md", help="Output directory (default: converted_md/)"
    )

    # PDF-specific options
    pdf_group = parser.add_argument_group("PDF options")
    pdf_group.add_argument(
        "--force-ocr", action="store_true", help="Force OCR even if text layer exists (PDF only)"
    )
    pdf_group.add_argument(
        "--lang",
        choices=["heb", "eng", "heb+eng"],
        help="Override language detection for OCR (PDF only)",
    )
    pdf_group.add_argument(
        "--dpi", type=int, default=300, help="DPI for OCR image conversion (default: 300, PDF only)"
    )

    # Text processing options
    text_group = parser.add_argument_group("Text processing options")
    text_group.add_argument(
        "--no-bidi",
        action="store_true",
        help="Disable BiDi algorithm for RTL text (default varies by format: PDF=enabled, DOCX=disabled)",
    )
    text_group.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization (keep niqqud, styling, whitespace)",
    )

    # Format override
    parser.add_argument(
        "--format", choices=["pdf", "docx"], help="Force specific format (auto-detected by default)"
    )

    args = parser.parse_args()

    # Expand glob patterns
    file_list = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            file_list.extend(matches)
        else:
            file_list.append(pattern)

    if not file_list:
        print("No files found")
        return 1

    # Display banner
    print("Document to Markdown Conversion")
    print(f"Found {len(file_list)} file(s) to convert")
    print(f"Output directory: {args.output_dir}/")
    if not DOCX_AVAILABLE:
        print("Note: DOCX support not available (install python-docx to enable)")
    print()

    # Statistics
    total_start = time.time()
    successful = 0
    failed = 0
    total_pages = 0
    total_chars = 0
    skipped = 0

    # Process each file
    for i, file_path in enumerate(file_list, 1):
        print(f"\n[{i}/{len(file_list)}]")

        try:
            # Prepare converter kwargs based on file type
            converter_kwargs = {"normalize": not args.no_normalize}

            # Only set apply_bidi if explicitly disabled by user
            # This allows converters to use their own defaults (e.g., DOCX uses False, PDF uses True)
            if args.no_bidi:
                converter_kwargs["apply_bidi"] = False

            # Add PDF-specific options
            if file_path.lower().endswith(".pdf"):
                converter_kwargs.update(
                    {"force_ocr": args.force_ocr, "dpi": args.dpi, "lang": args.lang}
                )

            # Get appropriate converter
            converter = get_converter_for_file(file_path, **converter_kwargs)

            # Convert document
            success, elapsed, metadata = converter.convert(file_path, output_dir=args.output_dir)

            if success:
                successful += 1
                total_pages += metadata.get("pages", 0)
                total_chars += metadata.get("characters", 0)
            else:
                failed += 1

        except ValueError as e:
            print(f"Error: {e}")
            skipped += 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Final summary
    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Total files: {len(file_list)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if skipped > 0:
        print(f"  Skipped: {skipped}")
    print(f"Total pages/sections: {total_pages:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
    if successful > 0:
        print(f"Average per file: {total_elapsed/successful:.2f}s")
        if total_pages > 0:
            print(f"Average per page: {total_elapsed/total_pages:.2f}s")
    print(f"{'='*70}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
