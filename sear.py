#!/usr/bin/env python3
"""
SEAR CLI: Summarization-Enhanced Augmented Retrieval
Command-line interface for SEAR library

For programmatic use, import sear_core instead:
    from sear_core import index_file, search, list_corpuses
"""

import sys
import os
from sear_core import (
    index_file,
    search,
    extract_relevant_content,
    validate_input_file,
    list_corpuses,
    get_corpus_info,
    delete_corpus,
    is_gpu_available,
    get_gpu_info
)

# Optional document converter support
try:
    from doc_converter import convert_document, DOCX_AVAILABLE
    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False
    DOCX_AVAILABLE = False

def main():
    if len(sys.argv) < 2:
        print("""
SEAR: Summarization-Enhanced Augmented Retrieval
(Ollama-based FAISS semantic search, no external APIs needed)

Usage:
  python sear.py index <file.txt> [corpus-name] [--gpu|--no-gpu]
  python sear.py search "query" [--corpus name1,name2,...] [--temperature 0.0-1.0] [--provider ollama|anthropic] [--api-key KEY] [--gpu|--no-gpu]
  python sear.py extract "query" [--output file.txt] [--corpus name1,name2,...] [--min-score 0.3] [--max-chunks N] [--gpu|--no-gpu]
  python sear.py convert <file.pdf|file.docx> [--output-dir DIR] [--no-normalize] [--force-ocr] [--lang heb|eng|heb+eng]
  python sear.py list
  python sear.py delete <corpus-name>
  python sear.py gpu-info

Commands:
  index     Index a text file into a named corpus
            Corpus name defaults to filename stem if not provided
            Use --gpu to force GPU acceleration
            Use --no-gpu to disable GPU acceleration

  search    Search across corpuses (defaults to ALL)
            Use --corpus flag to search specific corpuses only
            Use --temperature flag to control LLM creativity (0.0=deterministic, default)
            Use --provider to select LLM (ollama=default, anthropic=Claude 3.5 Sonnet 4.5)
            Use --api-key to provide Anthropic API key (or set ANTHROPIC_API_KEY env var)
            Use --gpu to force GPU acceleration
            Use --no-gpu to disable GPU acceleration

  extract   Extract ALL relevant content matching query above threshold
            Does NOT send to LLM - just extracts and saves raw content
            Use --output to specify output file (default: extracted_content.txt)
            Use --corpus flag to extract from specific corpuses only
            Use --min-score to set similarity threshold (default: 0.3)
            Use --max-chunks to limit total chunks extracted
            Use --gpu to force GPU acceleration
            Use --no-gpu to disable GPU acceleration

  convert   Convert PDF/DOCX documents to markdown for indexing
            Requires: pip install -e ".[converter]"
            Use --output-dir to specify output directory (default: converted_md/)
            Use --no-normalize to keep original formatting (niqqud, etc.)
            Use --force-ocr to force OCR on PDFs (PDF only)
            Use --lang to specify OCR language (PDF only, default: auto-detect)

  list      Show all available corpuses with metadata

  delete    Remove a corpus and all its data

  gpu-info  Show GPU availability and information

Setup:
  1. ollama pull all-minilm      # For embeddings
  2. ollama pull qwen2.5:0.5b    # For generation

Examples:
  # Index files (corpus name optional)
  python sear.py index my-code.txt
  python sear.py index docs.txt project-docs
  python sear.py index large-repo.txt --gpu  # Use GPU acceleration

  # Search all corpuses
  python sear.py search "authentication"
  python sear.py search "authentication" --gpu  # Use GPU acceleration

  # Search specific corpuses
  python sear.py search "git workflow" --corpus my-code,project-docs

  # Search with higher temperature for creative answers
  python sear.py search "what could go wrong?" --temperature 0.7

  # Search with Anthropic Claude (API key from environment)
  python sear.py search "authentication" --provider anthropic

  # Search with Anthropic Claude (API key from flag)
  python sear.py search "authentication" --provider anthropic --api-key sk-ant-xxx

  # Extract all relevant content (no LLM processing)
  python sear.py extract "authentication security validation"
  python sear.py extract "testing methodologies" --output testing_guide.txt
  python sear.py extract "neural networks" --corpus ml-corpus --min-score 0.4
  python sear.py extract "security audit" --max-chunks 200 --gpu

  # Convert documents to markdown (then index them)
  python sear.py convert document.pdf
  python sear.py convert report.docx --output-dir markdown_output/
  python sear.py convert scanned.pdf --force-ocr --lang heb+eng

  # List available corpuses
  python sear.py list

  # Delete a corpus
  python sear.py delete my-code

  # Check GPU availability
  python sear.py gpu-info

Library Usage:
  For programmatic use, import sear_core:
    from sear_core import index_file, search, list_corpuses
    index_file("code.txt", "my-corpus")
    results = search("my query", corpuses=["my-corpus"])
""")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "index":
        if len(sys.argv) < 3:
            print("❌ Error: Missing file path")
            print("Usage: python sear.py index <file.txt> [corpus-name] [--gpu|--no-gpu]")
            sys.exit(1)

        # Parse arguments
        filepath = sys.argv[2]
        corpus_name = None
        use_gpu = None

        # Parse remaining arguments
        for i in range(3, len(sys.argv)):
            if sys.argv[i] == '--gpu':
                use_gpu = True
            elif sys.argv[i] == '--no-gpu':
                use_gpu = False
            elif corpus_name is None and not sys.argv[i].startswith('--'):
                corpus_name = sys.argv[i]

        try:
            index_file(filepath, corpus_name, use_gpu=use_gpu)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("❌ Error: Missing search query")
            print("Usage: python sear.py search \"query\" [--corpus name1,name2,...] [--temperature 0.0-1.0] [--provider ollama|anthropic] [--api-key KEY] [--gpu|--no-gpu]")
            sys.exit(1)

        # Parse arguments
        args = sys.argv[2:]
        query_parts = []
        corpuses = None
        temperature = 0.0
        use_gpu = None
        provider = 'ollama'
        api_key = None

        i = 0
        while i < len(args):
            if args[i] == "--corpus":
                if i + 1 >= len(args):
                    print("❌ Error: --corpus flag requires corpus names")
                    sys.exit(1)
                corpuses = args[i + 1].split(',')
                i += 2
            elif args[i] == "--temperature":
                if i + 1 >= len(args):
                    print("❌ Error: --temperature flag requires a value (0.0-1.0)")
                    sys.exit(1)
                try:
                    temperature = float(args[i + 1])
                    if not 0.0 <= temperature <= 1.0:
                        print("❌ Error: temperature must be between 0.0 and 1.0")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ Error: invalid temperature value '{args[i + 1]}' (must be a number)")
                    sys.exit(1)
                i += 2
            elif args[i] == "--provider":
                if i + 1 >= len(args):
                    print("❌ Error: --provider flag requires a value (ollama or anthropic)")
                    sys.exit(1)
                provider = args[i + 1]
                if provider not in ['ollama', 'anthropic']:
                    print(f"❌ Error: invalid provider '{provider}' (must be 'ollama' or 'anthropic')")
                    sys.exit(1)
                i += 2
            elif args[i] == "--api-key":
                if i + 1 >= len(args):
                    print("❌ Error: --api-key flag requires an API key")
                    sys.exit(1)
                api_key = args[i + 1]
                i += 2
            elif args[i] == "--gpu":
                use_gpu = True
                i += 1
            elif args[i] == "--no-gpu":
                use_gpu = False
                i += 1
            else:
                query_parts.append(args[i])
                i += 1

        query = ' '.join(query_parts)
        if not query:
            print("❌ Error: Missing search query")
            sys.exit(1)

        try:
            search(query, corpuses=corpuses, temperature=temperature, use_gpu=use_gpu, provider=provider, api_key=api_key)
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif cmd == "extract":
        if len(sys.argv) < 3:
            print("❌ Error: Missing search query")
            print("Usage: python sear.py extract \"query\" [--output file.txt] [--corpus name1,name2,...] [--min-score 0.3] [--max-chunks N] [--gpu|--no-gpu]")
            sys.exit(1)

        # Parse arguments
        args = sys.argv[2:]
        query_parts = []
        corpuses = None
        output_file = None
        min_score = 0.3
        max_chunks = None
        use_gpu = None

        i = 0
        while i < len(args):
            if args[i] == "--output":
                if i + 1 >= len(args):
                    print("❌ Error: --output flag requires a file path")
                    sys.exit(1)
                output_file = args[i + 1]
                i += 2
            elif args[i] == "--corpus":
                if i + 1 >= len(args):
                    print("❌ Error: --corpus flag requires corpus names")
                    sys.exit(1)
                corpuses = args[i + 1].split(',')
                i += 2
            elif args[i] == "--min-score":
                if i + 1 >= len(args):
                    print("❌ Error: --min-score flag requires a value (0.0-1.0)")
                    sys.exit(1)
                try:
                    min_score = float(args[i + 1])
                    if not 0.0 <= min_score <= 1.0:
                        print("❌ Error: min-score must be between 0.0 and 1.0")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ Error: invalid min-score value '{args[i + 1]}' (must be a number)")
                    sys.exit(1)
                i += 2
            elif args[i] == "--max-chunks":
                if i + 1 >= len(args):
                    print("❌ Error: --max-chunks flag requires a number")
                    sys.exit(1)
                try:
                    max_chunks = int(args[i + 1])
                    if max_chunks <= 0:
                        print("❌ Error: max-chunks must be a positive number")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ Error: invalid max-chunks value '{args[i + 1]}' (must be an integer)")
                    sys.exit(1)
                i += 2
            elif args[i] == "--gpu":
                use_gpu = True
                i += 1
            elif args[i] == "--no-gpu":
                use_gpu = False
                i += 1
            else:
                query_parts.append(args[i])
                i += 1

        query = ' '.join(query_parts)
        if not query:
            print("❌ Error: Missing search query")
            sys.exit(1)

        try:
            extract_relevant_content(
                query=query,
                corpuses=corpuses,
                output_file=output_file,
                min_score=min_score,
                max_chunks=max_chunks,
                use_gpu=use_gpu,
                verbose=True
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif cmd == "list":
        corpuses = list_corpuses()
        if not corpuses:
            print("No corpuses found. Index a file first:")
            print("  python sear.py index <file.txt>")
            sys.exit(0)

        print(f"\nAvailable corpuses ({len(corpuses)}):")
        print("=" * 80)
        for corpus_name in corpuses:
            try:
                info = get_corpus_info(corpus_name)
                chunk_count = info.get('chunk_count', '?')
                model = info.get('embedding_model', '?')
                dim = info.get('embedding_dim', '?')
                print(f"  {corpus_name}")
                print(f"    - Chunks: {chunk_count}")
                print(f"    - Model: {model} ({dim}d)")
                print(f"    - Source: {info.get('file_path', '?')}")
            except (FileNotFoundError, ValueError) as e:
                print(f"  {corpus_name} (error: {e})")
        print("=" * 80)

    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("❌ Error: Missing corpus name")
            print("Usage: python sear.py delete <corpus-name>")
            sys.exit(1)

        corpus_name = sys.argv[2]
        try:
            delete_corpus(corpus_name)
            print(f"✅ Deleted corpus: {corpus_name}")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif cmd == "convert":
        if not CONVERTER_AVAILABLE:
            print("❌ Error: Document converter not installed")
            print("\nTo enable document conversion:")
            print("  pip install -e \".[converter]\"")
            print("\nThis will install support for PDF and DOCX conversion to markdown.")
            sys.exit(1)

        if len(sys.argv) < 3:
            print("❌ Error: Missing file path")
            print("Usage: python sear.py convert <file.pdf|file.docx> [--output-dir DIR] [--no-normalize] [--force-ocr] [--lang heb|eng|heb+eng]")
            sys.exit(1)

        # Parse arguments
        filepath = sys.argv[2]
        output_dir = "converted_md"
        normalize = True
        force_ocr = False
        lang = None

        # Parse flags
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--no-normalize':
                normalize = False
                i += 1
            elif sys.argv[i] == '--force-ocr':
                force_ocr = True
                i += 1
            elif sys.argv[i] == '--lang' and i + 1 < len(sys.argv):
                lang = sys.argv[i + 1]
                i += 2
            else:
                print(f"❌ Unknown flag: {sys.argv[i]}")
                sys.exit(1)

        # Check file exists
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found: {filepath}")
            sys.exit(1)

        # Check file format
        ext = os.path.splitext(filepath.lower())[1]
        if ext not in ['.pdf', '.docx']:
            print(f"❌ Error: Unsupported format: {ext}")
            print("Supported formats: .pdf" + (", .docx" if DOCX_AVAILABLE else ""))
            sys.exit(1)

        try:
            print(f"📄 Converting {os.path.basename(filepath)}...")
            kwargs = {
                'output_dir': output_dir,
                'force_ocr': force_ocr,
                'normalize': normalize
            }
            if lang is not None:
                kwargs['lang'] = lang

            success, elapsed, metadata = convert_document(filepath, **kwargs)

            if success:
                output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(filepath))[0] + '.md')
                print(f"✅ Converted in {elapsed:.2f}s")
                print(f"   Pages: {metadata.get('pages', '?')}")
                print(f"   Language: {metadata.get('language', '?')}")
                print(f"   Method: {metadata.get('method', '?')}")
                print(f"   Output: {output_file}")
                print("\n💡 Now you can index it:")
                print(f"   python sear.py index {output_file}")
            else:
                print("❌ Conversion failed")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            sys.exit(1)

    elif cmd == "gpu-info":
        print("\n" + "="*80)
        print("GPU INFORMATION")
        print("="*80)

        if is_gpu_available():
            gpu_info = get_gpu_info()
            print(f"✅ GPU acceleration available")
            print(f"   Number of GPUs: {gpu_info.get('num_gpus', 'Unknown')}")
            print(f"\nGPU support will be automatically used for faster indexing and search.")
            print(f"Use --no-gpu flag to disable GPU acceleration if needed.")
        else:
            print("❌ GPU acceleration not available")
            print("\nTo enable GPU support:")
            print("  1. Install CUDA toolkit (https://developer.nvidia.com/cuda-downloads)")
            print("  2. Uninstall faiss-cpu: pip uninstall faiss-cpu")
            print("  3. Install faiss-gpu: pip install faiss-gpu")
            print("\nSEAR will continue to work with CPU (slower but functional).")

        print("="*80)

    else:
        print(f"❌ Unknown command: {cmd}")
        print("Valid commands: index, search, extract, convert, list, delete, gpu-info")
        sys.exit(1)

if __name__ == "__main__":
    main()
