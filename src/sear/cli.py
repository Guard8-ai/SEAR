#!/usr/bin/env python3
"""
SEAR CLI: Summarization-Enhanced Augmented Retrieval
Command-line interface for SEAR library

For programmatic use, import sear instead:
    from sear import index_file, search, list_corpuses
"""

import os
import sys

from .core import (
    anthropic_generate,
    delete_corpus,
    execute_query,
    execute_sql_query,
    extract_relevant_content,
    get_corpus_info,
    get_gpu_info,
    index_file,
    is_gpu_available,
    list_corpuses,
    ollama_generate,
    search,
)

# Optional document converter support
try:
    from ..doc_converter import DOCX_AVAILABLE, convert_document

    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False
    DOCX_AVAILABLE = False


def send_chunks_to_llm(
    chunks, query, temperature=0.0, provider="ollama", api_key=None, verbose=True
):
    """
    Send filtered chunks to LLM for answer generation.

    Args:
        chunks: List of chunk dicts with keys: corpus, location, score, chunk
        query: Original query string
        temperature: LLM temperature
        provider: 'ollama' or 'anthropic'
        api_key: API key for Anthropic
        verbose: Print progress

    Returns:
        dict: Response with answer and sources
    """
    if not chunks:
        if verbose:
            print("❌ No chunks found matching the query")
        return {
            "answer": "No results found matching your query.",
            "sources": [],
            "quality_results": 0,
            "filtered_count": 0,
        }

    if verbose:
        print(f"🤖 Generating answer from {len(chunks)} results...")

    # Build context from chunks
    context = f"Question: {query}\n\n"
    for i, r in enumerate(chunks, 1):
        context += f"Source {i} from {r['corpus']} ({r['location']}):\n{r['chunk']}\n\n"

    prompt = f"""{context}
Based ONLY on the sources above, answer the question with precision and accuracy.

Instructions:
- Use specific facts, numbers, and quotes from the sources
- If sources contain contradictory information, acknowledge it
- If sources don't contain enough information to answer, state what's missing
- Keep your answer focused and fact-based
- Do not add information not present in the sources

Answer:"""

    # Generate answer
    if provider == "anthropic":
        answer = anthropic_generate(prompt, temperature=temperature, api_key=api_key)
    else:
        answer = ollama_generate(prompt, temperature=temperature)

    if verbose:
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)
        for i, result in enumerate(chunks, 1):
            print(f"{i}. [{result['corpus']}] {result['location']} (score: {result['score']:.3f})")
            print(f"   {result['chunk'][:200]}...")
            print()
        print("=" * 80)

    return {
        "answer": answer,
        "sources": chunks,
        "quality_results": len(chunks),
        "filtered_count": 0,
    }


def build_boolean_query(
    query,
    exclude=None,
    union=False,
    corpuses=None,
    min_score=0.3,
    max_chunks=None,
    semantic=False,
    threshold=0.7,
):
    """
    Convert CLI boolean arguments to JSON query format for execute_query().

    Args:
        query: Main query string (may contain commas if union=True)
        exclude: Optional exclusion query string (may contain commas)
        union: If True, split query by commas and create union operation
        corpuses: List of corpus names to search
        min_score: Minimum similarity score threshold
        max_chunks: Maximum number of chunks to return
        semantic: If True, use semantic similarity for exclusion
        threshold: Similarity threshold for semantic exclusion (0.0-1.0)

    Returns:
        dict: JSON query specification for execute_query()

    Examples:
        # Simple query: "physics lessons"
        >>> build_boolean_query("physics lessons")
        {"query": "physics lessons", "min_score": 0.3, ...}

        # Union: "thermo, quantum, EM" --union
        >>> build_boolean_query("thermo, quantum, EM", union=True)
        {"operation": "union", "queries": ["thermo", "quantum", "EM"], ...}

        # Difference: "physics" --exclude "mechanics"
        >>> build_boolean_query("physics", exclude="mechanics")
        {"operation": "difference", "query": "physics", "exclude": "mechanics", ...}

        # Complex: "thermo, quantum" --union --exclude "deprecated"
        >>> build_boolean_query("thermo, quantum", exclude="deprecated", union=True)
        {"operation": "difference", "left": {"operation": "union", "queries": [...]}, "right": {...}, ...}

        # Semantic exclusion: "physics" --exclude "mechanics" --semantic --threshold 0.75
        >>> build_boolean_query("physics", exclude="mechanics", semantic=True, threshold=0.75)
        {"operation": "difference", "query": "physics", "exclude": "mechanics", "semantic": True, "threshold": 0.75, ...}
    """
    # Base options
    query_spec = {"min_score": min_score, "sort": True, "merge_adjacent": True}

    if corpuses:
        query_spec["corpuses"] = corpuses
    if max_chunks:
        query_spec["max_results"] = max_chunks

    # Parse main query (handle union flag)
    if union:
        # Split by comma and strip whitespace
        queries = [q.strip() for q in query.split(",") if q.strip()]
        if len(queries) < 2:
            raise ValueError("Union requires at least 2 comma-separated queries")
        main_query = {"operation": "union", "queries": queries}
    else:
        # Simple query
        main_query = {"query": query}

    # Parse exclusion (handle multiple exclusions)
    if exclude:
        exclude_queries = [q.strip() for q in exclude.split(",") if q.strip()]

        if len(exclude_queries) == 1:
            # Single exclusion
            exclude_query = {"query": exclude_queries[0]}
        else:
            # Multiple exclusions: union them first
            exclude_query = {"operation": "union", "queries": exclude_queries}

        # Build difference operation
        query_spec["operation"] = "difference"
        query_spec["left"] = main_query
        query_spec["right"] = exclude_query
        # Add semantic filtering options if enabled
        if semantic:
            query_spec["semantic"] = True
            query_spec["threshold"] = threshold
    else:
        # No exclusion: use main query directly
        query_spec.update(main_query)

    return query_spec


def main():
    if len(sys.argv) < 2:
        print(
            """
SEAR: Summarization-Enhanced Augmented Retrieval
(Ollama-based FAISS semantic search, no external APIs needed)

Usage:
  python sear.py index <file.txt> [corpus-name] [--gpu|--no-gpu]
  python sear.py search "query" [--corpus name1,name2,...] [--exclude "query"] [--union] [--semantic] [--threshold 0.7] [--temperature 0.0-1.0] [--provider ollama|anthropic] [--api-key KEY] [--gpu|--no-gpu]
  python sear.py extract "query" [--output file.txt] [--corpus name1,name2,...] [--exclude "query"] [--union] [--semantic] [--threshold 0.7] [--min-score 0.3] [--max-chunks N] [--gpu|--no-gpu]
  python sear.py sql "SELECT * FROM search(...)" [--mode search|extract] [--output file.txt] [--temperature 0.0-1.0] [--provider ollama|anthropic] [--api-key KEY] [--gpu|--no-gpu]
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
            Use --exclude to exclude matching topics (e.g., "physics" --exclude "mechanics")
            Use --union to combine comma-separated queries (e.g., "thermo, quantum, EM" --union)
            Use --semantic to enable semantic exclusion (checks content similarity, not just exact matches)
            Use --threshold to set semantic similarity threshold (default: 0.7, requires --semantic)
            Use --temperature flag to control LLM creativity (0.0=deterministic, default)
            Use --provider to select LLM (ollama=default, anthropic=Claude 3.5 Sonnet 4.5)
            Use --api-key to provide Anthropic API key (or set ANTHROPIC_API_KEY env var)
            Use --gpu to force GPU acceleration
            Use --no-gpu to disable GPU acceleration

  extract   Extract ALL relevant content matching query above threshold
            Does NOT send to LLM - just extracts and saves raw content
            Use --output to specify output file (default: extracted_content.txt)
            Use --corpus flag to extract from specific corpuses only
            Use --exclude to exclude matching topics (e.g., "physics" --exclude "mechanics")
            Use --union to combine comma-separated queries (e.g., "thermo, quantum, EM" --union)
            Use --semantic to enable semantic exclusion (checks content similarity, not just exact matches)
            Use --threshold to set semantic similarity threshold (default: 0.7, requires --semantic)
            Use --min-score to set similarity threshold (default: 0.3)
            Use --max-chunks to limit total chunks extracted
            Use --gpu to force GPU acceleration
            Use --no-gpu to disable GPU acceleration

  sql       Execute SQL-like queries on corpuses
            Supports UNION, EXCEPT, INTERSECT operations
            Use --mode to choose 'search' (LLM answer) or 'extract' (save to file)
            Use --output to specify output file for extract mode
            Use --temperature flag to control LLM creativity (search mode only)
            Use --provider to select LLM (ollama=default, anthropic=Claude 3.5 Sonnet 4.5)
            Use --api-key to provide Anthropic API key (or set ANTHROPIC_API_KEY env var)
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

  # Boolean search: exclude topics
  python sear.py search "physics lessons" --exclude "mechanics"
  python sear.py search "physics" --exclude "mechanics, optics" --semantic --threshold 0.75

  # Boolean search: union of topics
  python sear.py search "thermodynamics, quantum mechanics, electromagnetism" --union

  # Boolean search: complex queries (union + exclusion)
  python sear.py search "security, authentication" --union --exclude "deprecated, legacy"

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

  # Boolean extract: exclude topics
  python sear.py extract "physics" --exclude "mechanics" --output physics_no_mechanics.txt

  # Boolean extract: union of topics
  python sear.py extract "thermo, quantum, EM" --union --output topics.txt

  # SQL queries (search mode - send to LLM)
  python sear.py sql "SELECT * FROM search(\"authentication\")"
  python sear.py sql "SELECT * FROM search(\"security\") UNION SELECT * FROM search(\"auth\")"
  python sear.py sql "SELECT * FROM search(\"physics\") EXCEPT SELECT * FROM search(\"mechanics\")"
  python sear.py sql "SELECT * FROM search(\"API\") INTERSECT SELECT * FROM search(\"security\")"

  # SQL queries with options
  python sear.py sql "SELECT * FROM search(\"security\") WHERE corpus IN ('backend', 'api') AND min_score >= 0.35"
  python sear.py sql "SELECT * FROM search(\"physics\") EXCEPT SELECT * FROM search(\"mechanics\") WHERE semantic = true AND threshold >= 0.75"

  # SQL extract mode (save to file without LLM)
  python sear.py sql "SELECT * FROM search(\"security\") EXCEPT SELECT * FROM search(\"deprecated\")" --mode extract --output security_clean.txt

  # Complex nested SQL queries
  python sear.py sql "SELECT * FROM (SELECT * FROM search(\"security\") UNION SELECT * FROM search(\"auth\")) EXCEPT SELECT * FROM search(\"deprecated\")"

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
"""
        )
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
            if sys.argv[i] == "--gpu":
                use_gpu = True
            elif sys.argv[i] == "--no-gpu":
                use_gpu = False
            elif corpus_name is None and not sys.argv[i].startswith("--"):
                corpus_name = sys.argv[i]

        try:
            index_file(filepath, corpus_name, use_gpu=use_gpu)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("❌ Error: Missing search query")
            print(
                'Usage: python sear.py search "query" [--corpus name1,name2,...] [--exclude "query"] [--union] [--semantic] [--threshold 0.7] [--temperature 0.0-1.0] [--provider ollama|anthropic] [--api-key KEY] [--gpu|--no-gpu]'
            )
            sys.exit(1)

        # Parse arguments
        args = sys.argv[2:]
        query_parts = []
        corpuses = None
        exclude_query = None
        use_union = False
        use_semantic = False
        semantic_threshold = 0.7
        temperature = 0.0
        use_gpu = None
        provider = "ollama"
        api_key = None

        i = 0
        while i < len(args):
            if args[i] == "--corpus":
                if i + 1 >= len(args):
                    print("❌ Error: --corpus flag requires corpus names")
                    sys.exit(1)
                corpuses = args[i + 1].split(",")
                i += 2
            elif args[i] == "--exclude":
                if i + 1 >= len(args):
                    print("❌ Error: --exclude flag requires an exclusion query")
                    sys.exit(1)
                exclude_query = args[i + 1]
                i += 2
            elif args[i] == "--union":
                use_union = True
                i += 1
            elif args[i] == "--semantic":
                use_semantic = True
                i += 1
            elif args[i] == "--threshold":
                if i + 1 >= len(args):
                    print("❌ Error: --threshold flag requires a value (0.0-1.0)")
                    sys.exit(1)
                try:
                    semantic_threshold = float(args[i + 1])
                    if not 0.0 <= semantic_threshold <= 1.0:
                        print("❌ Error: threshold must be between 0.0 and 1.0")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ Error: invalid threshold value '{args[i + 1]}' (must be a number)")
                    sys.exit(1)
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
                if provider not in ["ollama", "anthropic"]:
                    print(
                        f"❌ Error: invalid provider '{provider}' (must be 'ollama' or 'anthropic')"
                    )
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

        query = " ".join(query_parts)
        if not query:
            print("❌ Error: Missing search query")
            sys.exit(1)

        try:
            # Check if we need to use boolean query logic
            if exclude_query or use_union:
                # Build boolean query JSON
                query_spec = build_boolean_query(
                    query=query,
                    exclude=exclude_query,
                    union=use_union,
                    corpuses=corpuses,
                    min_score=0.3,  # Use default for search
                    max_chunks=None,
                    semantic=use_semantic,
                    threshold=semantic_threshold,
                )

                # Execute boolean query to get filtered chunks
                chunks = execute_query(query_spec, use_gpu=use_gpu, verbose=True)

                # Send filtered chunks to LLM for answer generation
                send_chunks_to_llm(
                    chunks,
                    query,
                    temperature=temperature,
                    provider=provider,
                    api_key=api_key,
                    verbose=True,
                )
            else:
                # Standard search without boolean operations
                search(
                    query,
                    corpuses=corpuses,
                    temperature=temperature,
                    use_gpu=use_gpu,
                    provider=provider,
                    api_key=api_key,
                )
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif cmd == "extract":
        if len(sys.argv) < 3:
            print("❌ Error: Missing search query")
            print(
                'Usage: python sear.py extract "query" [--output file.txt] [--corpus name1,name2,...] [--exclude "query"] [--union] [--semantic] [--threshold 0.7] [--min-score 0.3] [--max-chunks N] [--gpu|--no-gpu]'
            )
            sys.exit(1)

        # Parse arguments
        args = sys.argv[2:]
        query_parts = []
        corpuses = None
        output_file = None
        exclude_query = None
        use_union = False
        use_semantic = False
        semantic_threshold = 0.7
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
                corpuses = args[i + 1].split(",")
                i += 2
            elif args[i] == "--exclude":
                if i + 1 >= len(args):
                    print("❌ Error: --exclude flag requires an exclusion query")
                    sys.exit(1)
                exclude_query = args[i + 1]
                i += 2
            elif args[i] == "--union":
                use_union = True
                i += 1
            elif args[i] == "--semantic":
                use_semantic = True
                i += 1
            elif args[i] == "--threshold":
                if i + 1 >= len(args):
                    print("❌ Error: --threshold flag requires a value (0.0-1.0)")
                    sys.exit(1)
                try:
                    semantic_threshold = float(args[i + 1])
                    if not 0.0 <= semantic_threshold <= 1.0:
                        print("❌ Error: threshold must be between 0.0 and 1.0")
                        sys.exit(1)
                except ValueError:
                    print(f"❌ Error: invalid threshold value '{args[i + 1]}' (must be a number)")
                    sys.exit(1)
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
                    print(
                        f"❌ Error: invalid max-chunks value '{args[i + 1]}' (must be an integer)"
                    )
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

        query = " ".join(query_parts)
        if not query:
            print("❌ Error: Missing search query")
            sys.exit(1)

        try:
            # Check if we need to use boolean query logic
            if exclude_query or use_union:
                # Build boolean query JSON
                query_spec = build_boolean_query(
                    query=query,
                    exclude=exclude_query,
                    union=use_union,
                    corpuses=corpuses,
                    min_score=min_score,
                    max_chunks=max_chunks,
                    semantic=use_semantic,
                    threshold=semantic_threshold,
                )

                # Execute boolean query to get filtered chunks
                chunks = execute_query(query_spec, use_gpu=use_gpu, verbose=True)

                # Write chunks to output file
                if output_file is None:
                    output_file = "extracted_content.txt"

                with open(output_file, "w") as f:
                    f.write(f"# Extracted Content for Query: {query}\n")
                    f.write(f"# Total chunks: {len(chunks)}\n")
                    if exclude_query:
                        f.write(f"# Excluded: {exclude_query}\n")
                    if use_union:
                        f.write("# Union mode: enabled\n")
                    f.write("\n" + "=" * 80 + "\n\n")

                    for i, chunk in enumerate(chunks, 1):
                        f.write(f"## Chunk {i} from {chunk['corpus']}\n")
                        f.write(f"Location: {chunk['location']}\n")
                        f.write(f"Score: {chunk['score']:.3f}\n\n")
                        f.write(chunk["chunk"])
                        f.write("\n\n" + "-" * 80 + "\n\n")

                print(f"✅ Extracted {len(chunks)} chunks to: {output_file}")
            else:
                # Standard extract without boolean operations
                extract_relevant_content(
                    query=query,
                    corpuses=corpuses,
                    output_file=output_file,
                    min_score=min_score,
                    max_chunks=max_chunks,
                    use_gpu=use_gpu,
                    verbose=True,
                )
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif cmd == "sql":
        if len(sys.argv) < 3:
            print("❌ Error: Missing SQL query")
            print(r'Usage: python sear.py sql "SELECT * FROM search(\"query\") ..."')
            print("\nExamples:")
            print("  # Simple query")
            print(r'  python sear.py sql "SELECT * FROM search(\"physics\")"')
            print("")
            print("  # Union (OR)")
            print(
                r'  python sear.py sql "SELECT * FROM search(\"security\") UNION SELECT * FROM search(\"auth\")"'
            )
            print("")
            print("  # Difference (EXCEPT)")
            print(
                r'  python sear.py sql "SELECT * FROM search(\"physics\") EXCEPT SELECT * FROM search(\"mechanics\")"'
            )
            print("")
            print("  # With options (WHERE clause)")
            print(
                r'  python sear.py sql "SELECT * FROM search(\"security\") WHERE corpus IN (\'backend\', \'api\') AND min_score >= 0.35"'
            )
            sys.exit(1)

        # Parse remaining arguments
        sql_query = sys.argv[2]
        use_gpu = None
        mode = "search"  # default mode
        temperature = 0.0
        provider = "ollama"
        api_key = None
        output_file = None

        # Parse additional flags
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--gpu":
                use_gpu = True
                i += 1
            elif sys.argv[i] == "--no-gpu":
                use_gpu = False
                i += 1
            elif sys.argv[i] == "--mode":
                if i + 1 >= len(sys.argv):
                    print("❌ Error: --mode flag requires a value (search|extract)")
                    sys.exit(1)
                mode = sys.argv[i + 1]
                if mode not in ["search", "extract"]:
                    print("❌ Error: --mode must be 'search' or 'extract'")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--temperature":
                if i + 1 >= len(sys.argv):
                    print("❌ Error: --temperature flag requires a value (0.0-1.0)")
                    sys.exit(1)
                try:
                    temperature = float(sys.argv[i + 1])
                    if not 0.0 <= temperature <= 1.0:
                        print("❌ Error: Temperature must be between 0.0 and 1.0")
                        sys.exit(1)
                except ValueError:
                    print("❌ Error: Temperature must be a number")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--provider":
                if i + 1 >= len(sys.argv):
                    print("❌ Error: --provider flag requires a value (ollama|anthropic)")
                    sys.exit(1)
                provider = sys.argv[i + 1]
                if provider not in ["ollama", "anthropic"]:
                    print("❌ Error: --provider must be 'ollama' or 'anthropic'")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--api-key":
                if i + 1 >= len(sys.argv):
                    print("❌ Error: --api-key flag requires a value")
                    sys.exit(1)
                api_key = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--output":
                if i + 1 >= len(sys.argv):
                    print("❌ Error: --output flag requires a file path")
                    sys.exit(1)
                output_file = sys.argv[i + 1]
                i += 2
            else:
                print(f"❌ Error: Unknown flag: {sys.argv[i]}")
                sys.exit(1)

        try:
            # Execute SQL query
            print("🔍 Parsing SQL query...")
            chunks = execute_sql_query(sql_query, use_gpu=use_gpu, verbose=True)

            if not chunks:
                print("\n❌ No results found matching your SQL query")
                sys.exit(0)

            print(f"\n✅ Found {len(chunks)} chunks matching SQL query")

            # Mode: search (send to LLM) or extract (save to file)
            if mode == "search":
                # Send to LLM for answer generation
                # Extract original query from SQL for the prompt
                # For now, we'll use a generic prompt mentioning SQL query
                send_chunks_to_llm(
                    chunks,
                    query=f"Results from SQL query: {sql_query}",
                    temperature=temperature,
                    provider=provider,
                    api_key=api_key,
                    verbose=True,
                )
            else:
                # Extract mode: save to file
                if not output_file:
                    output_file = "extracted_sql_results.txt"

                # Format and save content
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"SQL Query: {sql_query}\n")
                    f.write(f"Total chunks: {len(chunks)}\n")
                    f.write("=" * 80 + "\n\n")

                    for i, chunk in enumerate(chunks, 1):
                        f.write(
                            f"[{i}] {chunk['corpus']} - {chunk['location']} (score: {chunk['score']:.3f})\n"
                        )
                        f.write(f"{chunk['chunk']}\n")
                        f.write("\n" + "-" * 80 + "\n\n")

                print(f"\n✅ Content extracted to: {output_file}")

        except ValueError as e:
            print(f"❌ SQL Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error executing SQL query: {e}")
            import traceback

            traceback.print_exc()
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
                chunk_count = info.get("chunk_count", "?")
                model = info.get("embedding_model", "?")
                dim = info.get("embedding_dim", "?")
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
            print('  pip install -e ".[converter]"')
            print("\nThis will install support for PDF and DOCX conversion to markdown.")
            sys.exit(1)

        if len(sys.argv) < 3:
            print("❌ Error: Missing file path")
            print(
                "Usage: python sear.py convert <file.pdf|file.docx> [--output-dir DIR] [--no-normalize] [--force-ocr] [--lang heb|eng|heb+eng]"
            )
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
            if sys.argv[i] == "--output-dir" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--no-normalize":
                normalize = False
                i += 1
            elif sys.argv[i] == "--force-ocr":
                force_ocr = True
                i += 1
            elif sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
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
        if ext not in [".pdf", ".docx"]:
            print(f"❌ Error: Unsupported format: {ext}")
            print("Supported formats: .pdf" + (", .docx" if DOCX_AVAILABLE else ""))
            sys.exit(1)

        try:
            print(f"📄 Converting {os.path.basename(filepath)}...")
            kwargs = {"output_dir": output_dir, "force_ocr": force_ocr, "normalize": normalize}
            if lang is not None:
                kwargs["lang"] = lang

            success, elapsed, metadata = convert_document(filepath, **kwargs)

            if success:
                output_file = os.path.join(
                    output_dir, os.path.splitext(os.path.basename(filepath))[0] + ".md"
                )
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
        print("\n" + "=" * 80)
        print("GPU INFORMATION")
        print("=" * 80)

        if is_gpu_available():
            gpu_info = get_gpu_info()
            print("✅ GPU acceleration available")
            print(f"   Number of GPUs: {gpu_info.get('num_gpus', 'Unknown')}")
            print("\nGPU support will be automatically used for faster indexing and search.")
            print("Use --no-gpu flag to disable GPU acceleration if needed.")
        else:
            print("❌ GPU acceleration not available")
            print("\nTo enable GPU support:")
            print("  1. Install CUDA toolkit (https://developer.nvidia.com/cuda-downloads)")
            print("  2. Uninstall faiss-cpu: pip uninstall faiss-cpu")
            print("  3. Install faiss-gpu: pip install faiss-gpu")
            print("\nSEAR will continue to work with CPU (slower but functional).")

        print("=" * 80)

    else:
        print(f"❌ Unknown command: {cmd}")
        print("Valid commands: index, search, extract, convert, list, delete, gpu-info")
        sys.exit(1)


if __name__ == "__main__":
    main()
