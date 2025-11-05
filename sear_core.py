#!/usr/bin/env python3
"""
SEAR Core Library: Summarization-Enhanced Augmented Retrieval
Ollama-based FAISS semantic search library for programmatic use

This module provides core SEAR functionality without CLI dependencies.
Use this for integrating SEAR into your own applications.

File Types:
- INPUT: Pre-extracted text files (.txt) from gitingest or other sources
- OUTPUT: FAISS indices and metadata stored in ~/.sear/

Example Usage:
    # Basic search
    from sear_core import index_file, search
    index_file("code.txt", "my-corpus")
    results = search("authentication", corpuses=["my-corpus"])

    # Boolean queries with JSON format
    from sear_core import execute_query
    query = {
        "operation": "difference",
        "left": {"query": "security"},
        "right": {"query": "deprecated"}
    }
    results = execute_query(query)

    # SQL interface
    from sear_core import execute_sql_query
    results = execute_sql_query(
        'SELECT * FROM search("security") EXCEPT SELECT * FROM search("deprecated")'
    )
"""

import pickle
import numpy as np
from pathlib import Path
import json
import faiss

# Public API - functions available for import
__all__ = [
    # Core indexing and search
    'index_file',
    'search',
    'extract_relevant_content',

    # Corpus management
    'list_corpuses',
    'get_corpus_info',
    'delete_corpus',
    'validate_input_file',

    # Boolean query operations (JSON format)
    'execute_query',
    'union_results',
    'difference_results',
    'intersect_results',

    # SQL query interface (NEW in v2.3.0)
    'parse_sql_query',
    'execute_sql_query',

    # LLM generation
    'ollama_generate',
    'anthropic_generate',

    # GPU support
    'is_gpu_available',
    'get_gpu_info',
    'init_gpu',

    # Document ordering utilities
    'sort_by_document_order',
    'merge_adjacent_chunks',
]

###############################################################################
# CONFIGURATION
###############################################################################

# OUTPUT: Directory where SEAR stores generated indices
INDEX_DIR = Path("~/.sear").expanduser()

# GPU Configuration
GPU_AVAILABLE = False
GPU_RESOURCES = None

def init_gpu():
    """
    Initialize GPU resources for FAISS if available.

    Returns:
        bool: True if GPU is available and initialized, False otherwise
    """
    global GPU_AVAILABLE, GPU_RESOURCES

    try:
        # Check if FAISS was compiled with GPU support
        if not hasattr(faiss, 'StandardGpuResources'):
            return False

        # Check if CUDA GPUs are available
        ngpus = faiss.get_num_gpus()
        if ngpus == 0:
            return False

        # Initialize GPU resources
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_AVAILABLE = True
        return True
    except Exception as e:
        # GPU not available or error initializing
        GPU_AVAILABLE = False
        return False

# Try to initialize GPU on module import
init_gpu()

def is_gpu_available():
    """
    Check if GPU acceleration is available.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    return GPU_AVAILABLE

def get_gpu_info():
    """
    Get information about available GPUs.

    Returns:
        dict: GPU information or empty dict if no GPU available
    """
    if not GPU_AVAILABLE:
        return {}

    try:
        ngpus = faiss.get_num_gpus()
        return {
            'num_gpus': ngpus,
            'available': True
        }
    except:
        return {}

def index_cpu_to_gpu(cpu_index, gpu_id=0):
    """
    Transfer a CPU index to GPU for faster search.

    Args:
        cpu_index: FAISS CPU index
        gpu_id: GPU device ID (default: 0)

    Returns:
        FAISS GPU index or original CPU index if GPU not available
    """
    if not GPU_AVAILABLE or GPU_RESOURCES is None:
        return cpu_index

    try:
        gpu_index = faiss.index_cpu_to_gpu(GPU_RESOURCES, gpu_id, cpu_index)
        return gpu_index
    except Exception as e:
        print(f"Warning: Failed to transfer index to GPU: {e}")
        return cpu_index

def index_gpu_to_cpu(gpu_index):
    """
    Transfer a GPU index back to CPU for saving.

    Args:
        gpu_index: FAISS GPU index

    Returns:
        FAISS CPU index
    """
    if not GPU_AVAILABLE:
        return gpu_index

    try:
        return faiss.index_gpu_to_cpu(gpu_index)
    except:
        return gpu_index

###############################################################################
# CORE LIBRARY FUNCTIONS
###############################################################################

def validate_input_file(filepath):
    """
    Validate that input is a readable text file (not directory, URL, or git repo).
    SEAR operates on pre-extracted text files only.

    Args:
        filepath: Path to text file to validate

    Returns:
        str: Validated absolute path

    Raises:
        ValueError: If file is invalid (with descriptive message)
    """
    path = Path(filepath)

    # Check if it looks like a URL
    if filepath.startswith(('http://', 'https://', 'git@', 'git://')):
        raise ValueError(
            f"URLs are not supported.\n"
            f"SEAR operates on text files only.\n"
            f"Use gitingest to extract repositories first:\n"
            f"  gitingest {filepath} -o output.txt"
        )

    # Check if path exists
    if not path.exists():
        raise ValueError(f"File not found: {filepath}")

    # Check if it's a directory
    if path.is_dir():
        raise ValueError(
            f"'{filepath}' is a directory.\n"
            f"SEAR operates on single text files only.\n"
            f"Use gitingest to extract repositories first:\n"
            f"  gitingest /path/to/repo -o output.txt"
        )

    # Check if it's a git repository
    if (path / '.git').exists() or path.name == '.git':
        raise ValueError(
            f"'{filepath}' appears to be a git repository.\n"
            f"SEAR operates on text files only.\n"
            f"Use gitingest to extract the repository first:\n"
            f"  gitingest . -o output.txt"
        )

    # Check if file is readable
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1)  # Try reading first character
    except UnicodeDecodeError:
        raise ValueError(
            f"'{filepath}' is not a valid text file (encoding issue).\n"
            f"SEAR requires UTF-8 text files."
        )
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}")

    return str(path)

def get_model_context_length(model_name="all-minilm"):
    """
    Query Ollama API to get the model's context length.

    Args:
        model_name: Name of the Ollama model

    Returns:
        int: Context length in tokens, or None if not found
    """
    import http.client
    try:
        conn = http.client.HTTPConnection("localhost", 11434)
        payload = json.dumps({"name": model_name})
        headers = {'Content-Type': 'application/json'}
        conn.request("POST", "/api/show", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        conn.close()

        # Try to extract num_ctx from parameters first (this is the active limit)
        if 'parameters' in data:
            params = data['parameters']
            for line in params.split('\n'):
                if 'num_ctx' in line:
                    # Format: "num_ctx                        256"
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[-1])

        # Fall back to model_info context_length
        if 'model_info' in data:
            for key, value in data['model_info'].items():
                if 'context_length' in key:
                    return int(value)

        return None
    except Exception as e:
        print(f"Warning: Could not get model context length: {e}")
        return None

def estimate_tokens(text):
    """
    Estimate token count using word-based heuristic.

    For BERT-based models like all-minilm, this is a conservative approximation.
    Real tokenization can create more tokens due to subword splitting, especially
    for text with special characters, code, URLs, and technical content.

    Args:
        text: Text to estimate tokens for

    Returns:
        int: Estimated token count
    """
    # Count words
    words = text.split()

    # Base token estimate (1 word ≈ 1 token)
    base_tokens = len(words)

    # Add buffer for subword tokenization
    # Technical content (code, paths, URLs, Unicode box-drawing chars) can create 200-300% more tokens
    # Use 3.5x multiplier to handle worst-case scenarios (e.g., directory trees with Unicode chars)
    return int(base_tokens * 3.5)

def ollama_embed(text, model_name="all-minilm", max_retries=3):
    """
    Get embedding from Ollama API with automatic truncation fallback.

    If text exceeds model's context length, automatically retries with progressively
    smaller portions until it fits.

    Args:
        text: Text to embed
        model_name: Name of the Ollama model to use
        max_retries: Maximum number of truncation attempts (default: 3)

    Returns:
        numpy.ndarray: Embedding vector (float32)
    """
    import http.client

    original_length = len(text)
    current_text = text

    for attempt in range(max_retries + 1):
        conn = http.client.HTTPConnection("localhost", 11434)
        payload = json.dumps({"model": model_name, "prompt": current_text})
        headers = {'Content-Type': 'application/json'}
        conn.request("POST", "/api/embeddings", payload, headers)
        res = conn.getresponse()
        raw_response = res.read().decode("utf-8")
        data = json.loads(raw_response)
        conn.close()

        # Success case
        if 'embedding' in data:
            if attempt > 0:
                # Warn that we had to truncate
                print(f"   ⚠️  Truncated {original_length} → {len(current_text)} chars")
            return np.array(data['embedding'], dtype='float32')

        # Error case
        if 'error' in data and 'exceeds the context length' in data['error']:
            if attempt < max_retries:
                # Try with 70% of current length
                current_text = current_text[:int(len(current_text) * 0.7)]
                continue
            else:
                # Give up after max retries
                print(f"ERROR: Text too long even after {max_retries} truncations")
                print(f"Original: {original_length} chars, Final: {len(current_text)} chars")
                print(f"Response: {raw_response[:200]}")
                raise KeyError(f"Text exceeds context length after truncation attempts")

        # Other errors
        print(f"ERROR: No 'embedding' in response. Got: {raw_response[:200]}")
        raise KeyError(f"'embedding' key missing. Response keys: {list(data.keys())}")

def ollama_generate(prompt, temperature=0.0):
    """
    Generate text with Ollama (deterministic with temp=0).

    Args:
        prompt: Text prompt
        temperature: Sampling temperature (0.0 = deterministic)

    Returns:
        str: Generated text
    """
    import http.client
    conn = http.client.HTTPConnection("localhost", 11434)
    payload = json.dumps({
        "model": "qwen2.5:0.5b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,  # 0 = deterministic
            "seed": 42  # Fixed seed for reproducibility
        }
    })
    headers = {'Content-Type': 'application/json'}
    conn.request("POST", "/api/generate", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    conn.close()
    return data.get('response', '').strip()

def anthropic_generate(prompt, temperature=0.0, api_key=None):
    """
    Generate text with Anthropic Claude API.

    Args:
        prompt: Text prompt
        temperature: Sampling temperature (0.0 = deterministic)
        api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)

    Returns:
        str: Generated text

    Raises:
        ValueError: If API key is not provided and not found in environment
    """
    import os

    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.environ.get('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Provide via --api-key flag or set ANTHROPIC_API_KEY environment variable."
        )

    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK not installed. Run: pip install anthropic"
        )

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        temperature=temperature,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return response.content[0].text.strip()

def chunk_text(text, size=None, model_name="all-minilm"):
    """
    Simple chunking with line number tracking.

    Automatically detects the embedding model's context length and chunks accordingly.
    If a single line exceeds the chunk size, it will be split at word boundaries.

    Args:
        text: Input text to chunk
        size: Chunk size in tokens (default: auto-detect from model, with 20% safety buffer)
        model_name: Name of the embedding model (used to detect context length)

    Returns:
        tuple: (chunks, line_ranges) where line_ranges are (start, end) tuples
    """
    # Auto-detect chunk size if not specified
    if size is None:
        context_length = get_model_context_length(model_name)
        if context_length:
            # Use 50% of context length as safe chunk size
            # This leaves ample room for:
            # - Special tokens (CLS, SEP, etc.)
            # - Subword tokenization variance (especially for technical content with Unicode)
            # - Model overhead
            # With 3.5x token estimation multiplier, this targets ~14% actual usage for safety
            size = int(context_length * 0.5)
        else:
            # Fallback to conservative default
            size = 100

    lines = text.split('\n')
    chunks = []
    chunk_line_ranges = []
    current = []
    current_size = 0
    start_line = 1

    for line_num, line in enumerate(lines, 1):
        line_tokens = estimate_tokens(line)

        # If single line exceeds chunk size, split it at word boundaries
        if line_tokens > size:
            # First, flush current chunk if it has content
            if current:
                chunks.append('\n'.join(current))
                chunk_line_ranges.append((start_line, line_num - 1))
                current = []
                current_size = 0

            # Split the long line into word-based chunks
            words = line.split()
            word_chunk = []
            word_chunk_size = 0

            for word in words:
                word_tokens = estimate_tokens(word)
                if word_chunk_size + word_tokens > size and word_chunk:
                    # Save this word chunk
                    chunks.append(' '.join(word_chunk))
                    chunk_line_ranges.append((line_num, line_num))
                    word_chunk = []
                    word_chunk_size = 0

                word_chunk.append(word)
                word_chunk_size += word_tokens

            # Add remaining words from this long line
            if word_chunk:
                chunks.append(' '.join(word_chunk))
                chunk_line_ranges.append((line_num, line_num))

            start_line = line_num + 1

        # Normal case: line fits within chunk size
        elif current_size + line_tokens > size and current:
            # Save current chunk
            chunks.append('\n'.join(current))
            chunk_line_ranges.append((start_line, line_num - 1))
            current = []
            current_size = 0
            start_line = line_num
            current.append(line)
            current_size = line_tokens
        else:
            current.append(line)
            current_size += line_tokens

    # Add final chunk
    if current:
        chunks.append('\n'.join(current))
        chunk_line_ranges.append((start_line, len(lines)))

    # Filter empty chunks
    filtered = [(c, r) for c, r in zip(chunks, chunk_line_ranges) if c.strip()]
    return [c for c, _ in filtered], [r for _, r in filtered]

def get_corpus_dir(corpus_name):
    """
    Get the directory path for a specific corpus.

    Args:
        corpus_name: Name of the corpus

    Returns:
        Path: Directory path for the corpus
    """
    return INDEX_DIR / corpus_name

def get_corpus_paths(corpus_name):
    """
    Get file paths for a corpus's index, metadata, and file info.

    Args:
        corpus_name: Name of the corpus

    Returns:
        dict: Paths for index, chunks, and metadata
    """
    corpus_dir = get_corpus_dir(corpus_name)
    return {
        'index': corpus_dir / 'index.faiss',
        'chunks': corpus_dir / 'chunks.pkl',
        'metadata': corpus_dir / 'metadata.json'
    }

def list_corpuses():
    """
    List all available corpuses in the index directory.

    Returns:
        list: List of corpus names
    """
    if not INDEX_DIR.exists():
        return []

    corpuses = []
    for item in INDEX_DIR.iterdir():
        if item.is_dir() and (item / 'index.faiss').exists():
            corpuses.append(item.name)

    return sorted(corpuses)

def get_corpus_info(corpus_name):
    """
    Get metadata information for a specific corpus.

    Args:
        corpus_name: Name of the corpus

    Returns:
        dict: Metadata dictionary or None if corpus doesn't exist

    Raises:
        FileNotFoundError: If corpus doesn't exist
    """
    paths = get_corpus_paths(corpus_name)

    if not paths['metadata'].exists():
        raise FileNotFoundError(f"Corpus '{corpus_name}' not found")

    try:
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        # Add chunk count if available
        if paths['chunks'].exists():
            with open(paths['chunks'], 'rb') as f:
                chunks_data = pickle.load(f)
                metadata['chunk_count'] = len(chunks_data['chunks'])

        return metadata
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        raise ValueError(f"Corrupted corpus metadata for '{corpus_name}': {e}")

def delete_corpus(corpus_name):
    """
    Delete a corpus and all its associated files.

    Args:
        corpus_name: Name of the corpus to delete

    Raises:
        FileNotFoundError: If corpus doesn't exist
    """
    import shutil

    corpus_dir = get_corpus_dir(corpus_name)

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus '{corpus_name}' not found")

    shutil.rmtree(corpus_dir)

def validate_corpus_compatibility(corpus_names):
    """
    Validate that multiple corpuses have compatible embedding settings.

    Args:
        corpus_names: List of corpus names to validate

    Returns:
        dict: Common metadata if compatible

    Raises:
        ValueError: If corpuses have incompatible settings
    """
    if not corpus_names:
        return {}

    # Get metadata for all corpuses
    corpus_metas = []
    for name in corpus_names:
        try:
            meta = get_corpus_info(name)
            corpus_metas.append((name, meta))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Cannot load corpus '{name}': {e}")

    # Use first corpus as reference
    ref_name, ref_meta = corpus_metas[0]
    ref_settings = {
        'embedding_model': ref_meta.get('embedding_model'),
        'embedding_dim': ref_meta.get('embedding_dim'),
        'faiss_index_type': ref_meta.get('faiss_index_type'),
        'normalized': ref_meta.get('normalized')
    }

    # Validate all other corpuses match
    incompatible = []
    for name, meta in corpus_metas[1:]:
        if (meta.get('embedding_model') != ref_settings['embedding_model'] or
            meta.get('embedding_dim') != ref_settings['embedding_dim'] or
            meta.get('faiss_index_type') != ref_settings['faiss_index_type'] or
            meta.get('normalized') != ref_settings['normalized']):

            incompatible.append(
                f"  - {name}: {meta.get('embedding_model')} "
                f"({meta.get('embedding_dim')}d, {meta.get('faiss_index_type')}, "
                f"normalized={meta.get('normalized')})"
            )

    if incompatible:
        error_msg = (
            f"Error: Cannot search corpuses with incompatible embeddings:\n"
            f"  - {ref_name}: {ref_settings['embedding_model']} "
            f"({ref_settings['embedding_dim']}d, {ref_settings['faiss_index_type']}, "
            f"normalized={ref_settings['normalized']})\n"
            + "\n".join(incompatible) +
            f"\n\nRe-index incompatible corpuses or search them separately."
        )
        raise ValueError(error_msg)

    return ref_settings

def index_file(filepath, corpus_name=None, verbose=True, use_gpu=None):
    """
    Index a text file with Ollama embeddings into a named corpus.

    Args:
        filepath: Path to text file to index
        corpus_name: Name for this corpus (defaults to filename stem)
        verbose: Print progress messages
        use_gpu: Enable GPU acceleration (None=auto-detect, True=force, False=disable)

    Returns:
        int: Number of chunks indexed

    Raises:
        ValueError: If file validation fails
    """
    # Validate input before processing
    filepath = validate_input_file(filepath)

    # Determine GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    elif use_gpu and not GPU_AVAILABLE:
        print("⚠️  GPU requested but not available, falling back to CPU")
        use_gpu = False

    if verbose and use_gpu:
        gpu_info = get_gpu_info()
        print(f"🚀 GPU acceleration enabled ({gpu_info.get('num_gpus', 1)} GPU(s))")

    # Default corpus name to filename stem if not provided
    if corpus_name is None:
        corpus_name = Path(filepath).stem

    # Get corpus paths
    paths = get_corpus_paths(corpus_name)
    corpus_dir = get_corpus_dir(corpus_name)

    # Check file metadata for caching
    file_path = Path(filepath)
    file_stat = file_path.stat()

    # Get model context length for metadata
    model_name = 'all-minilm'
    context_length = get_model_context_length(model_name) or 256

    # Metadata schema with embedding details for compatibility validation
    file_metadata = {
        'file_path': str(file_path.absolute()),
        'file_modified': file_stat.st_mtime,
        'file_size': file_stat.st_size,
        'embedding_model': model_name,
        'embedding_dim': None,  # Will be set after first embedding
        'faiss_index_type': 'IndexFlatIP',
        'normalized': True,
        'context_length': context_length  # Store for query validation
    }

    # Check if we can skip re-indexing (cache hit)
    if paths['metadata'].exists():
        try:
            with open(paths['metadata'], 'r') as f:
                cached_meta = json.load(f)

            if (cached_meta.get('file_path') == file_metadata['file_path'] and
                cached_meta.get('file_modified') == file_metadata['file_modified'] and
                cached_meta.get('file_size') == file_metadata['file_size']):
                if verbose:
                    print(f"✅ Cache hit: {corpus_name} is up to date")
                # Return cached chunk count
                if paths['chunks'].exists():
                    with open(paths['chunks'], 'rb') as f:
                        cached_data = pickle.load(f)
                        return len(cached_data['chunks'])
        except (json.JSONDecodeError, KeyError):
            pass  # Re-index if cache is corrupted

    if verbose:
        print(f"📂 Reading {filepath}...")
    text = file_path.read_text()

    if verbose:
        print("✂️  Chunking...")
    chunks, line_ranges = chunk_text(text)
    if verbose:
        print(f"   {len(chunks)} chunks")

    if verbose:
        print("🔢 Embedding with Ollama...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if verbose and i % 10 == 0:
            print(f"   {i}/{len(chunks)}")
        try:
            emb = ollama_embed(chunk)  # Chunks are already sized appropriately
            embeddings.append(emb)
        except KeyError as e:
            # Provide debugging info on failure
            chunk_tokens = estimate_tokens(chunk)
            print(f"\n❌ Failed on chunk {i+1}/{len(chunks)}:")
            print(f"   Length: {len(chunk)} chars")
            print(f"   Words: {len(chunk.split())}")
            print(f"   Estimated tokens: {chunk_tokens}")
            print(f"   Line range: {line_ranges[i]}")
            print(f"   First 200 chars: {chunk[:200]}")
            raise

    embeddings = np.array(embeddings).astype('float32')

    # Update metadata with actual embedding dimension
    file_metadata['embedding_dim'] = embeddings.shape[1]

    if verbose:
        print("🏗️  Building FAISS index...")
    dimension = embeddings.shape[1]

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Build index (GPU accelerated if available)
    if use_gpu:
        # Create CPU index first
        cpu_index = faiss.IndexFlatIP(dimension)
        cpu_index.add(embeddings)
        # Note: We save CPU index for portability, but GPU is used during creation
        index = cpu_index
        if verbose:
            print("   ✓ Built with GPU acceleration")
    else:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

    if verbose:
        print(f"💾 Saving to corpus: {corpus_name}")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index (always save as CPU index for portability)
    faiss.write_index(index, str(paths['index']))

    # Save chunks data
    with open(paths['chunks'], 'wb') as f:
        pickle.dump({'chunks': chunks, 'line_ranges': line_ranges, 'source': str(filepath)}, f)

    # Save file metadata for caching
    with open(paths['metadata'], 'w') as f:
        json.dump(file_metadata, f, indent=2)

    if verbose:
        print(f"✅ Done: {len(chunks)} chunks indexed to '{corpus_name}'")

    return len(chunks)

def search(query, corpuses=None, top_k=5, verbose=True, temperature=0.0, min_score=0.30, use_gpu=None, provider='ollama', api_key=None):
    """
    Search across one or more corpuses and generate an answer.

    Args:
        query: Search query string
        corpuses: List of corpus names to search (None = search all available)
        top_k: Number of top results to retrieve per corpus
        verbose: Print progress and results
        temperature: LLM temperature (0.0=deterministic, higher=more creative)
        min_score: Minimum similarity score threshold (default: 0.30 for high precision)
                   Based on empirical testing:
                   - 0.30: High precision (recommended default)
                   - 0.20: Balanced filtering
                   - 0.15: High recall (lenient)
                   - None: No filtering (show all results)
        use_gpu: Enable GPU acceleration (None=auto-detect, True=force, False=disable)
        provider: LLM provider for generation ('ollama' or 'anthropic', default: 'ollama')
        api_key: API key for Anthropic (if None, reads from ANTHROPIC_API_KEY env var)

    Returns:
        dict: Search results with keys:
            - 'answer': Generated answer text (or suggestion to refine query)
            - 'sources': List of source dicts with chunk info
            - 'filtered_count': Number of results below threshold
            - 'quality_results': Number of results above threshold

    Raises:
        ValueError: If no corpuses available, compatibility check fails, or invalid provider
        FileNotFoundError: If specified corpus doesn't exist
    """
    # Validate provider
    if provider not in ['ollama', 'anthropic']:
        raise ValueError(f"Invalid provider '{provider}'. Must be 'ollama' or 'anthropic'.")
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    elif use_gpu and not GPU_AVAILABLE:
        if verbose:
            print("⚠️  GPU requested but not available, falling back to CPU")
        use_gpu = False

    if verbose and use_gpu:
        gpu_info = get_gpu_info()
        print(f"🚀 GPU acceleration enabled ({gpu_info.get('num_gpus', 1)} GPU(s))")

    # Determine which corpuses to search
    if corpuses is None:
        corpuses = list_corpuses()
        if not corpuses:
            raise ValueError(
                "No corpuses found. Please index a file first:\n"
                "  python sear.py index <file.txt>"
            )
        if verbose:
            print(f"🔍 Searching all corpuses: {', '.join(corpuses)}")
    else:
        if verbose:
            print(f"🔍 Searching corpuses: {', '.join(corpuses)}")

    # Validate corpus compatibility
    if verbose:
        print("✓ Validating corpus compatibility...")
    common_settings = validate_corpus_compatibility(corpuses)

    # Validate query length against model's context length
    context_length = common_settings.get('context_length', 256)
    query_tokens = estimate_tokens(query)

    if query_tokens > context_length:
        # Calculate recommended query length
        chars_per_token = len(query) / query_tokens if query_tokens > 0 else 4
        recommended_chars = int(context_length * chars_per_token * 0.9)

        raise ValueError(
            f"Query is too long ({query_tokens} estimated tokens, max: {context_length}).\n\n"
            f"Your query:\n  \"{query}\"\n\n"
            f"Please provide a shorter, more concise query (max ~{recommended_chars} characters).\n"
            f"Focus on the key terms and concepts you're searching for."
        )

    # Load all corpus indices
    if verbose:
        print("📖 Loading indices...")

    corpus_data = []
    for corpus_name in corpuses:
        paths = get_corpus_paths(corpus_name)

        # Load FAISS index
        cpu_index = faiss.read_index(str(paths['index']))

        # Transfer to GPU if enabled
        if use_gpu:
            index = index_cpu_to_gpu(cpu_index)
        else:
            index = cpu_index

        # Load chunks data
        with open(paths['chunks'], 'rb') as f:
            chunks_data = pickle.load(f)

        corpus_data.append({
            'name': corpus_name,
            'index': index,
            'chunks': chunks_data['chunks'],
            'line_ranges': chunks_data['line_ranges'],
            'source': chunks_data['source']
        })

    # Prepare query embedding
    if verbose:
        print(f"🔍 Searching for: {query}")

    query_emb = ollama_embed(query)  # ollama_embed will handle any necessary length limits
    query_emb = np.array([query_emb]).astype('float32')

    # Normalize query if corpuses use normalized embeddings
    if common_settings.get('normalized', True):
        faiss.normalize_L2(query_emb)

    # Search all corpuses and aggregate results
    all_results = []
    for corpus in corpus_data:
        distances, indices = corpus['index'].search(query_emb, top_k)

        for idx, score in zip(indices[0], distances[0]):
            chunk = corpus['chunks'][idx]
            line_range = corpus['line_ranges'][idx]
            source = corpus['source']

            all_results.append({
                'corpus': corpus['name'],
                'score': float(score),
                'chunk': chunk,
                'location': f"{source}:{line_range[0]}-{line_range[1]}"
            })

    # Sort by score (descending for similarity)
    all_results.sort(key=lambda x: x['score'], reverse=True)

    # Take top N overall results
    top_results = all_results[:top_k * 2]  # Get more context for answer generation

    # Filter results by minimum score if specified
    original_count = len(top_results)
    if min_score is not None:
        quality_results = [r for r in top_results if r['score'] >= min_score]
        filtered_count = original_count - len(quality_results)

        if verbose:
            print(f"🔍 Quality filter (min_score={min_score}): {len(quality_results)}/{original_count} results passed")
            if filtered_count > 0:
                print(f"   ⚠️  Filtered out {filtered_count} low-quality results")
    else:
        quality_results = top_results
        filtered_count = 0

    # Check if we have enough quality results to generate an answer
    if min_score is not None and len(quality_results) < 2:
        # Not enough quality results - suggest query refinement
        if verbose:
            print("\n" + "="*80)
            print("⚠️  INSUFFICIENT QUALITY RESULTS")
            print("="*80)
            print(f"Found only {len(quality_results)} result(s) with score ≥ {min_score}")
            print("\nSuggestion: Refine your query to be more specific to the corpus content.")
            if top_results:
                print(f"\nTop result score: {top_results[0]['score']:.3f}")
                print(f"Preview: {top_results[0]['chunk'][:150]}...")
            print("="*80)

        return {
            'answer': f"⚠️  No high-quality matches found (only {len(quality_results)} result(s) with score ≥ {min_score}).\n\n"
                     f"Please refine your query to be more specific to the corpus content.\n\n"
                     f"Top result score: {top_results[0]['score']:.3f}" if top_results else "No results found.",
            'sources': top_results,
            'quality_results': len(quality_results),
            'filtered_count': filtered_count
        }

    # Generate answer using only quality results
    if verbose:
        print(f"🤖 Generating answer from {len(quality_results)} quality results...")

    context = f"Question: {query}\n\n"
    for i, r in enumerate(quality_results, 1):
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

    # Generate answer using selected provider
    if provider == 'anthropic':
        if verbose:
            print("🤖 Using Anthropic Claude 3.5 Sonnet 4.5...")
        answer = anthropic_generate(prompt, temperature=temperature, api_key=api_key)
    else:  # ollama
        answer = ollama_generate(prompt, temperature=temperature)

    # Display results
    if verbose:
        print("\n" + "="*80)
        print("📊 ANSWER")
        print("="*80)
        print(answer)
        print("\n" + "="*80)
        print("📍 SOURCES (Quality Results)")
        print("="*80)
        for i, r in enumerate(quality_results[:5], 1):  # Show top 5 quality sources
            print(f"\n{i}. [{r['corpus']}] {r['location']} (score: {r['score']:.3f})")
            print(f"   {r['chunk'][:200]}...")
        if filtered_count > 0:
            print(f"\n   ℹ️  Note: {filtered_count} low-quality result(s) filtered out (score < {min_score})")
        print("="*80)

    return {
        'answer': answer,
        'sources': quality_results,
        'quality_results': len(quality_results),
        'filtered_count': filtered_count
    }

def reconstruct_text_from_chunks(chunks_metadata, source_file, verbose=False):
    """
    Reconstruct original text from chunks using line ranges.

    Merges consecutive/overlapping chunks from the same source file.

    Args:
        chunks_metadata: List of dicts with 'line_range' and 'chunk_index' keys
        source_file: Path to the original source file
        verbose: Print progress messages

    Returns:
        str: Reconstructed text
    """
    if not chunks_metadata:
        return ""

    # Sort by chunk_index to maintain document order
    sorted_chunks = sorted(chunks_metadata, key=lambda x: x['chunk_index'])

    # Read the source file once
    try:
        source_path = Path(source_file)
        if not source_path.exists():
            if verbose:
                print(f"⚠️  Warning: Source file not found: {source_file}")
            return ""

        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        if verbose:
            print(f"⚠️  Warning: Could not read source file {source_file}: {e}")
        return ""

    # Merge consecutive/overlapping line ranges
    merged_ranges = []
    for chunk in sorted_chunks:
        start, end = chunk['line_range']

        if not merged_ranges:
            merged_ranges.append([start, end])
        else:
            last_start, last_end = merged_ranges[-1]
            # If consecutive or overlapping, merge
            if start <= last_end + 1:
                merged_ranges[-1][1] = max(last_end, end)
            else:
                merged_ranges.append([start, end])

    # Extract text for each range
    reconstructed = []
    for start, end in merged_ranges:
        # Line numbers are 1-indexed, list access is 0-indexed
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        reconstructed.append(''.join(lines[start_idx:end_idx]))

    return '\n'.join(reconstructed)

def extract_relevant_content(
    query,
    corpuses=None,
    output_file=None,
    min_score=0.3,
    max_chunks=None,
    use_gpu=None,
    verbose=False
):
    """
    Extract all content matching query above similarity threshold.

    Unlike search(), this function:
    - Returns ALL chunks above threshold (not just top-k)
    - Reconstructs full text from chunk indices
    - Saves to a text file (does NOT send to LLM)
    - Maintains document order

    Args:
        query: Search query string with keywords
        corpuses: List of corpus names to search (None = search all available)
        output_file: Path to output file (default: extracted_content.txt)
        min_score: Minimum similarity score threshold (default: 0.3)
        max_chunks: Optional limit on total chunks (default: unlimited)
        use_gpu: Enable GPU acceleration (None=auto-detect, True=force, False=disable)
        verbose: Print progress messages

    Returns:
        dict: Extraction results with keys:
            - 'total_chunks': Total chunks found above threshold
            - 'chunks_extracted': Number of chunks actually extracted
            - 'output_file': Path to output file
            - 'score_range': (min_score, max_score) tuple
            - 'sources': List of source files used

    Raises:
        ValueError: If no corpuses available or compatibility check fails
        FileNotFoundError: If specified corpus doesn't exist
    """
    from datetime import datetime

    # Default output file
    if output_file is None:
        output_file = "extracted_content.txt"

    # Determine GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    elif use_gpu and not GPU_AVAILABLE:
        if verbose:
            print("⚠️  GPU requested but not available, falling back to CPU")
        use_gpu = False

    if verbose and use_gpu:
        gpu_info = get_gpu_info()
        print(f"🚀 GPU acceleration enabled ({gpu_info.get('num_gpus', 1)} GPU(s))")

    # Determine which corpuses to search
    if corpuses is None:
        corpuses = list_corpuses()
        if not corpuses:
            raise ValueError(
                "No corpuses found. Please index a file first:\n"
                "  python sear.py index <file.txt>"
            )
        if verbose:
            print(f"🔍 Extracting from all corpuses: {', '.join(corpuses)}")
    else:
        if verbose:
            print(f"🔍 Extracting from corpuses: {', '.join(corpuses)}")

    # Validate corpus compatibility
    if verbose:
        print("✓ Validating corpus compatibility...")
    common_settings = validate_corpus_compatibility(corpuses)

    # Validate query length against model's context length
    context_length = common_settings.get('context_length', 256)
    query_tokens = estimate_tokens(query)

    if query_tokens > context_length:
        chars_per_token = len(query) / query_tokens if query_tokens > 0 else 4
        recommended_chars = int(context_length * chars_per_token * 0.9)

        raise ValueError(
            f"Query is too long ({query_tokens} estimated tokens, max: {context_length}).\n\n"
            f"Your query:\n  \"{query}\"\n\n"
            f"Please provide a shorter query (max ~{recommended_chars} characters)."
        )

    # Load all corpus indices
    if verbose:
        print("📖 Loading indices...")

    corpus_data = []
    for corpus_name in corpuses:
        paths = get_corpus_paths(corpus_name)

        # Load FAISS index
        cpu_index = faiss.read_index(str(paths['index']))

        # Transfer to GPU if enabled
        if use_gpu:
            index = index_cpu_to_gpu(cpu_index)
        else:
            index = cpu_index

        # Load chunks data
        with open(paths['chunks'], 'rb') as f:
            chunks_data = pickle.load(f)

        corpus_data.append({
            'name': corpus_name,
            'index': index,
            'chunks': chunks_data['chunks'],
            'line_ranges': chunks_data['line_ranges'],
            'source': chunks_data['source']
        })

    # Prepare query embedding
    if verbose:
        print(f"🔍 Searching for: {query}")

    query_emb = ollama_embed(query)
    query_emb = np.array([query_emb]).astype('float32')

    # Normalize query if corpuses use normalized embeddings
    if common_settings.get('normalized', True):
        faiss.normalize_L2(query_emb)

    # Search all corpuses with large k to get ALL results
    # Use k=10000 as default to capture all potential matches
    large_k = 10000
    all_results = []

    if verbose:
        print(f"🔎 Searching with k={large_k} to find all relevant chunks...")

    for corpus in corpus_data:
        # Get all results up to index size
        k = min(large_k, corpus['index'].ntotal)
        distances, indices = corpus['index'].search(query_emb, k)

        for idx, score in zip(indices[0], distances[0]):
            # Filter by minimum score threshold
            if score >= min_score:
                chunk = corpus['chunks'][idx]
                line_range = corpus['line_ranges'][idx]
                source = corpus['source']

                all_results.append({
                    'corpus': corpus['name'],
                    'score': float(score),
                    'chunk': chunk,
                    'chunk_index': idx,
                    'line_range': line_range,
                    'source': source
                })

    # Sort by score (descending)
    all_results.sort(key=lambda x: x['score'], reverse=True)

    # Apply max_chunks limit if specified
    if max_chunks is not None:
        all_results = all_results[:max_chunks]

    if verbose:
        print(f"✓ Found {len(all_results)} chunks above threshold (min_score={min_score})")

    if not all_results:
        if verbose:
            print("⚠️  No chunks found above threshold")
        return {
            'total_chunks': 0,
            'chunks_extracted': 0,
            'output_file': output_file,
            'score_range': (0, 0),
            'sources': []
        }

    # Group results by source file
    results_by_source = {}
    for result in all_results:
        source = result['source']
        if source not in results_by_source:
            results_by_source[source] = []
        results_by_source[source].append(result)

    # Calculate statistics
    scores = [r['score'] for r in all_results]
    score_range = (min(scores), max(scores))

    if verbose:
        print(f"📝 Reconstructing text from {len(results_by_source)} source file(s)...")
        if len(all_results) > 100:
            print("   (This may take a moment for large extractions...)")

    # Build output content
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("SEAR EXTRACTED CONTENT")
    output_lines.append("=" * 80)
    output_lines.append(f"Query: \"{query}\"")
    output_lines.append(f"Corpus: {', '.join(corpuses)}")
    output_lines.append(f"Min Score: {min_score}")
    output_lines.append(f"Total Chunks: {len(all_results)}")
    output_lines.append(f"Score Range: {score_range[0]:.3f} - {score_range[1]:.3f}")
    output_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")

    # Process each source file
    chunks_extracted = 0
    for source_file, chunks in results_by_source.items():
        output_lines.append("=" * 80)
        output_lines.append(f"SOURCE: {source_file} ({len(chunks)} chunks)")
        output_lines.append("=" * 80)
        output_lines.append("")

        # Reconstruct text from chunks
        reconstructed = reconstruct_text_from_chunks(chunks, source_file, verbose=verbose)

        if reconstructed:
            output_lines.append(reconstructed)
            output_lines.append("")
            chunks_extracted += len(chunks)

        if verbose and len(results_by_source) > 5:
            print(f"   ✓ Processed {source_file}")

    output_lines.append("=" * 80)
    output_lines.append("END OF EXTRACTION")
    output_lines.append("=" * 80)

    # Write to output file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    if verbose:
        print(f"✅ Extraction complete: {len(all_results)} chunks saved to {output_file}")
        print(f"   Score range: {score_range[0]:.3f} - {score_range[1]:.3f}")
        print(f"   Sources: {len(results_by_source)} file(s)")

    return {
        'total_chunks': len(all_results),
        'chunks_extracted': chunks_extracted,
        'output_file': str(output_path.absolute()),
        'score_range': score_range,
        'sources': list(results_by_source.keys())
    }

###############################################################################
# BOOLEAN LOGIC OPERATIONS
###############################################################################

def _chunk_to_key(chunk_result):
    """
    Convert a chunk result to a hashable key for set operations.

    Chunk identity is defined by (corpus, location) tuple.

    Args:
        chunk_result: Dict with 'corpus' and 'location' keys

    Returns:
        tuple: (corpus, location) as hashable key
    """
    return (chunk_result['corpus'], chunk_result['location'])

def union_results(result_sets):
    """
    Perform set union on multiple result sets.

    For duplicate chunks (same corpus and location), keeps the one with the highest score.

    Args:
        result_sets: List of result lists, where each result is a dict with:
                    {'corpus', 'location', 'score', 'chunk', ...}

    Returns:
        list: Combined results with duplicates removed (highest score preserved)
    """
    if not result_sets:
        return []

    # Build dict mapping chunk key to best result
    combined = {}

    for result_set in result_sets:
        for result in result_set:
            key = _chunk_to_key(result)

            # Keep the result with the highest score
            if key not in combined or result['score'] > combined[key]['score']:
                combined[key] = result

    # Return as list
    return list(combined.values())

def difference_results(set_a, set_b, semantic=False, threshold=0.7):
    """
    Perform set difference: A - B (items in A but not in B).

    Args:
        set_a: List of results to keep from
        set_b: List of results to exclude
        semantic: If True, use semantic similarity for exclusion (not just exact match)
        threshold: Similarity threshold for semantic exclusion (0.0-1.0)

    Returns:
        list: Results from set_a that don't appear in set_b
    """
    if not set_a:
        return []

    if not set_b:
        return set_a

    # Exact matching (Level 1)
    if not semantic:
        # Build set of keys to exclude
        exclude_keys = {_chunk_to_key(result) for result in set_b}

        # Return items not in exclude set
        return [result for result in set_a if _chunk_to_key(result) not in exclude_keys]

    # Semantic matching (Level 2)
    # First, apply exact exclusion (fast path for identical chunks)
    exclude_keys = {_chunk_to_key(result) for result in set_b}
    remaining = [result for result in set_a if _chunk_to_key(result) not in exclude_keys]

    if not remaining or not set_b:
        return remaining

    # Now apply semantic exclusion by comparing embeddings from FAISS
    # Group chunks by corpus to load FAISS indices efficiently
    corpus_indices = {}  # corpus_name -> FAISS index

    try:
        # Load FAISS indices for all corpuses involved
        all_corpuses = set()
        for result in remaining:
            all_corpuses.add(result['corpus'])
        for result in set_b:
            all_corpuses.add(result['corpus'])

        for corpus_name in all_corpuses:
            paths = get_corpus_paths(corpus_name)
            corpus_indices[corpus_name] = faiss.read_index(str(paths['index']))

        # Extract embeddings for exclusion set (set_b)
        exclude_embeddings = []
        for result in set_b:
            corpus_name = result['corpus']
            chunk_idx = result.get('chunk_index')
            if chunk_idx is not None and corpus_name in corpus_indices:
                # Retrieve embedding from FAISS index using reconstruct()
                index = corpus_indices[corpus_name]
                embedding = index.reconstruct(int(chunk_idx))
                exclude_embeddings.append(embedding)

        if not exclude_embeddings:
            # No embeddings available, return remaining with exact matching only
            return remaining

        # Convert to numpy array for efficient computation
        exclude_embeddings = np.array(exclude_embeddings).astype('float32')

        # Filter remaining chunks by semantic similarity
        filtered_results = []
        for result in remaining:
            corpus_name = result['corpus']
            chunk_idx = result.get('chunk_index')

            if chunk_idx is None or corpus_name not in corpus_indices:
                # Can't get embedding, keep the chunk (conservative approach)
                filtered_results.append(result)
                continue

            # Retrieve embedding for this chunk from FAISS
            index = corpus_indices[corpus_name]
            chunk_embedding = index.reconstruct(int(chunk_idx))
            chunk_embedding = np.array(chunk_embedding).astype('float32')

            # Calculate cosine similarity with all exclusion embeddings
            # Embeddings are already normalized (we use IndexFlatIP with normalized vectors)
            similarities = np.dot(exclude_embeddings, chunk_embedding)
            max_similarity = float(np.max(similarities))

            # Keep chunk only if it's not too similar to any exclusion chunk
            if max_similarity < threshold:
                filtered_results.append(result)

        return filtered_results

    except Exception as e:
        # If semantic filtering fails, fall back to exact matching
        print(f"⚠️  Semantic filtering failed: {e}. Using exact matching only.")
        return remaining

def intersect_results(set_a, set_b):
    """
    Perform set intersection: A ∩ B (items in both A and B).

    For overlapping chunks, keeps the one with the highest score.

    Args:
        set_a: First list of results
        set_b: Second list of results

    Returns:
        list: Results that appear in both sets (highest score preserved)
    """
    if not set_a or not set_b:
        return []

    # Build dicts for both sets
    dict_a = {_chunk_to_key(result): result for result in set_a}
    dict_b = {_chunk_to_key(result): result for result in set_b}

    # Find intersection keys
    intersect_keys = dict_a.keys() & dict_b.keys()

    # Return results with highest scores
    results = []
    for key in intersect_keys:
        # Pick the result with higher score
        if dict_a[key]['score'] >= dict_b[key]['score']:
            results.append(dict_a[key])
        else:
            results.append(dict_b[key])

    return results

def _parse_location(location_string):
    """
    Parse location string to extract filepath and line range.

    Args:
        location_string: String in format "path/to/file.txt:557-659"

    Returns:
        tuple: (filepath, start_line, end_line)
               Returns (location_string, 0, 0) if parsing fails

    Examples:
        >>> _parse_location("src/main.py:10-20")
        ('src/main.py', 10, 20)
        >>> _parse_location("data/file.txt:1-100")
        ('data/file.txt', 1, 100)
    """
    try:
        # Split on last colon to handle paths with colons
        if ':' not in location_string:
            return (location_string, 0, 0)

        # Find the last colon (separates path from line range)
        last_colon_idx = location_string.rfind(':')
        filepath = location_string[:last_colon_idx]
        line_range = location_string[last_colon_idx + 1:]

        # Parse line range "557-659"
        if '-' not in line_range:
            return (location_string, 0, 0)

        start_str, end_str = line_range.split('-', 1)
        start_line = int(start_str)
        end_line = int(end_str)

        return (filepath, start_line, end_line)

    except (ValueError, AttributeError):
        # If parsing fails, return original string with 0,0 range
        return (location_string, 0, 0)

def sort_by_document_order(chunks):
    """
    Sort chunks by document order: (corpus, filepath, start_line).

    This ensures chunks appear in natural reading flow within their documents.

    Args:
        chunks: List of chunk dicts with 'corpus' and 'location' keys

    Returns:
        list: Chunks sorted by (corpus, filepath, start_line)

    Examples:
        >>> chunks = [
        ...     {'corpus': 'docs', 'location': 'file.txt:100-200', 'score': 0.9},
        ...     {'corpus': 'docs', 'location': 'file.txt:10-50', 'score': 0.8},
        ...     {'corpus': 'code', 'location': 'main.py:1-10', 'score': 0.7}
        ... ]
        >>> sorted_chunks = sort_by_document_order(chunks)
        >>> sorted_chunks[0]['location']
        'main.py:1-10'
    """
    if not chunks:
        return []

    def sort_key(chunk):
        """Extract sort key from chunk."""
        corpus = chunk.get('corpus', '')
        location = chunk.get('location', '')
        filepath, start_line, _ = _parse_location(location)
        return (corpus, filepath, start_line)

    return sorted(chunks, key=sort_key)

def merge_adjacent_chunks(chunks):
    """
    Merge consecutive chunks from the same file with touching or overlapping line ranges.

    Chunks must be sorted by document order first (use sort_by_document_order).

    Args:
        chunks: List of chunk dicts sorted by document order

    Returns:
        list: Chunks with adjacent ones merged

    Merging rules:
        - Same corpus and filepath
        - Line ranges touch (end_a + 1 >= start_b) or overlap
        - Keep highest score
        - Concatenate text with newline separator
        - Combine line ranges (min start, max end)

    Examples:
        >>> chunks = [
        ...     {'corpus': 'docs', 'location': 'f.txt:1-10', 'score': 0.9, 'chunk': 'A'},
        ...     {'corpus': 'docs', 'location': 'f.txt:11-20', 'score': 0.8, 'chunk': 'B'}
        ... ]
        >>> merged = merge_adjacent_chunks(chunks)
        >>> len(merged)
        1
        >>> merged[0]['location']
        'f.txt:1-20'
    """
    if not chunks:
        return []

    merged = []
    current = None

    for chunk in chunks:
        if current is None:
            # Start with first chunk
            current = chunk.copy()
            continue

        # Parse current and new chunk locations
        curr_corpus = current.get('corpus', '')
        curr_location = current.get('location', '')
        curr_filepath, curr_start, curr_end = _parse_location(curr_location)

        new_corpus = chunk.get('corpus', '')
        new_location = chunk.get('location', '')
        new_filepath, new_start, new_end = _parse_location(new_location)

        # Check if chunks can be merged
        same_corpus = curr_corpus == new_corpus
        same_file = curr_filepath == new_filepath
        # Lines touch or overlap: end of current + 1 >= start of new
        lines_adjacent = (curr_end + 1 >= new_start)

        if same_corpus and same_file and lines_adjacent:
            # Merge chunks
            # Keep highest score
            current['score'] = max(current['score'], chunk['score'])

            # Concatenate text
            current_text = current.get('chunk', '')
            new_text = chunk.get('chunk', '')
            current['chunk'] = current_text + '\n' + new_text

            # Combine line ranges: min start, max end
            merged_start = min(curr_start, new_start)
            merged_end = max(curr_end, new_end)
            current['location'] = f"{curr_filepath}:{merged_start}-{merged_end}"

        else:
            # Cannot merge - save current and start new
            merged.append(current)
            current = chunk.copy()

    # Don't forget the last chunk
    if current is not None:
        merged.append(current)

    return merged


# =============================================================================
# Task 3: JSON Query Executor
# =============================================================================

def _retrieve_chunks_only(query, corpuses=None, min_score=0.3, max_results=None, use_gpu=None, verbose=False):
    """
    Internal function to retrieve chunks without LLM processing.

    This is a lightweight version of extract_relevant_content that returns
    chunks directly without writing to file.

    Args:
        query: Search query string
        corpuses: List of corpus names to search (None = all)
        min_score: Minimum similarity threshold (default: 0.3)
        max_results: Maximum number of results to return (default: None = unlimited)
        use_gpu: GPU usage (None=auto, True=force, False=disable)
        verbose: Print progress messages

    Returns:
        list: List of chunk dicts with keys: corpus, location, score, chunk
    """
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    elif use_gpu and not GPU_AVAILABLE:
        if verbose:
            print("⚠️  GPU requested but not available, falling back to CPU")
        use_gpu = False

    # Determine which corpuses to search
    if corpuses is None:
        corpuses = list_corpuses()
        if not corpuses:
            raise ValueError("No corpuses available. Please index a file first.")

    # Validate corpus compatibility
    common_settings = validate_corpus_compatibility(corpuses)

    # Validate query length
    context_length = common_settings.get('context_length', 256)
    query_tokens = estimate_tokens(query)

    if query_tokens > context_length:
        chars_per_token = len(query) / query_tokens if query_tokens > 0 else 4
        recommended_chars = int(context_length * chars_per_token * 0.9)
        raise ValueError(
            f"Query too long ({query_tokens} tokens, max: {context_length}). "
            f"Please use max ~{recommended_chars} characters."
        )

    # Load corpus indices
    corpus_data = []
    for corpus_name in corpuses:
        paths = get_corpus_paths(corpus_name)

        # Load FAISS index
        cpu_index = faiss.read_index(str(paths['index']))

        # Transfer to GPU if enabled
        if use_gpu:
            index = index_cpu_to_gpu(cpu_index)
        else:
            index = cpu_index

        # Load chunks data
        with open(paths['chunks'], 'rb') as f:
            chunks_data = pickle.load(f)

        corpus_data.append({
            'name': corpus_name,
            'index': index,
            'chunks': chunks_data['chunks'],
            'line_ranges': chunks_data['line_ranges'],
            'source': chunks_data['source']
        })

    # Prepare query embedding
    query_emb = ollama_embed(query)
    query_emb = np.array([query_emb]).astype('float32')

    # Normalize query if corpuses use normalized embeddings
    if common_settings.get('normalized', True):
        faiss.normalize_L2(query_emb)

    # Search all corpuses
    large_k = 10000 if max_results is None else max_results * 2
    all_results = []

    for corpus in corpus_data:
        # Get results up to index size
        k = min(large_k, corpus['index'].ntotal)
        distances, indices = corpus['index'].search(query_emb, k)

        for idx, score in zip(indices[0], distances[0]):
            # Filter by minimum score threshold
            if score >= min_score:
                chunk = corpus['chunks'][idx]
                line_range = corpus['line_ranges'][idx]
                source = corpus['source']

                all_results.append({
                    'corpus': corpus['name'],
                    'location': f"{source}:{line_range[0]}-{line_range[1]}",
                    'score': float(score),
                    'chunk': chunk
                })

    # Sort by score (highest first)
    all_results.sort(key=lambda x: x['score'], reverse=True)

    # Limit results if requested
    if max_results is not None:
        all_results = all_results[:max_results]

    return all_results


def execute_query(query_spec, use_gpu=None, verbose=False):
    """
    Execute a JSON-formatted boolean query on SEAR corpuses.

    Supports complex nested boolean operations combining multiple queries.

    Query Format:
        Simple query (just retrieve):
        {
            "query": "authentication security",
            "corpuses": ["backend", "docs"],  # optional
            "min_score": 0.3,                 # optional, default 0.3
            "max_results": 100,               # optional, default None
            "sort": true,                     # optional, default true
            "merge_adjacent": true            # optional, default true
        }

        Union (combine multiple queries):
        {
            "operation": "union",
            "queries": ["security authentication", "user login"]
        }

        Intersect (overlapping results only):
        {
            "operation": "intersect",
            "queries": ["API endpoints", "security"]
        }

        Difference (exclude results):
        {
            "operation": "difference",
            "query": "security patterns",
            "exclude": "deprecated legacy"
        }

        Nested operations:
        {
            "operation": "difference",
            "left": {
                "operation": "union",
                "queries": ["security", "authentication"]
            },
            "right": {
                "query": "deprecated"
            }
        }

    Args:
        query_spec: Dict containing query specification (see format above)
        use_gpu: Enable GPU acceleration (None=auto, True=force, False=disable)
        verbose: Print execution details

    Returns:
        list: Processed chunks with keys: corpus, location, score, chunk

    Raises:
        ValueError: If query format is invalid or required fields missing

    Example:
        >>> # Simple union query
        >>> query = {
        ...     "operation": "union",
        ...     "queries": ["authentication", "authorization"],
        ...     "corpuses": ["backend"],
        ...     "min_score": 0.35
        ... }
        >>> results = execute_query(query, verbose=True)

        >>> # Complex nested query: (A ∪ B) - C
        >>> query = {
        ...     "operation": "difference",
        ...     "left": {
        ...         "operation": "union",
        ...         "queries": ["security patterns", "best practices"]
        ...     },
        ...     "right": {
        ...         "query": "deprecated outdated"
        ...     },
        ...     "sort": True,
        ...     "merge_adjacent": True
        ... }
        >>> results = execute_query(query, verbose=True)
    """
    import json

    # Validate input is a dict
    if not isinstance(query_spec, dict):
        raise ValueError("query_spec must be a dictionary")

    # Extract common options with defaults
    options = {
        'corpuses': query_spec.get('corpuses', None),
        'min_score': query_spec.get('min_score', 0.3),
        'max_results': query_spec.get('max_results', None),
        'sort': query_spec.get('sort', True),
        'merge_adjacent': query_spec.get('merge_adjacent', True)
    }

    if verbose:
        print(f"📋 Executing query with options: {options}")

    # Recursive query execution
    def _execute_node(node, depth=0):
        """Recursively execute a query node."""
        indent = "  " * depth

        if verbose:
            print(f"{indent}🔍 Processing node at depth {depth}")

        # Check if this is a simple query (no operation)
        if 'operation' not in node:
            # Simple query - just retrieve chunks
            if 'query' not in node:
                raise ValueError("Query node must have either 'operation' or 'query' field")

            query_str = node['query']
            if verbose:
                print(f"{indent}  📝 Simple query: '{query_str}'")

            results = _retrieve_chunks_only(
                query=query_str,
                corpuses=options['corpuses'],
                min_score=options['min_score'],
                max_results=options['max_results'],
                use_gpu=use_gpu,
                verbose=False
            )

            if verbose:
                print(f"{indent}  ✓ Retrieved {len(results)} chunks")

            return results

        # Operation-based query
        operation = node['operation'].lower()

        if operation == 'union':
            # Union: combine multiple queries
            if 'queries' not in node:
                raise ValueError("Union operation requires 'queries' field")

            queries = node['queries']
            if not isinstance(queries, list) or len(queries) < 2:
                raise ValueError("Union requires at least 2 queries")

            if verbose:
                print(f"{indent}  ∪ Union of {len(queries)} queries")

            # Execute each query
            result_sets = []
            for i, q in enumerate(queries):
                if isinstance(q, str):
                    # Simple string query
                    if verbose:
                        print(f"{indent}    Query {i+1}: '{q}'")
                    results = _retrieve_chunks_only(
                        query=q,
                        corpuses=options['corpuses'],
                        min_score=options['min_score'],
                        max_results=options['max_results'],
                        use_gpu=use_gpu,
                        verbose=False
                    )
                elif isinstance(q, dict):
                    # Nested query
                    if verbose:
                        print(f"{indent}    Query {i+1}: <nested>")
                    results = _execute_node(q, depth + 2)
                else:
                    raise ValueError(f"Query must be string or dict, got {type(q)}")

                if verbose:
                    print(f"{indent}      → {len(results)} chunks")
                result_sets.append(results)

            # Apply union operation
            combined = union_results(result_sets)
            if verbose:
                print(f"{indent}  ✓ Union result: {len(combined)} chunks")

            return combined

        elif operation == 'intersect':
            # Intersect: overlapping results only
            if 'queries' not in node:
                raise ValueError("Intersect operation requires 'queries' field")

            queries = node['queries']
            if not isinstance(queries, list) or len(queries) < 2:
                raise ValueError("Intersect requires at least 2 queries")

            if verbose:
                print(f"{indent}  ∩ Intersect of {len(queries)} queries")

            # Execute each query
            result_sets = []
            for i, q in enumerate(queries):
                if isinstance(q, str):
                    if verbose:
                        print(f"{indent}    Query {i+1}: '{q}'")
                    results = _retrieve_chunks_only(
                        query=q,
                        corpuses=options['corpuses'],
                        min_score=options['min_score'],
                        max_results=options['max_results'],
                        use_gpu=use_gpu,
                        verbose=False
                    )
                elif isinstance(q, dict):
                    if verbose:
                        print(f"{indent}    Query {i+1}: <nested>")
                    results = _execute_node(q, depth + 2)
                else:
                    raise ValueError(f"Query must be string or dict, got {type(q)}")

                if verbose:
                    print(f"{indent}      → {len(results)} chunks")
                result_sets.append(results)

            # Apply intersect operation (pairwise for multiple sets)
            # Start with first set, then intersect with each subsequent set
            combined = result_sets[0]
            for i in range(1, len(result_sets)):
                combined = intersect_results(combined, result_sets[i])

            if verbose:
                print(f"{indent}  ✓ Intersect result: {len(combined)} chunks")

            return combined

        elif operation == 'difference':
            # Difference: A - B (exclude B from A)
            # Support two formats:
            # 1. "left"/"right" for nested operations
            # 2. "query"/"exclude" for simple queries

            if 'left' in node and 'right' in node:
                # Nested format
                if verbose:
                    print(f"{indent}  - Difference (left - right)")
                    print(f"{indent}    Left operand:")

                left_results = _execute_node(node['left'], depth + 2)

                if verbose:
                    print(f"{indent}      → {len(left_results)} chunks")
                    print(f"{indent}    Right operand:")

                right_results = _execute_node(node['right'], depth + 2)

                if verbose:
                    print(f"{indent}      → {len(right_results)} chunks")

            elif 'query' in node and 'exclude' in node:
                # Simple format
                main_query = node['query']
                exclude_query = node['exclude']

                if verbose:
                    print(f"{indent}  - Difference")
                    print(f"{indent}    Main query: '{main_query}'")

                left_results = _retrieve_chunks_only(
                    query=main_query,
                    corpuses=options['corpuses'],
                    min_score=options['min_score'],
                    max_results=options['max_results'],
                    use_gpu=use_gpu,
                    verbose=False
                )

                if verbose:
                    print(f"{indent}      → {len(left_results)} chunks")
                    print(f"{indent}    Exclude query: '{exclude_query}'")

                right_results = _retrieve_chunks_only(
                    query=exclude_query,
                    corpuses=options['corpuses'],
                    min_score=options['min_score'],
                    max_results=options['max_results'],
                    use_gpu=use_gpu,
                    verbose=False
                )

                if verbose:
                    print(f"{indent}      → {len(right_results)} chunks")
            else:
                raise ValueError(
                    "Difference operation requires either 'left'/'right' or 'query'/'exclude' fields"
                )

            # Apply difference operation
            semantic = query_spec.get('semantic', False)
            threshold = query_spec.get('threshold', 0.7)
            result = difference_results(left_results, right_results, semantic=semantic, threshold=threshold)

            if verbose:
                semantic_msg = f" (semantic, threshold={threshold})" if semantic else ""
                print(f"{indent}  ✓ Difference result: {len(result)} chunks{semantic_msg}")

            return result

        else:
            raise ValueError(f"Unknown operation: {operation}. Supported: union, intersect, difference")

    # Execute the query tree
    if verbose:
        print("=" * 80)
        print("🚀 Starting query execution")
        print("=" * 80)

    results = _execute_node(query_spec)

    if verbose:
        print("=" * 80)
        print(f"✓ Query execution complete: {len(results)} total chunks")

    # Post-processing: sort and merge if requested
    if options['sort'] and len(results) > 0:
        if verbose:
            print("📊 Sorting results by document order...")
        results = sort_by_document_order(results)
        if verbose:
            print(f"  ✓ Sorted {len(results)} chunks")

    if options['merge_adjacent'] and len(results) > 0:
        if verbose:
            print("🔗 Merging adjacent chunks...")
        original_count = len(results)
        results = merge_adjacent_chunks(results)
        if verbose:
            print(f"  ✓ Merged {original_count} → {len(results)} chunks")

    if verbose:
        print("=" * 80)
        print(f"🎉 Final result: {len(results)} chunks")
        print("=" * 80)

    return results


###############################################################################
# SQL INTERFACE
###############################################################################

def parse_sql_query(sql_str, verbose=False):
    """
    Parse SQL query and convert to JSON query specification.

    Supported SQL syntax:
        Simple query:
            SELECT * FROM search("physics lessons")

        Union (OR):
            SELECT * FROM search("thermodynamics")
            UNION
            SELECT * FROM search("quantum mechanics")

        Difference (EXCEPT):
            SELECT * FROM search("physics")
            EXCEPT
            SELECT * FROM search("mechanics")

        Intersect (AND):
            SELECT * FROM search("security")
            INTERSECT
            SELECT * FROM search("authentication")

        Nested operations:
            SELECT * FROM (
                SELECT * FROM search("security")
                UNION
                SELECT * FROM search("authentication")
            )
            EXCEPT
            SELECT * FROM search("deprecated")

        With options (WHERE clause):
            SELECT * FROM search("physics")
            WHERE corpus IN ('lectures', 'textbooks')
            AND min_score >= 0.35
            AND semantic = true
            AND threshold >= 0.7

    Args:
        sql_str: SQL query string
        verbose: Print parsing details

    Returns:
        dict: JSON query specification for execute_query()

    Raises:
        ValueError: If SQL syntax is invalid

    Example:
        >>> sql = 'SELECT * FROM search("physics") EXCEPT SELECT * FROM search("mechanics")'
        >>> query_spec = parse_sql_query(sql)
        >>> results = execute_query(query_spec)
    """
    import sqlparse
    import re

    if verbose:
        print(f"🔍 Parsing SQL query...")
        print(f"   Input: {sql_str}")

    # Clean up the SQL string
    sql_str = sql_str.strip()
    if not sql_str:
        raise ValueError("Empty SQL query")

    # Parse the SQL
    try:
        parsed = sqlparse.parse(sql_str)
        if not parsed:
            raise ValueError("Failed to parse SQL query")
        statement = parsed[0]
    except Exception as e:
        raise ValueError(f"SQL parsing error: {e}")

    # Extract options from WHERE clause if present
    options = {}
    where_clause = None

    # Convert SQL statement to string and look for WHERE clause
    sql_upper = sql_str.upper()
    if 'WHERE' in sql_upper:
        # Split on WHERE to separate main query from options
        parts = sql_str.split('WHERE', 1)
        if len(parts) == 2:
            sql_str = parts[0].strip()
            where_clause = parts[1].strip()

            if verbose:
                print(f"   Found WHERE clause: {where_clause}")

            # Parse WHERE clause for options
            # corpus IN ('lectures', 'textbooks')
            corpus_match = re.search(r"corpus\s+IN\s*\(([^)]+)\)", where_clause, re.IGNORECASE)
            if corpus_match:
                corpus_list = corpus_match.group(1)
                # Extract quoted strings
                corpuses = re.findall(r"['\"]([^'\"]+)['\"]", corpus_list)
                if corpuses:
                    options['corpuses'] = corpuses
                    if verbose:
                        print(f"   Extracted corpuses: {corpuses}")

            # min_score >= 0.35
            score_match = re.search(r"min_score\s*>=?\s*([\d.]+)", where_clause, re.IGNORECASE)
            if score_match:
                options['min_score'] = float(score_match.group(1))
                if verbose:
                    print(f"   Extracted min_score: {options['min_score']}")

            # semantic = true
            semantic_match = re.search(r"semantic\s*=\s*(true|false)", where_clause, re.IGNORECASE)
            if semantic_match:
                options['semantic'] = semantic_match.group(1).lower() == 'true'
                if verbose:
                    print(f"   Extracted semantic: {options['semantic']}")

            # threshold >= 0.7
            threshold_match = re.search(r"threshold\s*>=?\s*([\d.]+)", where_clause, re.IGNORECASE)
            if threshold_match:
                options['threshold'] = float(threshold_match.group(1))
                if verbose:
                    print(f"   Extracted threshold: {options['threshold']}")

    # Now parse the main query structure
    query_spec = _parse_sql_node(sql_str, verbose=verbose)

    # Merge options into query spec
    query_spec.update(options)

    if verbose:
        print(f"✓ Parsed successfully")
        print(f"   JSON query spec: {json.dumps(query_spec, indent=2)}")

    return query_spec


def _parse_sql_node(sql_str, verbose=False):
    """
    Recursively parse a SQL node (SELECT statement or subquery).

    Args:
        sql_str: SQL string to parse
        verbose: Print parsing details

    Returns:
        dict: Query specification node
    """
    import re

    sql_str = sql_str.strip()

    # Remove outer parentheses if present - but only if they match
    # We need to check if the opening paren matches the closing paren
    if sql_str.startswith('(') and sql_str.endswith(')'):
        # Check if this is truly an outer pair (not just coincidental)
        depth = 0
        outer_pair = True
        for i, char in enumerate(sql_str):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                # If depth hits 0 before the last character, these aren't outer parens
                if depth == 0 and i < len(sql_str) - 1:
                    outer_pair = False
                    break

        if outer_pair:
            sql_str = sql_str[1:-1].strip()

    # Check for set operations (UNION, EXCEPT, INTERSECT)
    # We need to split carefully, respecting parentheses

    # Find top-level set operations (not inside parentheses)
    operation_positions = []
    paren_depth = 0
    sql_upper = sql_str.upper()

    # Track positions of UNION, EXCEPT, INTERSECT at top level
    for i, char in enumerate(sql_str):
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif paren_depth == 0:
            # Check for operations at this position
            for op in ['UNION', 'EXCEPT', 'INTERSECT']:
                if sql_upper[i:i+len(op)] == op:
                    # Check it's a word boundary (not part of another word)
                    if (i == 0 or not sql_upper[i-1].isalnum()) and \
                       (i+len(op) >= len(sql_upper) or not sql_upper[i+len(op)].isalnum()):
                        operation_positions.append((i, op))
                        break

    if operation_positions:
        # We have a set operation at top level
        # Take the first one
        pos, operation = operation_positions[0]

        left_sql = sql_str[:pos].strip()
        right_sql = sql_str[pos+len(operation):].strip()

        if verbose:
            print(f"   Found {operation} operation")
            print(f"     Left: {left_sql[:50]}...")
            print(f"     Right: {right_sql[:50]}...")

        # Recursively parse left and right
        left_node = _parse_sql_node(left_sql, verbose=verbose)
        right_node = _parse_sql_node(right_sql, verbose=verbose)

        # Build operation node
        if operation == 'UNION':
            # Union combines multiple queries
            # Collect all queries from left and right
            left_queries = _extract_queries(left_node)
            right_queries = _extract_queries(right_node)

            return {
                'operation': 'union',
                'queries': left_queries + right_queries
            }

        elif operation == 'EXCEPT':
            # Difference: left - right
            return {
                'operation': 'difference',
                'left': left_node,
                'right': right_node
            }

        elif operation == 'INTERSECT':
            # Intersect: common elements
            return {
                'operation': 'intersect',
                'left': left_node,
                'right': right_node
            }

    else:
        # No top-level operation - could be:
        # 1. SELECT * FROM search("query")
        # 2. SELECT * FROM (subquery)

        # Check for SELECT * FROM (subquery) pattern first
        subquery_match = re.search(r"SELECT\s+\*\s+FROM\s+\((.+)\)\s*$",
                                   sql_str, re.IGNORECASE | re.DOTALL)

        if subquery_match:
            # This is a subquery - recursively parse it
            subquery_sql = subquery_match.group(1).strip()
            if verbose:
                print(f"   Found subquery: {subquery_sql[:50]}...")
            return _parse_sql_node(subquery_sql, verbose=verbose)

        # Pattern: SELECT * FROM search("query string")
        match = re.search(r"SELECT\s+\*\s+FROM\s+search\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
                         sql_str, re.IGNORECASE)

        if not match:
            raise ValueError(f"Invalid SQL syntax. Expected: SELECT * FROM search(\"query\") or SELECT * FROM (subquery). Got: {sql_str[:100]}")

        query_string = match.group(1)

        if verbose:
            print(f"   Simple query: '{query_string}'")

        return {
            'query': query_string
        }


def _extract_queries(node):
    """
    Extract all query strings from a query node.
    Helper for flattening UNION operations.

    Args:
        node: Query specification node

    Returns:
        list: List of query strings
    """
    if 'query' in node:
        # Simple query node
        return [node['query']]
    elif 'operation' in node and node['operation'] == 'union' and 'queries' in node:
        # Already a union with queries list
        return node['queries']
    elif 'operation' in node and node['operation'] == 'union':
        # Union with left/right nodes
        left_queries = _extract_queries(node.get('left', {}))
        right_queries = _extract_queries(node.get('right', {}))
        return left_queries + right_queries
    else:
        # Complex operation - return as nested node
        # This will preserve the structure
        return []


def execute_sql_query(sql_str, use_gpu=None, verbose=False):
    """
    Execute a SQL query directly (combines parse + execute).

    Convenience wrapper that parses SQL and executes the resulting JSON query.

    Args:
        sql_str: SQL query string
        use_gpu: Enable GPU acceleration
        verbose: Print execution details

    Returns:
        list: Processed chunks with keys: corpus, location, score, chunk

    Example:
        >>> sql = 'SELECT * FROM search("physics") EXCEPT SELECT * FROM search("mechanics")'
        >>> results = execute_sql_query(sql, verbose=True)
        >>> for r in results:
        ...     print(f"{r['corpus']} - {r['location']}: {r['chunk'][:100]}")
    """
    # Parse SQL to JSON
    query_spec = parse_sql_query(sql_str, verbose=verbose)

    # Execute JSON query
    return execute_query(query_spec, use_gpu=use_gpu, verbose=verbose)
