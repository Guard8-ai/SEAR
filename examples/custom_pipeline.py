#!/usr/bin/env python3
"""
SEAR Custom Pipeline Example
Use core SEAR functions to build custom workflows with multi-corpus architecture
"""

import pickle
from pathlib import Path

import faiss
import numpy as np

from sear.core import (
    chunk_text,
    get_corpus_dir,
    is_gpu_available,
    ollama_embed,
    ollama_generate,
    validate_input_file,
)


def custom_embedding_pipeline(filepath, custom_chunk_size=256):
    """
    Custom pipeline: Use SEAR's chunking with custom embedding logic.

    Args:
        filepath: Path to text file
        custom_chunk_size: Custom chunk size (default: 256)

    Returns:
        tuple: (chunks, embeddings)
    """
    print(f"🔧 Custom Pipeline: Processing {filepath}")
    print(f"   Chunk size: {custom_chunk_size} tokens")
    print("-" * 80)

    # Validate input file
    filepath = validate_input_file(filepath)

    # Read and chunk
    text = Path(filepath).read_text()
    chunks, line_ranges = chunk_text(text, size=custom_chunk_size)

    print(f"📊 Chunked into {len(chunks)} pieces")

    # Custom embedding logic
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            print(f"   Embedding: {i}/{len(chunks)}")

        # You can customize embedding behavior here
        emb = ollama_embed(chunk[:500])  # Custom truncation
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    print(f"✅ Created {len(embeddings)} embeddings")

    return chunks, embeddings, line_ranges


def custom_search_pipeline(query, corpus_name="backend", top_k=5, use_gpu=False):
    """
    Custom search: Search a specific corpus with custom top_k parameter.

    Args:
        query: Search query
        corpus_name: Name of corpus to search
        top_k: Number of results to return
        use_gpu: Whether to use GPU acceleration

    Returns:
        list: Top k results with scores
    """
    print("🔍 Custom Search Pipeline")
    print(f"   Query: {query}")
    print(f"   Corpus: [{corpus_name}]")
    print(f"   Top K: {top_k}")
    print("-" * 80)

    # Get corpus paths
    corpus_dir = get_corpus_dir(corpus_name)
    index_file = corpus_dir / "index.faiss"
    metadata_file = corpus_dir / "chunks.pkl"

    # Load index
    if not index_file.exists() or not metadata_file.exists():
        print(f"❌ Corpus [{corpus_name}] not found. Index it first.")
        return []

    # Load FAISS index
    corpus_index = faiss.read_index(str(index_file))

    # Move to GPU if requested and available
    if use_gpu and is_gpu_available():
        print("🚀 Using GPU acceleration")
        res = faiss.StandardGpuResources()
        corpus_index = faiss.index_cpu_to_gpu(res, 0, corpus_index)

    with open(metadata_file, "rb") as f:
        corpus_meta = pickle.load(f)

    # Custom query embedding
    query_emb = ollama_embed(query)
    query_emb = np.array([query_emb]).astype("float32")
    faiss.normalize_L2(query_emb)

    # Search with custom top_k
    distances, indices = corpus_index.search(query_emb, top_k)

    # Custom result formatting
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx >= 0:  # Valid index
            result = {
                "chunk": corpus_meta["chunks"][idx],
                "score": float(score),
                "lines": corpus_meta["line_ranges"][idx],
                "source": corpus_meta["source"],
                "corpus": corpus_name,
            }
            results.append(result)

    print(f"✅ Found {len(results)} results")
    return results


def custom_generation_pipeline(context, custom_prompt_template=None):
    """
    Custom generation: Use custom prompt template.

    Args:
        context: Context text
        custom_prompt_template: Custom template (optional)

    Returns:
        str: Generated text
    """
    print("🤖 Custom Generation Pipeline")
    print("-" * 80)

    # Default template
    if custom_prompt_template is None:
        custom_prompt_template = """Context:
{context}

Task: Summarize the above context in 3 bullet points.
Summary:"""

    # Format prompt
    prompt = custom_prompt_template.format(context=context)

    # Generate with custom temperature
    response = ollama_generate(prompt, temperature=0.3)  # Slightly creative

    print(f"✅ Generated {len(response)} characters")
    return response


def main():
    print("=" * 80)
    print("SEAR Custom Pipeline Examples")
    print("=" * 80)

    # Check GPU
    use_gpu = is_gpu_available()
    if use_gpu:
        print("✅ GPU acceleration available")
    print()

    # Example 1: Custom embedding pipeline
    print("\n1. Custom Embedding Pipeline")
    print("=" * 80)
    try:
        chunks, embeddings, line_ranges = custom_embedding_pipeline(
            "sample-code.txt", custom_chunk_size=256  # Smaller chunks
        )
        print("\nResults:")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Line ranges: {len(line_ranges)}")
    except FileNotFoundError:
        print("⚠️  sample-code.txt not found - skipping example")
        print("   Create a sample file: echo 'Sample code' > sample-code.txt")
    except ValueError as e:
        print(f"❌ Error: {e}")

    # Example 2: Custom search pipeline
    print("\n2. Custom Search Pipeline (Top-5)")
    print("=" * 80)
    try:
        results = custom_search_pipeline(
            query="authentication security patterns",
            corpus_name="backend",  # Specify corpus
            top_k=5,  # Get 5 results instead of default
            use_gpu=use_gpu,
        )

        for i, result in enumerate(results[:3], 1):  # Show top 3
            print(f"\nResult {i}:")
            print(f"  Corpus: [{result['corpus']}]")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Lines: {result['lines']}")
            print(f"  Preview: {result['chunk'][:100]}...")

    except Exception as e:
        print(f"⚠️  Search failed: {e}")
        print("   Make sure you have indexed a corpus named 'backend'")

    # Example 3: Custom generation pipeline
    print("\n3. Custom Generation Pipeline")
    print("=" * 80)

    sample_context = """
    SEAR is a three-stage semantic search system.
    It uses FAISS for fast similarity search.
    Ollama provides local embeddings and generation.
    """

    custom_template = """You are a technical writer.

Context:
{context}

Write a one-sentence description:"""

    try:
        summary = custom_generation_pipeline(
            context=sample_context, custom_prompt_template=custom_template
        )
        print("\nGenerated Summary:")
        print(f"  {summary}")
    except Exception as e:
        print(f"⚠️  Generation failed: {e}")

    # Example 4: Multi-corpus custom search
    print("\n4. Multi-Corpus Custom Search")
    print("=" * 80)
    print("Searching multiple corpuses with custom logic:")
    try:
        # Search different corpuses for different aspects
        backend_results = custom_search_pipeline(
            query="API security", corpus_name="backend", top_k=3, use_gpu=use_gpu
        )

        print(f"\n✅ Backend results: {len(backend_results)}")

        # You can combine results from multiple corpuses
        print("\nThis demonstrates searching multiple corpuses")
        print("with custom ranking and filtering logic")
    except Exception as e:
        print(f"⚠️  Multi-corpus search example: {e}")

    # Example 5: Combining custom functions
    print("\n5. Combined Custom Workflow")
    print("=" * 80)
    print("Building custom workflow:")
    print("  1. Custom chunking (256 tokens)")
    print("  2. Custom search (top-k per corpus)")
    print("  3. Custom generation (creative mode)")
    print("  4. Multi-corpus aggregation")
    print("\nThis demonstrates how to chain SEAR core functions")
    print("to create specialized pipelines for your use case.")

    print("\n" + "=" * 80)
    print("Custom pipeline examples complete!")
    print("=" * 80)
    print("\n💡 Tips:")
    print("  - Use custom_chunk_size for granular/coarse chunking")
    print("  - Adjust top_k to get more/fewer search results")
    print("  - Modify temperature for deterministic/creative generation")
    print("  - Create domain-specific prompt templates")
    print("  - Search multiple corpuses and combine results programmatically")
    print("  - Use GPU acceleration for large-scale custom pipelines")


if __name__ == "__main__":
    main()
