#!/usr/bin/env python3
"""
SEAR Library Usage Example
Demonstrates programmatic use of SEAR core library with multi-corpus architecture
"""

# Import SEAR core library
from sear.core import (
    index_file,
    search,
    extract_relevant_content,
    list_corpuses,
    validate_corpus_compatibility,
    is_gpu_available
)

def main():
    print("="*80)
    print("SEAR Library Usage Example")
    print("="*80)

    # Check GPU availability
    print("\n0. Checking GPU availability...")
    print("-" * 80)
    if is_gpu_available():
        print("✅ GPU acceleration available!")
        use_gpu = True
    else:
        print("ℹ️  Using CPU mode (GPU not available)")
        use_gpu = False

    # Example 1: Index files as separate corpuses
    print("\n1. Indexing multiple corpuses...")
    print("-" * 80)
    try:
        # Index backend code
        num_chunks = index_file(
            filepath="backend-code.txt",
            corpus_name="backend",
            use_gpu=use_gpu,
            verbose=True  # Set to False for silent operation
        )
        print(f"✅ Indexed {num_chunks} chunks into [backend] corpus")
    except FileNotFoundError:
        print("⚠️  Skipping: backend-code.txt not found (example file)")
    except ValueError as e:
        print(f"❌ Error: {e}")

    try:
        # Index documentation
        num_chunks = index_file(
            filepath="docs.txt",
            corpus_name="documentation",
            use_gpu=use_gpu,
            verbose=True
        )
        print(f"✅ Indexed {num_chunks} chunks into [documentation] corpus")
    except FileNotFoundError:
        print("⚠️  Skipping: docs.txt not found (example file)")
    except ValueError as e:
        print(f"❌ Error: {e}")

    # Example 2: List all available corpuses
    print("\n2. Listing all corpuses...")
    print("-" * 80)
    try:
        corpuses = list_corpuses()
        print(f"✅ Found {len(corpuses)} corpuses:")
        for corpus_info in corpuses:
            print(f"   • {corpus_info['name']}: {corpus_info['chunks']} chunks")
            print(f"     Model: {corpus_info['embedding_model']}, Dimension: {corpus_info['dimension']}")
    except Exception as e:
        print(f"⚠️  Could not list corpuses: {e}")

    # Example 3: Validate corpus compatibility
    print("\n3. Validating corpus compatibility...")
    print("-" * 80)
    try:
        # Check if corpuses use compatible embeddings
        validate_corpus_compatibility(["backend", "documentation"])
        print("✅ Corpuses are compatible (same embedding model and dimension)")
    except ValueError as e:
        print(f"⚠️  Compatibility warning: {e}")
    except Exception as e:
        print(f"⚠️  Could not validate: {e}")

    # Example 4: Perform a search across all corpuses
    print("\n4. Performing search across all corpuses...")
    print("-" * 80)
    try:
        # Search with verbose output
        results = search(
            query="authentication security patterns",
            corpuses=None,  # None = search all corpuses
            use_gpu=use_gpu,
            verbose=True  # Set to False for silent operation
        )

        # Access results programmatically
        print("\n5. Accessing search results programmatically...")
        print("-" * 80)
        print(f"Answer: {results['answer'][:200]}...")
        print(f"\nNumber of sources: {len(results['sources'])}")

        for i, source in enumerate(results['sources'], 1):
            print(f"\nSource {i}:")
            print(f"  Corpus: [{source['corpus']}]")
            print(f"  Location: {source['location']}")
            print(f"  Score: {source['score']:.3f}")
            print(f"  Preview: {source['text'][:100]}...")

    except FileNotFoundError as e:
        print(f"⚠️  Cannot search: {e}")
        print("Index some corpuses first!")
    except Exception as e:
        print(f"❌ Search error: {e}")

    # Example 5: Search specific corpuses only
    print("\n6. Searching specific corpuses...")
    print("-" * 80)
    try:
        # Search only backend corpus
        results = search(
            query="API endpoints",
            corpuses=["backend"],  # Specify corpus names
            use_gpu=use_gpu,
            verbose=False
        )
        print(f"✅ Search completed in [backend] corpus")
        print(f"   Found {len(results['sources'])} results")
    except Exception as e:
        print(f"⚠️  Could not search: {e}")

    # Example 6: Extract relevant content (no LLM processing)
    print("\n7. Extracting relevant content...")
    print("-" * 80)
    try:
        # Extract all relevant content without LLM processing
        extraction = extract_relevant_content(
            query="security testing authentication",
            corpuses=["backend", "documentation"],
            output_file="extracted_content.txt",
            min_score=0.3,
            max_chunks=50,
            use_gpu=use_gpu,
            verbose=False
        )

        print(f"✅ Extraction complete!")
        print(f"   Total chunks: {extraction['total_chunks']}")
        print(f"   Score range: {extraction['score_range']}")
        print(f"   Saved to: {extraction['output_file']}")
        print(f"   Corpus breakdown:")
        for corpus, count in extraction['corpus_breakdown'].items():
            print(f"     [{corpus}]: {count} chunks")

    except FileNotFoundError as e:
        print(f"⚠️  Cannot extract: {e}")
    except Exception as e:
        print(f"⚠️  Extraction skipped: {e}")

    # Example 7: Silent mode (no output)
    print("\n8. Silent mode example...")
    print("-" * 80)
    try:
        # Search silently (no progress messages)
        results = search(
            query="error handling",
            verbose=False,
            use_gpu=use_gpu
        )
        # Process results programmatically
        print("✅ Search completed silently")
        print(f"   Answer length: {len(results['answer'])} chars")
        print(f"   Sources found: {len(results['sources'])}")
        print(f"   Corpuses used: {set(src['corpus'] for src in results['sources'])}")
    except FileNotFoundError:
        print("⚠️  Indices not found - index corpuses first")
    except Exception as e:
        print(f"⚠️  Could not perform silent search: {e}")

    print("\n" + "="*80)
    print("Library usage example complete!")
    print("="*80)
    print("\n💡 Key Takeaways:")
    print("   - Index files as separate corpuses (independent indices)")
    print("   - Search across all corpuses or specify specific ones")
    print("   - Extract content without LLM for raw data analysis")
    print("   - Validate corpus compatibility before combining")
    print("   - Use GPU acceleration when available for better performance")
    print("="*80)

if __name__ == "__main__":
    main()
