#!/usr/bin/env python3
"""
Test script to measure SEAR performance with and without GPU acceleration.
"""

import time
import sys
from pathlib import Path
from sear_core import index_file, search, delete_corpus, is_gpu_available

def create_test_data():
    """Create a test text file with some content."""
    test_file = Path("test_data.txt")

    # Create a reasonably sized test document
    content = []
    for i in range(100):
        content.append(f"Section {i}: This is a test section about topic {i}.")
        content.append(f"It contains important information about subject {i}.")
        content.append(f"Here are some details: implementation of feature {i}.")
        content.append(f"We discuss algorithms, data structures, and best practices for {i}.")
        content.append(f"This section also covers testing methodologies for component {i}.")
        content.append("")

    test_file.write_text("\n".join(content))
    return str(test_file)

def cleanup_test_data():
    """Remove test files and corpus."""
    test_file = Path("test_data.txt")
    if test_file.exists():
        test_file.unlink()

    try:
        delete_corpus("test_corpus_cpu")
    except:
        pass

    try:
        delete_corpus("test_corpus_gpu")
    except:
        pass

def main():
    print("="*80)
    print("SEAR PERFORMANCE TEST")
    print("="*80)

    # Check GPU availability
    gpu_available = is_gpu_available()
    print(f"\nGPU Available: {gpu_available}")

    if not gpu_available:
        print("\nNote: GPU not available. Testing CPU performance only.")
        print("To test GPU performance, install CUDA and faiss-gpu.\n")

    # Create test data
    print("\nCreating test data...")
    test_file = create_test_data()
    print(f"✓ Created test file: {test_file}")

    # Test CPU indexing
    print("\n" + "="*80)
    print("TEST 1: CPU INDEXING")
    print("="*80)

    start_time = time.time()
    try:
        chunk_count = index_file(test_file, "test_corpus_cpu", verbose=True, use_gpu=False)
        cpu_index_time = time.time() - start_time
        print(f"\n✓ CPU Indexing completed in {cpu_index_time:.2f} seconds")
        print(f"  Indexed {chunk_count} chunks")
    except Exception as e:
        print(f"❌ CPU indexing failed: {e}")
        cleanup_test_data()
        return 1

    # Test CPU search
    print("\n" + "="*80)
    print("TEST 2: CPU SEARCH")
    print("="*80)

    query = "testing methodologies"
    start_time = time.time()
    try:
        results = search(query, corpuses=["test_corpus_cpu"], verbose=True, use_gpu=False)
        cpu_search_time = time.time() - start_time
        print(f"\n✓ CPU Search completed in {cpu_search_time:.2f} seconds")
        print(f"  Found {results.get('quality_results', 0)} quality results")
    except Exception as e:
        print(f"❌ CPU search failed: {e}")
        cleanup_test_data()
        return 1

    # Test GPU if available
    if gpu_available:
        print("\n" + "="*80)
        print("TEST 3: GPU INDEXING")
        print("="*80)

        start_time = time.time()
        try:
            chunk_count = index_file(test_file, "test_corpus_gpu", verbose=True, use_gpu=True)
            gpu_index_time = time.time() - start_time
            print(f"\n✓ GPU Indexing completed in {gpu_index_time:.2f} seconds")
            print(f"  Indexed {chunk_count} chunks")
            print(f"  Speedup: {cpu_index_time/gpu_index_time:.2f}x faster")
        except Exception as e:
            print(f"❌ GPU indexing failed: {e}")
            cleanup_test_data()
            return 1

        print("\n" + "="*80)
        print("TEST 4: GPU SEARCH")
        print("="*80)

        start_time = time.time()
        try:
            results = search(query, corpuses=["test_corpus_gpu"], verbose=True, use_gpu=True)
            gpu_search_time = time.time() - start_time
            print(f"\n✓ GPU Search completed in {gpu_search_time:.2f} seconds")
            print(f"  Found {results.get('quality_results', 0)} quality results")
            print(f"  Speedup: {cpu_search_time/gpu_search_time:.2f}x faster")
        except Exception as e:
            print(f"❌ GPU search failed: {e}")
            cleanup_test_data()
            return 1

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"CPU Indexing:  {cpu_index_time:.2f}s")
    print(f"CPU Search:    {cpu_search_time:.2f}s")

    if gpu_available:
        print(f"GPU Indexing:  {gpu_index_time:.2f}s ({cpu_index_time/gpu_index_time:.2f}x speedup)")
        print(f"GPU Search:    {gpu_search_time:.2f}s ({cpu_search_time/gpu_search_time:.2f}x speedup)")
    else:
        print("\nGPU tests skipped (GPU not available)")

    print("="*80)

    # Cleanup
    print("\nCleaning up test data...")
    cleanup_test_data()
    print("✓ Cleanup complete")

    return 0

if __name__ == "__main__":
    sys.exit(main())
