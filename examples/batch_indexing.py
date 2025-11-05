#!/usr/bin/env python3
"""
SEAR Batch Indexing Example
Index multiple files programmatically as separate corpuses or using shell scripting
"""

from pathlib import Path

from sear.core import index_file, is_gpu_available, list_corpuses, search


def batch_index_as_corpuses(files_dict, use_gpu=False):
    """
    Index multiple files as separate corpuses.

    Args:
        files_dict: Dictionary mapping corpus names to file paths
                   e.g., {"backend": "repo1.txt", "frontend": "repo2.txt"}
        use_gpu: Whether to use GPU acceleration (if available)

    Returns:
        dict: Results with corpus names and chunk counts
    """
    print(f"🔄 Batch indexing {len(files_dict)} files as separate corpuses...")
    if use_gpu and is_gpu_available():
        print("🚀 GPU acceleration enabled")
    print("=" * 80)

    results = {}

    for i, (corpus_name, filepath) in enumerate(files_dict.items(), 1):
        print(f"\n📂 [{i}/{len(files_dict)}] Indexing: {filepath} → [{corpus_name}]")
        print("-" * 80)

        try:
            # Index as separate corpus
            num_chunks = index_file(
                filepath=filepath,
                corpus_name=corpus_name,
                use_gpu=use_gpu,
                verbose=False,  # Silent mode for batch processing
            )

            results[corpus_name] = {"file": filepath, "chunks": num_chunks, "status": "success"}

            print(f"✅ Indexed {num_chunks} chunks into corpus [{corpus_name}]")

        except Exception as e:
            print(f"❌ Error indexing {filepath}: {e}")
            results[corpus_name] = {
                "file": filepath,
                "chunks": 0,
                "status": "failed",
                "error": str(e),
            }
            continue

    print(f"\n{'='*80}")
    print(
        f"✅ Batch indexing complete! Indexed {len([r for r in results.values() if r['status'] == 'success'])} corpuses"
    )
    print(f"{'='*80}")

    return results


def batch_index_with_bash():
    """
    Example: Use bash script for batch indexing (recommended for large batches).

    This approach is simpler and more efficient for many files.
    """
    print("🔄 Batch indexing using shell script approach...")
    print("=" * 80)
    print("\nRecommended approach for batch indexing:")
    print("-" * 80)

    bash_script = """
# Bash script for batch indexing multiple repos
for repo in repo1 repo2 repo3; do
    gitingest https://github.com/user/$repo -o ${repo}.txt
    python sear.py index ${repo}.txt $repo
done

# Or for local files with custom corpus names
python sear.py index backend-code.txt backend
python sear.py index frontend-code.txt frontend
python sear.py index api-docs.txt documentation
python sear.py index database-schema.txt database
"""

    print(bash_script)
    print("\n" + "=" * 80)
    print("💡 This approach is recommended because:")
    print("   - Simpler and more maintainable")
    print("   - Each corpus is independent (can be updated separately)")
    print("   - Better for multi-corpus search patterns")
    print("   - No complex merging logic needed")
    print("=" * 80)


def main():
    print("=" * 80)
    print("SEAR Batch Indexing Example")
    print("=" * 80)

    # Example 1: Batch index as separate corpuses (Python approach)
    print("\n1. Python: Batch indexing as separate corpuses...")
    print("-" * 80)

    corpus_files = {
        "backend": "repo1.txt",
        "frontend": "repo2.txt",
        "docs": "repo3.txt",
    }

    # Check which files exist
    existing_corpuses = {name: path for name, path in corpus_files.items() if Path(path).exists()}

    if existing_corpuses:
        # Check GPU availability
        use_gpu = is_gpu_available()
        if use_gpu:
            print("✅ GPU detected - will use acceleration")

        results = batch_index_as_corpuses(existing_corpuses, use_gpu=use_gpu)

        print("\n📊 Indexing Summary:")
        print("-" * 80)
        for corpus_name, result in results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} [{corpus_name}]: {result['chunks']} chunks from {result['file']}")

        # List all corpuses
        print("\n📚 Available corpuses:")
        print("-" * 80)
        try:
            corpuses = list_corpuses()
            for corpus_info in corpuses:
                print(f"  • {corpus_info['name']}: {corpus_info['chunks']} chunks")
        except Exception as e:
            print(f"⚠️  Could not list corpuses: {e}")

        # Example search across all indexed corpuses
        print("\n🔍 Example: Search across all corpuses")
        print("-" * 80)
        try:
            query = "authentication security"
            print(f"Query: '{query}'")
            results = search(query, verbose=False, use_gpu=use_gpu)
            print(f"✅ Found {len(results.get('sources', []))} results")
            for src in results.get("sources", [])[:3]:
                print(f"   [{src['corpus']}] {src['location']} (score: {src['score']:.3f})")
        except Exception as e:
            print(f"⚠️  Search example skipped: {e}")
    else:
        print("⚠️  No files found. Create sample files:")
        print("   echo 'Sample backend code' > repo1.txt")
        print("   echo 'Sample frontend code' > repo2.txt")
        print("   echo 'Sample documentation' > repo3.txt")

    # Example 2: Show bash script approach (recommended)
    print("\n2. Shell Script: Batch indexing (recommended approach)...")
    print("-" * 80)
    batch_index_with_bash()

    print("\n" + "=" * 80)
    print("Batch indexing examples complete!")
    print("=" * 80)
    print("\n💡 Key Concepts:")
    print("   - Each file is indexed as a separate corpus (independent index)")
    print("   - Corpuses can be searched individually or in combination")
    print("   - Organize by: project, type, domain, or version")
    print("   - Use shell scripts for simple batch operations")
    print("=" * 80)


if __name__ == "__main__":
    main()
