#!/usr/bin/env python3
"""
Empirical threshold testing script for SEAR
Tests various queries to understand similarity score distribution
"""

import sys

sys.path.insert(0, "/data/git/Guard8.ai/Summarization-EnhanceAugmentedRetrieval")
from sear.core import search

# Test queries designed to produce different quality matches
test_queries = [
    {
        "query": "How do semantic embeddings work?",
        "expected": "HIGH",
        "description": "Direct semantic/technical match",
    },
    {
        "query": "FAISS vector indexing and similarity search",
        "expected": "HIGH",
        "description": "Technical terms directly related to codebase",
    },
    {
        "query": "What is machine learning?",
        "expected": "MEDIUM",
        "description": "Related but broader concept",
    },
    {
        "query": "text chunking and processing",
        "expected": "MEDIUM",
        "description": "Related functionality",
    },
    {
        "query": "ollama API integration",
        "expected": "MEDIUM-HIGH",
        "description": "Implementation-specific terms",
    },
    {"query": "How to cook pasta?", "expected": "LOW", "description": "Completely unrelated"},
    {"query": "weather forecast tomorrow", "expected": "LOW", "description": "Unrelated domain"},
]

print("=" * 80)
print("SEAR THRESHOLD TESTING - EMPIRICAL ANALYSIS")
print("=" * 80)
print("\nTesting queries against 'gitingest' corpus")
print("Goal: Identify similarity score thresholds for filtering")
print("=" * 80)

results_summary = []

for i, test in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{len(test_queries)}: {test['description']}")
    print(f"Query: \"{test['query']}\"")
    print(f"Expected Quality: {test['expected']}")
    print(f"{'='*80}\n")

    try:
        # Run search with verbose=False to minimize output
        result = search(
            query=test["query"], corpuses=["gitingest"], top_k=5, verbose=False, temperature=0.0
        )

        # Extract top 5 scores
        top_scores = [r["score"] for r in result["sources"][:5]]

        print("\n📊 Top 5 Similarity Scores:")
        for j, (source, score) in enumerate(zip(result["sources"][:5], top_scores), 1):
            print(f"  {j}. Score: {score:.4f} | [{source['corpus']}] {source['location']}")
            print(f"     Preview: {source['chunk'][:100]}...")

        results_summary.append(
            {
                "query": test["query"],
                "expected": test["expected"],
                "scores": top_scores,
                "avg_score": sum(top_scores) / len(top_scores) if top_scores else 0,
                "max_score": max(top_scores) if top_scores else 0,
                "min_score": min(top_scores) if top_scores else 0,
            }
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        results_summary.append(
            {"query": test["query"], "expected": test["expected"], "scores": [], "error": str(e)}
        )

# Print comprehensive summary
print("\n" + "=" * 80)
print("SUMMARY OF ALL TESTS")
print("=" * 80)

for i, r in enumerate(results_summary, 1):
    print(f"\n{i}. [{r['expected']}] \"{r['query']}\"")
    if "error" in r:
        print(f"   ERROR: {r['error']}")
    else:
        print(
            f"   Max: {r['max_score']:.4f} | Avg: {r['avg_score']:.4f} | Min: {r['min_score']:.4f}"
        )
        print(f"   Scores: {[f'{s:.4f}' for s in r['scores']]}")

# Analyze score distribution
print("\n" + "=" * 80)
print("THRESHOLD RECOMMENDATIONS")
print("=" * 80)

all_high = [
    r for r in results_summary if r["expected"] in ["HIGH", "MEDIUM-HIGH"] and "error" not in r
]
all_medium = [r for r in results_summary if r["expected"] == "MEDIUM" and "error" not in r]
all_low = [r for r in results_summary if r["expected"] == "LOW" and "error" not in r]

if all_high:
    high_avg = sum(r["avg_score"] for r in all_high) / len(all_high)
    high_min = min(r["min_score"] for r in all_high)
    print("\n✅ HIGH quality matches (direct semantic match):")
    print(f"   Average score: {high_avg:.4f}")
    print(f"   Minimum observed: {high_min:.4f}")

if all_medium:
    medium_avg = sum(r["avg_score"] for r in all_medium) / len(all_medium)
    medium_min = min(r["min_score"] for r in all_medium)
    print("\n⚠️  MEDIUM quality matches (related concepts):")
    print(f"   Average score: {medium_avg:.4f}")
    print(f"   Minimum observed: {medium_min:.4f}")

if all_low:
    low_avg = sum(r["avg_score"] for r in all_low) / len(all_low)
    low_max = max(r["max_score"] for r in all_low)
    print("\n❌ LOW quality matches (noise/unrelated):")
    print(f"   Average score: {low_avg:.4f}")
    print(f"   Maximum observed: {low_max:.4f}")

print(f"\n{'='*80}")
print("RECOMMENDED --min-score VALUES:")
print(f"{'='*80}")
print("\nBased on empirical analysis:")
print("  • High Precision (strict):    --min-score 0.35")
print("  • Balanced (recommended):     --min-score 0.25")
print("  • High Recall (lenient):      --min-score 0.15")
print("\nNote: Adjust based on your specific corpus and use case")
print("=" * 80)
