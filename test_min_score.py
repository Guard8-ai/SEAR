#!/usr/bin/env python3
"""
Test the min_score filtering implementation
"""

import sys
sys.path.insert(0, '/data/git/Guard8.ai/Summarization-EnhanceAugmentedRetrieval')
from sear_core import search

print("="*80)
print("TEST 1: High-quality query (should pass with default min_score=0.30)")
print("="*80)
result1 = search(
    query="ollama API integration",
    corpuses=['gitingest'],
    top_k=5,
    verbose=True,
    temperature=0.0
)
print(f"\nQuality results: {result1['quality_results']}")
print(f"Filtered count: {result1['filtered_count']}")

print("\n\n" + "="*80)
print("TEST 2: Low-quality query (should suggest refinement)")
print("="*80)
result2 = search(
    query="How to cook pasta?",
    corpuses=['gitingest'],
    top_k=5,
    verbose=True,
    temperature=0.0
)
print(f"\nQuality results: {result2['quality_results']}")
print(f"Filtered count: {result2['filtered_count']}")

print("\n\n" + "="*80)
print("TEST 3: Medium-quality query with lower threshold (min_score=0.20)")
print("="*80)
result3 = search(
    query="What is machine learning?",
    corpuses=['gitingest'],
    top_k=5,
    verbose=True,
    temperature=0.0,
    min_score=0.20
)
print(f"\nQuality results: {result3['quality_results']}")
print(f"Filtered count: {result3['filtered_count']}")

print("\n\n" + "="*80)
print("TEST 4: No filtering (min_score=None)")
print("="*80)
result4 = search(
    query="weather forecast",
    corpuses=['gitingest'],
    top_k=5,
    verbose=True,
    temperature=0.0,
    min_score=None
)
print(f"\nQuality results: {result4['quality_results']}")
print(f"Filtered count: {result4['filtered_count']}")
