#!/usr/bin/env python3
"""
Test script for boolean query CLI parser
Tests the build_boolean_query function without requiring full SEAR environment
"""

import json
import sys


# Extract just the build_boolean_query function to test in isolation
def build_boolean_query(
    query, exclude=None, union=False, corpuses=None, min_score=0.3, max_chunks=None
):
    """
    Convert CLI boolean arguments to JSON query format for execute_query().
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
    else:
        # No exclusion: use main query directly
        query_spec.update(main_query)

    return query_spec


def test_query(description, **kwargs):
    """Test a query and print results"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Input: {kwargs}")
    try:
        result = build_boolean_query(**kwargs)
        print("\nGenerated JSON:")
        print(json.dumps(result, indent=2))
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("SEAR Boolean Query CLI Parser Tests")
    print("=" * 80)

    tests_passed = 0
    tests_total = 0

    # Test 1: Simple query
    tests_total += 1
    if test_query("Simple query: 'physics lessons'", query="physics lessons"):
        tests_passed += 1

    # Test 2: Union of topics
    tests_total += 1
    if test_query(
        "Union: 'thermodynamics, quantum, electromagnetism' --union",
        query="thermodynamics, quantum, electromagnetism",
        union=True,
    ):
        tests_passed += 1

    # Test 3: Exclusion
    tests_total += 1
    if test_query(
        "Exclusion: 'physics' --exclude 'mechanics'", query="physics", exclude="mechanics"
    ):
        tests_passed += 1

    # Test 4: Multiple exclusions
    tests_total += 1
    if test_query(
        "Multiple exclusions: 'physics' --exclude 'mechanics, optics'",
        query="physics",
        exclude="mechanics, optics",
    ):
        tests_passed += 1

    # Test 5: Complex - union + exclusion
    tests_total += 1
    if test_query(
        "Complex: 'security, authentication' --union --exclude 'deprecated, legacy'",
        query="security, authentication",
        exclude="deprecated, legacy",
        union=True,
    ):
        tests_passed += 1

    # Test 6: With corpus filter
    tests_total += 1
    if test_query(
        "With corpus: 'physics' --corpus 'lectures,notes' --exclude 'mechanics'",
        query="physics",
        exclude="mechanics",
        corpuses=["lectures", "notes"],
    ):
        tests_passed += 1

    # Test 7: With max_chunks
    tests_total += 1
    if test_query(
        "With limits: 'security' --exclude 'old' --max-chunks 50",
        query="security",
        exclude="old",
        max_chunks=50,
    ):
        tests_passed += 1

    # Test 8: Error case - union with single query
    print(f"\n{'='*80}")
    print("TEST: Error case - union with single query")
    print(f"{'='*80}")
    tests_total += 1
    try:
        build_boolean_query(query="single query", union=True)
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised error: {e}")
        tests_passed += 1

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {tests_total - tests_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
