#!/usr/bin/env python3
"""
Test suite for SEAR boolean logic operations (Task 1)
Tests union, difference, and intersect operations on chunk results.
"""

import sys

from sear.core import _chunk_to_key, difference_results, intersect_results, union_results


def create_mock_result(corpus, location, score, chunk_text):
    """Helper to create mock chunk results."""
    return {"corpus": corpus, "location": location, "score": score, "chunk": chunk_text}


def test_chunk_to_key():
    """Test chunk key generation."""
    print("Testing _chunk_to_key...")
    result = create_mock_result("physics", "doc.txt:1-10", 0.85, "some text")
    key = _chunk_to_key(result)
    assert key == ("physics", "doc.txt:1-10"), f"Expected ('physics', 'doc.txt:1-10'), got {key}"
    print("✓ _chunk_to_key works correctly")


def test_union_basic():
    """Test basic union operation."""
    print("\nTesting union_results (basic)...")

    set_a = [
        create_mock_result("physics", "doc1.txt:1-10", 0.85, "thermodynamics"),
        create_mock_result("physics", "doc1.txt:11-20", 0.75, "quantum"),
    ]

    set_b = [
        create_mock_result("physics", "doc2.txt:1-10", 0.80, "electromagnetism"),
    ]

    result = union_results([set_a, set_b])

    assert len(result) == 3, f"Expected 3 results, got {len(result)}"
    print(f"✓ Union produced {len(result)} results (expected 3)")


def test_union_duplicates():
    """Test union with duplicate chunks (should keep highest score)."""
    print("\nTesting union_results (with duplicates)...")

    set_a = [
        create_mock_result("physics", "doc1.txt:1-10", 0.85, "thermodynamics"),
    ]

    set_b = [
        create_mock_result(
            "physics", "doc1.txt:1-10", 0.90, "thermodynamics"
        ),  # Same location, higher score
    ]

    result = union_results([set_a, set_b])

    assert len(result) == 1, f"Expected 1 result (duplicate removed), got {len(result)}"
    assert result[0]["score"] == 0.90, f"Expected score 0.90 (highest), got {result[0]['score']}"
    print(f"✓ Union kept highest score: {result[0]['score']}")


def test_difference_basic():
    """Test basic difference operation."""
    print("\nTesting difference_results (basic)...")

    set_a = [
        create_mock_result("physics", "doc1.txt:1-10", 0.85, "thermodynamics"),
        create_mock_result("physics", "doc1.txt:11-20", 0.75, "quantum"),
        create_mock_result("physics", "doc1.txt:21-30", 0.70, "mechanics"),
    ]

    set_b = [
        create_mock_result("physics", "doc1.txt:21-30", 0.70, "mechanics"),  # Exclude this
    ]

    result = difference_results(set_a, set_b)

    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    assert all("mechanics" not in r["chunk"] for r in result), "Mechanics should be excluded"
    print(f"✓ Difference excluded 1 chunk, returned {len(result)} results")


def test_difference_no_overlap():
    """Test difference with no overlapping chunks."""
    print("\nTesting difference_results (no overlap)...")

    set_a = [
        create_mock_result("physics", "doc1.txt:1-10", 0.85, "thermodynamics"),
    ]

    set_b = [
        create_mock_result("physics", "doc2.txt:1-10", 0.75, "mechanics"),
    ]

    result = difference_results(set_a, set_b)

    assert len(result) == 1, f"Expected 1 result (no overlap to exclude), got {len(result)}"
    print(f"✓ Difference with no overlap returned all {len(result)} results")


def test_intersect_basic():
    """Test basic intersection operation."""
    print("\nTesting intersect_results (basic)...")

    set_a = [
        create_mock_result("physics", "doc1.txt:1-10", 0.85, "thermodynamics"),
        create_mock_result("physics", "doc1.txt:11-20", 0.75, "quantum"),
    ]

    set_b = [
        create_mock_result("physics", "doc1.txt:1-10", 0.80, "thermodynamics"),  # Same location
        create_mock_result("physics", "doc2.txt:1-10", 0.70, "mechanics"),
    ]

    result = intersect_results(set_a, set_b)

    assert len(result) == 1, f"Expected 1 result (intersection), got {len(result)}"
    assert (
        result[0]["location"] == "doc1.txt:1-10"
    ), f"Expected doc1.txt:1-10, got {result[0]['location']}"
    assert result[0]["score"] == 0.85, f"Expected score 0.85 (highest), got {result[0]['score']}"
    print(f"✓ Intersection found {len(result)} overlapping chunk with highest score")


def test_intersect_no_overlap():
    """Test intersection with no overlapping chunks."""
    print("\nTesting intersect_results (no overlap)...")

    set_a = [
        create_mock_result("physics", "doc1.txt:1-10", 0.85, "thermodynamics"),
    ]

    set_b = [
        create_mock_result("physics", "doc2.txt:1-10", 0.75, "mechanics"),
    ]

    result = intersect_results(set_a, set_b)

    assert len(result) == 0, f"Expected 0 results (no overlap), got {len(result)}"
    print(f"✓ Intersection with no overlap returned {len(result)} results")


def test_complex_scenario():
    """Test complex scenario: (A OR B OR C) - D"""
    print("\nTesting complex scenario: (A ∪ B ∪ C) - D...")

    set_a = [
        create_mock_result("physics", "doc1.txt:1-10", 0.85, "thermodynamics"),
        create_mock_result("physics", "doc1.txt:11-20", 0.75, "quantum"),
    ]

    set_b = [
        create_mock_result("physics", "doc1.txt:21-30", 0.80, "electromagnetism"),
    ]

    set_c = [
        create_mock_result("physics", "doc1.txt:31-40", 0.70, "optics"),
    ]

    set_d = [
        create_mock_result("physics", "doc1.txt:11-20", 0.75, "quantum"),  # Exclude quantum
    ]

    # Perform (A OR B OR C)
    union_result = union_results([set_a, set_b, set_c])
    assert len(union_result) == 4, f"Expected 4 results from union, got {len(union_result)}"

    # Perform - D
    final_result = difference_results(union_result, set_d)
    assert len(final_result) == 3, f"Expected 3 results after exclusion, got {len(final_result)}"
    assert all("quantum" not in r["chunk"] for r in final_result), "Quantum should be excluded"

    print(
        f"✓ Complex operation: Union produced {len(union_result)} results, difference excluded 1, final: {len(final_result)}"
    )


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("SEAR Boolean Operations Test Suite (Task 1)")
    print("=" * 80)

    try:
        test_chunk_to_key()
        test_union_basic()
        test_union_duplicates()
        test_difference_basic()
        test_difference_no_overlap()
        test_intersect_basic()
        test_intersect_no_overlap()
        test_complex_scenario()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
