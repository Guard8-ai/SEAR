#!/usr/bin/env python3
"""
Test suite for SEAR Boolean Logic Task 2: Document Order Preservation

Tests the following functions:
- _parse_location(location_string)
- sort_by_document_order(chunks)
- merge_adjacent_chunks(chunks)
"""

import sys
from sear.core import _parse_location, sort_by_document_order, merge_adjacent_chunks


def test_parse_location():
    """Test location string parsing."""
    print("\n=== Testing _parse_location() ===")

    # Test 1: Standard format
    filepath, start, end = _parse_location("src/main.py:10-20")
    assert filepath == "src/main.py", f"Expected 'src/main.py', got '{filepath}'"
    assert start == 10, f"Expected 10, got {start}"
    assert end == 20, f"Expected 20, got {end}"
    print("✓ Standard format: src/main.py:10-20")

    # Test 2: Path with subdirectories
    filepath, start, end = _parse_location("path/to/deep/file.txt:557-659")
    assert filepath == "path/to/deep/file.txt"
    assert start == 557
    assert end == 659
    print("✓ Deep path: path/to/deep/file.txt:557-659")

    # Test 3: Single-level path
    filepath, start, end = _parse_location("file.txt:1-100")
    assert filepath == "file.txt"
    assert start == 1
    assert end == 100
    print("✓ Simple path: file.txt:1-100")

    # Test 4: Large line numbers
    filepath, start, end = _parse_location("data.csv:10000-20000")
    assert filepath == "data.csv"
    assert start == 10000
    assert end == 20000
    print("✓ Large line numbers: data.csv:10000-20000")

    # Test 5: Edge case - missing colon
    filepath, start, end = _parse_location("invalid_format")
    assert filepath == "invalid_format"
    assert start == 0
    assert end == 0
    print("✓ Edge case - missing colon (graceful fallback)")

    # Test 6: Edge case - missing dash
    filepath, start, end = _parse_location("file.txt:100")
    assert filepath == "file.txt:100"
    assert start == 0
    assert end == 0
    print("✓ Edge case - missing dash (graceful fallback)")

    # Test 7: Edge case - malformed line range
    filepath, start, end = _parse_location("file.txt:abc-def")
    assert filepath == "file.txt:abc-def"
    assert start == 0
    assert end == 0
    print("✓ Edge case - malformed line range (graceful fallback)")

    print("All _parse_location() tests passed! ✅\n")


def test_sort_by_document_order():
    """Test document order sorting."""
    print("\n=== Testing sort_by_document_order() ===")

    # Test 1: Sort by start line within same file
    chunks = [
        {'corpus': 'docs', 'location': 'file.txt:100-200', 'score': 0.9, 'chunk': 'C'},
        {'corpus': 'docs', 'location': 'file.txt:10-50', 'score': 0.8, 'chunk': 'A'},
        {'corpus': 'docs', 'location': 'file.txt:51-99', 'score': 0.7, 'chunk': 'B'},
    ]
    sorted_chunks = sort_by_document_order(chunks)
    assert sorted_chunks[0]['location'] == 'file.txt:10-50'
    assert sorted_chunks[1]['location'] == 'file.txt:51-99'
    assert sorted_chunks[2]['location'] == 'file.txt:100-200'
    print("✓ Sort by line number within same file")

    # Test 2: Sort by filename within same corpus
    chunks = [
        {'corpus': 'docs', 'location': 'z_last.txt:1-10', 'score': 0.9, 'chunk': 'Z'},
        {'corpus': 'docs', 'location': 'a_first.txt:1-10', 'score': 0.8, 'chunk': 'A'},
        {'corpus': 'docs', 'location': 'm_middle.txt:1-10', 'score': 0.7, 'chunk': 'M'},
    ]
    sorted_chunks = sort_by_document_order(chunks)
    assert sorted_chunks[0]['location'] == 'a_first.txt:1-10'
    assert sorted_chunks[1]['location'] == 'm_middle.txt:1-10'
    assert sorted_chunks[2]['location'] == 'z_last.txt:1-10'
    print("✓ Sort by filename within same corpus")

    # Test 3: Sort by corpus
    chunks = [
        {'corpus': 'zeta', 'location': 'file.txt:1-10', 'score': 0.9, 'chunk': 'Z'},
        {'corpus': 'alpha', 'location': 'file.txt:1-10', 'score': 0.8, 'chunk': 'A'},
        {'corpus': 'beta', 'location': 'file.txt:1-10', 'score': 0.7, 'chunk': 'B'},
    ]
    sorted_chunks = sort_by_document_order(chunks)
    assert sorted_chunks[0]['corpus'] == 'alpha'
    assert sorted_chunks[1]['corpus'] == 'beta'
    assert sorted_chunks[2]['corpus'] == 'zeta'
    print("✓ Sort by corpus name")

    # Test 4: Multi-level sort (corpus → file → line)
    chunks = [
        {'corpus': 'docs', 'location': 'b.txt:100-200', 'score': 0.9, 'chunk': 'D'},
        {'corpus': 'code', 'location': 'main.py:1-10', 'score': 0.8, 'chunk': 'C'},
        {'corpus': 'docs', 'location': 'a.txt:50-60', 'score': 0.7, 'chunk': 'A'},
        {'corpus': 'docs', 'location': 'a.txt:10-20', 'score': 0.6, 'chunk': 'B'},
    ]
    sorted_chunks = sort_by_document_order(chunks)
    # Expected order: code/main.py:1-10, docs/a.txt:10-20, docs/a.txt:50-60, docs/b.txt:100-200
    assert sorted_chunks[0]['corpus'] == 'code' and sorted_chunks[0]['location'] == 'main.py:1-10'
    assert sorted_chunks[1]['corpus'] == 'docs' and sorted_chunks[1]['location'] == 'a.txt:10-20'
    assert sorted_chunks[2]['corpus'] == 'docs' and sorted_chunks[2]['location'] == 'a.txt:50-60'
    assert sorted_chunks[3]['corpus'] == 'docs' and sorted_chunks[3]['location'] == 'b.txt:100-200'
    print("✓ Multi-level sort (corpus → file → line)")

    # Test 5: Empty list
    sorted_chunks = sort_by_document_order([])
    assert sorted_chunks == []
    print("✓ Empty list edge case")

    # Test 6: Single item
    chunks = [{'corpus': 'test', 'location': 'file.txt:1-10', 'score': 0.5, 'chunk': 'X'}]
    sorted_chunks = sort_by_document_order(chunks)
    assert len(sorted_chunks) == 1
    assert sorted_chunks[0]['chunk'] == 'X'
    print("✓ Single item edge case")

    print("All sort_by_document_order() tests passed! ✅\n")


def test_merge_adjacent_chunks():
    """Test merging of adjacent chunks."""
    print("\n=== Testing merge_adjacent_chunks() ===")

    # Test 1: Merge two adjacent chunks (touching)
    chunks = [
        {'corpus': 'docs', 'location': 'f.txt:1-10', 'score': 0.9, 'chunk': 'First chunk'},
        {'corpus': 'docs', 'location': 'f.txt:11-20', 'score': 0.8, 'chunk': 'Second chunk'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 1
    assert merged[0]['location'] == 'f.txt:1-20'
    assert merged[0]['score'] == 0.9  # Kept highest score
    assert merged[0]['chunk'] == 'First chunk\nSecond chunk'
    print("✓ Merge two touching chunks")

    # Test 2: Merge overlapping chunks
    chunks = [
        {'corpus': 'docs', 'location': 'f.txt:1-15', 'score': 0.9, 'chunk': 'Overlap A'},
        {'corpus': 'docs', 'location': 'f.txt:10-20', 'score': 0.8, 'chunk': 'Overlap B'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 1
    assert merged[0]['location'] == 'f.txt:1-20'
    assert merged[0]['score'] == 0.9
    print("✓ Merge overlapping chunks")

    # Test 3: Do NOT merge chunks with gap
    chunks = [
        {'corpus': 'docs', 'location': 'f.txt:1-10', 'score': 0.9, 'chunk': 'Gap A'},
        {'corpus': 'docs', 'location': 'f.txt:15-20', 'score': 0.8, 'chunk': 'Gap B'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 2
    assert merged[0]['location'] == 'f.txt:1-10'
    assert merged[1]['location'] == 'f.txt:15-20'
    print("✓ Do NOT merge chunks with gap")

    # Test 4: Do NOT merge chunks from different files
    chunks = [
        {'corpus': 'docs', 'location': 'a.txt:1-10', 'score': 0.9, 'chunk': 'File A'},
        {'corpus': 'docs', 'location': 'b.txt:1-10', 'score': 0.8, 'chunk': 'File B'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 2
    assert merged[0]['location'] == 'a.txt:1-10'
    assert merged[1]['location'] == 'b.txt:1-10'
    print("✓ Do NOT merge chunks from different files")

    # Test 5: Do NOT merge chunks from different corpuses
    chunks = [
        {'corpus': 'docs', 'location': 'f.txt:1-10', 'score': 0.9, 'chunk': 'Corpus A'},
        {'corpus': 'code', 'location': 'f.txt:11-20', 'score': 0.8, 'chunk': 'Corpus B'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 2
    assert merged[0]['corpus'] == 'docs'
    assert merged[1]['corpus'] == 'code'
    print("✓ Do NOT merge chunks from different corpuses")

    # Test 6: Merge multiple consecutive chunks
    chunks = [
        {'corpus': 'docs', 'location': 'f.txt:1-10', 'score': 0.9, 'chunk': 'A'},
        {'corpus': 'docs', 'location': 'f.txt:11-20', 'score': 0.8, 'chunk': 'B'},
        {'corpus': 'docs', 'location': 'f.txt:21-30', 'score': 0.7, 'chunk': 'C'},
        {'corpus': 'docs', 'location': 'f.txt:31-40', 'score': 0.6, 'chunk': 'D'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 1
    assert merged[0]['location'] == 'f.txt:1-40'
    assert merged[0]['score'] == 0.9  # Kept highest score
    assert merged[0]['chunk'] == 'A\nB\nC\nD'
    print("✓ Merge multiple consecutive chunks")

    # Test 7: Complex scenario - merge some, keep others
    chunks = [
        {'corpus': 'docs', 'location': 'a.txt:1-10', 'score': 0.9, 'chunk': 'A1'},
        {'corpus': 'docs', 'location': 'a.txt:11-20', 'score': 0.8, 'chunk': 'A2'},
        {'corpus': 'docs', 'location': 'a.txt:30-40', 'score': 0.7, 'chunk': 'A3'},  # Gap
        {'corpus': 'docs', 'location': 'b.txt:1-10', 'score': 0.6, 'chunk': 'B1'},   # Different file
        {'corpus': 'docs', 'location': 'b.txt:11-20', 'score': 0.5, 'chunk': 'B2'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 3
    assert merged[0]['location'] == 'a.txt:1-20'   # A1+A2 merged
    assert merged[1]['location'] == 'a.txt:30-40'  # A3 alone (gap)
    assert merged[2]['location'] == 'b.txt:1-20'   # B1+B2 merged
    print("✓ Complex scenario - selective merging")

    # Test 8: Empty list
    merged = merge_adjacent_chunks([])
    assert merged == []
    print("✓ Empty list edge case")

    # Test 9: Single chunk
    chunks = [{'corpus': 'test', 'location': 'f.txt:1-10', 'score': 0.5, 'chunk': 'Solo'}]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 1
    assert merged[0]['chunk'] == 'Solo'
    print("✓ Single chunk edge case")

    # Test 10: Chunks with same start line (overlapping)
    chunks = [
        {'corpus': 'docs', 'location': 'f.txt:1-10', 'score': 0.9, 'chunk': 'First'},
        {'corpus': 'docs', 'location': 'f.txt:1-15', 'score': 0.8, 'chunk': 'Second'}
    ]
    merged = merge_adjacent_chunks(chunks)
    assert len(merged) == 1
    assert merged[0]['location'] == 'f.txt:1-15'
    print("✓ Chunks with same start line")

    print("All merge_adjacent_chunks() tests passed! ✅\n")


def test_integration():
    """Test sort + merge workflow together."""
    print("\n=== Testing Integration (sort + merge) ===")

    # Realistic scenario: chunks from multiple files, out of order
    chunks = [
        {'corpus': 'docs', 'location': 'chapter3.txt:200-300', 'score': 0.9, 'chunk': 'Chapter 3 part 2'},
        {'corpus': 'docs', 'location': 'chapter1.txt:50-100', 'score': 0.8, 'chunk': 'Chapter 1 part 2'},
        {'corpus': 'docs', 'location': 'chapter1.txt:1-49', 'score': 0.85, 'chunk': 'Chapter 1 part 1'},
        {'corpus': 'docs', 'location': 'chapter3.txt:1-199', 'score': 0.7, 'chunk': 'Chapter 3 part 1'},
        {'corpus': 'docs', 'location': 'chapter2.txt:1-100', 'score': 0.6, 'chunk': 'Chapter 2'},
    ]

    # Step 1: Sort by document order
    sorted_chunks = sort_by_document_order(chunks)

    # Verify sort order
    assert sorted_chunks[0]['location'] == 'chapter1.txt:1-49'
    assert sorted_chunks[1]['location'] == 'chapter1.txt:50-100'
    assert sorted_chunks[2]['location'] == 'chapter2.txt:1-100'
    assert sorted_chunks[3]['location'] == 'chapter3.txt:1-199'
    assert sorted_chunks[4]['location'] == 'chapter3.txt:200-300'
    print("✓ Chunks sorted in document order")

    # Step 2: Merge adjacent chunks
    merged = merge_adjacent_chunks(sorted_chunks)

    # Verify merging
    assert len(merged) == 3  # Chapter 1 (merged), Chapter 2, Chapter 3 (merged)
    assert merged[0]['location'] == 'chapter1.txt:1-100'
    assert merged[0]['chunk'] == 'Chapter 1 part 1\nChapter 1 part 2'
    assert merged[0]['score'] == 0.85  # Highest score from merged chunks

    assert merged[1]['location'] == 'chapter2.txt:1-100'
    assert merged[1]['chunk'] == 'Chapter 2'

    assert merged[2]['location'] == 'chapter3.txt:1-300'
    assert merged[2]['chunk'] == 'Chapter 3 part 1\nChapter 3 part 2'
    assert merged[2]['score'] == 0.9

    print("✓ Adjacent chunks merged correctly")
    print("✓ Scores preserved correctly (max)")
    print("✓ Text concatenated correctly")

    print("All integration tests passed! ✅\n")


def main():
    """Run all tests."""
    print("="*80)
    print("SEAR Boolean Logic - Task 2: Document Order Preservation Tests")
    print("="*80)

    try:
        test_parse_location()
        test_sort_by_document_order()
        test_merge_adjacent_chunks()
        test_integration()

        print("="*80)
        print("✅ ALL TESTS PASSED ✅")
        print("="*80)
        print("\nTask 2 Implementation Complete!")
        print("Ready to move to Task 3: JSON Query Executor")
        print()

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
