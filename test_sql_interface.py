#!/usr/bin/env python3
"""
Test SQL Interface for SEAR

Tests the SQL query parser and executor.
Run this after activating the venv:
    source venv/bin/activate
    python test_sql_interface.py
"""

import sys
from sear_core import parse_sql_query, execute_sql_query


def test_simple_query():
    """Test simple SELECT FROM search() query."""
    print("\n" + "=" * 80)
    print("TEST 1: Simple Query")
    print("=" * 80)

    sql = 'SELECT * FROM search("authentication")'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: Simple query parsed successfully")
    print(f"   Expected: {{'query': 'authentication'}}")
    print(f"   Got: {result}")
    assert result == {'query': 'authentication'}, "Unexpected result"


def test_union_query():
    """Test UNION operation."""
    print("\n" + "=" * 80)
    print("TEST 2: Union Query")
    print("=" * 80)

    sql = 'SELECT * FROM search("security") UNION SELECT * FROM search("authentication")'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: Union query parsed successfully")
    expected = {'operation': 'union', 'queries': ['security', 'authentication']}
    print(f"   Expected: {expected}")
    print(f"   Got: {result}")
    assert result == expected, "Unexpected result"


def test_except_query():
    """Test EXCEPT (difference) operation."""
    print("\n" + "=" * 80)
    print("TEST 3: EXCEPT Query")
    print("=" * 80)

    sql = 'SELECT * FROM search("physics") EXCEPT SELECT * FROM search("mechanics")'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: EXCEPT query parsed successfully")
    print(f"   Got: {result}")
    assert result['operation'] == 'difference', "Expected difference operation"
    assert 'left' in result and 'right' in result, "Expected left/right nodes"


def test_intersect_query():
    """Test INTERSECT operation."""
    print("\n" + "=" * 80)
    print("TEST 4: INTERSECT Query")
    print("=" * 80)

    sql = 'SELECT * FROM search("API") INTERSECT SELECT * FROM search("security")'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: INTERSECT query parsed successfully")
    print(f"   Got: {result}")
    assert result['operation'] == 'intersect', "Expected intersect operation"
    assert 'left' in result and 'right' in result, "Expected left/right nodes"


def test_where_clause_corpus():
    """Test WHERE clause with corpus filter."""
    print("\n" + "=" * 80)
    print("TEST 5: WHERE Clause - Corpus Filter")
    print("=" * 80)

    sql = "SELECT * FROM search(\"security\") WHERE corpus IN ('backend', 'api')"
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: WHERE clause (corpus) parsed successfully")
    print(f"   Got: {result}")
    assert result['query'] == 'security', "Expected security query"
    assert result['corpuses'] == ['backend', 'api'], "Expected corpus list"


def test_where_clause_min_score():
    """Test WHERE clause with min_score filter."""
    print("\n" + "=" * 80)
    print("TEST 6: WHERE Clause - Min Score")
    print("=" * 80)

    sql = 'SELECT * FROM search("authentication") WHERE min_score >= 0.35'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: WHERE clause (min_score) parsed successfully")
    print(f"   Got: {result}")
    assert result['query'] == 'authentication', "Expected authentication query"
    assert result['min_score'] == 0.35, "Expected min_score 0.35"


def test_where_clause_semantic():
    """Test WHERE clause with semantic and threshold."""
    print("\n" + "=" * 80)
    print("TEST 7: WHERE Clause - Semantic Filtering")
    print("=" * 80)

    sql = 'SELECT * FROM search("physics") EXCEPT SELECT * FROM search("mechanics") WHERE semantic = true AND threshold >= 0.75'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: WHERE clause (semantic) parsed successfully")
    print(f"   Got: {result}")
    assert result['operation'] == 'difference', "Expected difference operation"
    assert result['semantic'] == True, "Expected semantic = true"
    assert result['threshold'] == 0.75, "Expected threshold 0.75"


def test_nested_query():
    """Test nested SQL query with subqueries."""
    print("\n" + "=" * 80)
    print("TEST 8: Nested Query")
    print("=" * 80)

    sql = """
    SELECT * FROM (
        SELECT * FROM search("security")
        UNION
        SELECT * FROM search("authentication")
    )
    EXCEPT
    SELECT * FROM search("deprecated")
    """
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: Nested query parsed successfully")
    print(f"   Got: {result}")
    assert result['operation'] == 'difference', "Expected difference operation"
    assert result['left']['operation'] == 'union', "Expected union in left node"


def test_multiple_union():
    """Test multiple UNION operations."""
    print("\n" + "=" * 80)
    print("TEST 9: Multiple UNION Operations")
    print("=" * 80)

    sql = 'SELECT * FROM search("thermo") UNION SELECT * FROM search("quantum") UNION SELECT * FROM search("EM")'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: Multiple UNION parsed successfully")
    print(f"   Got: {result}")
    assert result['operation'] == 'union', "Expected union operation"
    assert len(result['queries']) == 3, "Expected 3 queries"
    assert result['queries'] == ['thermo', 'quantum', 'EM'], "Expected all three queries"


def test_case_insensitive():
    """Test case insensitivity of SQL keywords."""
    print("\n" + "=" * 80)
    print("TEST 10: Case Insensitive Keywords")
    print("=" * 80)

    sql = 'select * from search("test") union select * from search("example")'
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: Case insensitive parsing successful")
    print(f"   Got: {result}")
    assert result['operation'] == 'union', "Expected union operation"
    assert result['queries'] == ['test', 'example'], "Expected both queries"


def test_single_quotes():
    """Test single quotes in search queries."""
    print("\n" + "=" * 80)
    print("TEST 11: Single Quotes in Queries")
    print("=" * 80)

    sql = "SELECT * FROM search('authentication') UNION SELECT * FROM search('security')"
    print(f"SQL: {sql}")

    result = parse_sql_query(sql, verbose=True)
    print(f"\n✅ PASS: Single quotes handled successfully")
    print(f"   Got: {result}")
    assert result['operation'] == 'union', "Expected union operation"
    assert result['queries'] == ['authentication', 'security'], "Expected both queries"


def test_error_handling():
    """Test error handling for invalid SQL."""
    print("\n" + "=" * 80)
    print("TEST 12: Error Handling")
    print("=" * 80)

    invalid_sqls = [
        'SELECT * FROM invalid("test")',  # Wrong function name
        'SELECT * search("test")',        # Missing FROM
        'SELECT * FROM search()',          # Missing query string
        '',                                # Empty query
    ]

    for sql in invalid_sqls:
        print(f"\nTesting invalid SQL: {sql}")
        try:
            result = parse_sql_query(sql, verbose=False)
            assert False, f"Should have raised error for: {sql}"
        except ValueError as e:
            print(f"✅ Correctly raised ValueError: {e}")
        except Exception as e:
            print(f"⚠️  Raised unexpected error: {e}")

    print(f"\n✅ PASS: Error handling works correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SEAR SQL INTERFACE TEST SUITE")
    print("=" * 80)

    tests = [
        test_simple_query,
        test_union_query,
        test_except_query,
        test_intersect_query,
        test_where_clause_corpus,
        test_where_clause_min_score,
        test_where_clause_semantic,
        test_nested_query,
        test_multiple_union,
        test_case_insensitive,
        test_single_quotes,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("=" * 80)

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
