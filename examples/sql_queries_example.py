#!/usr/bin/env python3
"""
SQL Query Interface Example for SEAR

This example demonstrates how to use the SQL interface for boolean queries.
Shows both CLI usage and programmatic Python usage.

Prerequisites:
    1. Index some corpus first:
       python sear.py index your-file.txt my-corpus

    2. Make sure you have content indexed before running examples
"""

from sear.core import parse_sql_query, execute_sql_query, execute_query


def example_1_simple_query():
    """Example 1: Simple SQL query"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple SQL Query")
    print("=" * 80)

    sql = 'SELECT * FROM search("authentication")'
    print(f"SQL: {sql}\n")

    # Parse to JSON (for inspection)
    query_spec = parse_sql_query(sql, verbose=True)
    print(f"\nParsed to JSON: {query_spec}")

    # Note: To actually execute, you'd need indexed corpus:
    # results = execute_sql_query(sql, verbose=True)


def example_2_union_query():
    """Example 2: Union operation (OR)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Union Query (OR)")
    print("=" * 80)

    sql = 'SELECT * FROM search("security") UNION SELECT * FROM search("authentication")'
    print(f"SQL: {sql}\n")

    query_spec = parse_sql_query(sql, verbose=True)
    print(f"\nParsed to JSON: {query_spec}")

    # Equivalent CLI command:
    print("\nEquivalent CLI:")
    print('python sear.py search "security, authentication" --union')


def example_3_difference_query():
    """Example 3: Difference operation (EXCEPT)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Difference Query (EXCEPT)")
    print("=" * 80)

    sql = 'SELECT * FROM search("physics") EXCEPT SELECT * FROM search("mechanics")'
    print(f"SQL: {sql}\n")

    query_spec = parse_sql_query(sql, verbose=True)
    print(f"\nParsed to JSON: {query_spec}")

    # Equivalent CLI command:
    print("\nEquivalent CLI:")
    print('python sear.py search "physics" --exclude "mechanics"')


def example_4_intersect_query():
    """Example 4: Intersect operation (AND)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Intersect Query (AND)")
    print("=" * 80)

    sql = 'SELECT * FROM search("API") INTERSECT SELECT * FROM search("security")'
    print(f"SQL: {sql}\n")

    query_spec = parse_sql_query(sql, verbose=True)
    print(f"\nParsed to JSON: {query_spec}")


def example_5_where_clause():
    """Example 5: WHERE clause with options"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: WHERE Clause with Options")
    print("=" * 80)

    sql = """
    SELECT * FROM search("security")
    WHERE corpus IN ('backend', 'api')
    AND min_score >= 0.35
    AND semantic = true
    AND threshold >= 0.75
    """
    print(f"SQL: {sql}\n")

    query_spec = parse_sql_query(sql, verbose=True)
    print(f"\nParsed to JSON: {query_spec}")


def example_6_nested_query():
    """Example 6: Complex nested query"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Complex Nested Query")
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
    print(f"SQL: {sql}\n")

    query_spec = parse_sql_query(sql, verbose=True)
    print(f"\nParsed to JSON: {query_spec}")

    # Equivalent CLI command:
    print("\nEquivalent CLI:")
    print('python sear.py search "security, authentication" --union --exclude "deprecated"')


def example_7_programmatic_usage():
    """Example 7: Programmatic usage with real execution"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Programmatic Usage")
    print("=" * 80)

    print("To execute queries programmatically:\n")

    # Show code example
    code = """
# Method 1: Execute SQL directly
from sear.core import execute_sql_query

sql = 'SELECT * FROM search("security") EXCEPT SELECT * FROM search("deprecated")'
results = execute_sql_query(sql, verbose=True)

for chunk in results:
    print(f"{chunk['corpus']} - {chunk['location']}")
    print(f"Score: {chunk['score']:.3f}")
    print(f"Content: {chunk['chunk'][:100]}...")
    print()

# Method 2: Parse SQL, modify, then execute
from sear.core import parse_sql_query, execute_query

sql = 'SELECT * FROM search("physics") EXCEPT SELECT * FROM search("mechanics")'
query_spec = parse_sql_query(sql)

# Modify the query spec
query_spec['semantic'] = True
query_spec['threshold'] = 0.75
query_spec['min_score'] = 0.4

# Execute modified query
results = execute_query(query_spec, verbose=True)

# Method 3: Build JSON query directly (no SQL)
query_spec = {
    "operation": "difference",
    "left": {
        "operation": "union",
        "queries": ["security", "authentication"]
    },
    "right": {"query": "deprecated"},
    "corpuses": ["backend", "api"],
    "semantic": True,
    "threshold": 0.7,
    "min_score": 0.35
}
results = execute_query(query_spec)
"""
    print(code)


def example_8_cli_usage():
    """Example 8: CLI usage examples"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: CLI Usage Examples")
    print("=" * 80)

    examples = [
        ("Simple query",
         'python sear.py sql "SELECT * FROM search(\\"authentication\\")"'),

        ("Union query",
         'python sear.py sql "SELECT * FROM search(\\"security\\") UNION SELECT * FROM search(\\"auth\\")"'),

        ("Difference query",
         'python sear.py sql "SELECT * FROM search(\\"physics\\") EXCEPT SELECT * FROM search(\\"mechanics\\")"'),

        ("With WHERE clause",
         'python sear.py sql "SELECT * FROM search(\\"security\\") WHERE corpus IN (\'backend\', \'api\') AND min_score >= 0.35"'),

        ("Nested query",
         'python sear.py sql "SELECT * FROM (SELECT * FROM search(\\"security\\") UNION SELECT * FROM search(\\"auth\\")) EXCEPT SELECT * FROM search(\\"deprecated\\")"'),

        ("Extract mode (save to file)",
         'python sear.py sql "SELECT * FROM search(\\"security\\") EXCEPT SELECT * FROM search(\\"deprecated\\")" --mode extract --output security_clean.txt'),

        ("With Anthropic Claude",
         'python sear.py sql "SELECT * FROM search(\\"authentication\\")" --provider anthropic --temperature 0.5'),

        ("With GPU acceleration",
         'python sear.py sql "SELECT * FROM search(\\"security\\") UNION SELECT * FROM search(\\"auth\\")" --gpu'),
    ]

    for title, cmd in examples:
        print(f"\n{title}:")
        print(f"  {cmd}")


def example_9_comparison():
    """Example 9: Three ways to express the same query"""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Three Ways to Express the Same Query")
    print("=" * 80)

    print("\nQuery: Get security OR authentication content, excluding deprecated items")
    print("\n1. SQL Interface:")
    print('   python sear.py sql "SELECT * FROM search(\\"security\\") UNION SELECT * FROM search(\\"auth\\") EXCEPT SELECT * FROM search(\\"deprecated\\") WHERE semantic = true AND threshold >= 0.75"')

    print("\n2. CLI Flags:")
    print('   python sear.py search "security, auth" --union --exclude "deprecated" --semantic --threshold 0.75')

    print("\n3. JSON (programmatic):")
    json_example = """{
    "operation": "difference",
    "left": {
        "operation": "union",
        "queries": ["security", "auth"]
    },
    "right": {"query": "deprecated"},
    "semantic": True,
    "threshold": 0.75
}"""
    print(f"   {json_example}")

    print("\n\nWhen to use each:")
    print("  • SQL: Complex nested queries, familiar syntax, one-off commands")
    print("  • CLI Flags: Quick simple queries, shell scripts, most common use case")
    print("  • JSON: Programmatic use, dynamic query generation, precise control")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("SEAR SQL QUERY INTERFACE - EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the SQL interface for boolean queries in SEAR.")
    print("Note: To execute queries, you need to index content first.")
    print("\nSetup:")
    print("  1. python sear.py index your-file.txt my-corpus")
    print("  2. Run examples below")

    # Run all examples
    example_1_simple_query()
    example_2_union_query()
    example_3_difference_query()
    example_4_intersect_query()
    example_5_where_clause()
    example_6_nested_query()
    example_7_programmatic_usage()
    example_8_cli_usage()
    example_9_comparison()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nFor more information:")
    print("  • BOOLEAN_QUERIES.md - Complete SQL syntax reference")
    print("  • README.md - Main SEAR documentation")
    print("  • test_sql_interface.py - Test suite with more examples")
    print()


if __name__ == '__main__':
    main()
