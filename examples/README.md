# SEAR Examples

This directory contains example code demonstrating various SEAR features.

## Available Examples

### 1. SQL Query Interface

**File:** [sql_queries_example.py](sql_queries_example.py)

Comprehensive examples of the SQL query interface for boolean operations.

**Run it:**
```bash
cd examples
python sql_queries_example.py
```

**What it demonstrates:**
- Simple SQL queries
- UNION operations (OR)
- EXCEPT operations (difference/exclusion)
- INTERSECT operations (AND)
- WHERE clause with filtering options
- Complex nested queries
- Programmatic Python usage
- CLI usage patterns
- Comparison: SQL vs CLI flags vs JSON

**Prerequisites:**
To execute queries (not just see examples), you need to index content first:
```bash
python sear.py index ../your-file.txt my-corpus
```

### 2. Library Usage

**File:** [library_usage.py](library_usage.py)

Demonstrates programmatic use of SEAR core library with multi-corpus architecture.

**Run it:**
```bash
cd examples
python library_usage.py
```

**What it demonstrates:**
- Indexing files as separate corpuses
- Listing all available corpuses
- Validating corpus compatibility
- Performing searches across all corpuses or specific ones
- Extracting relevant content (without LLM processing)
- Silent mode for programmatic use
- GPU acceleration detection and usage
- Accessing search results programmatically

**Key functions shown:**
- `index_file()` - Index files as separate corpuses
- `search()` - Search with various options
- `extract_relevant_content()` - Extract without LLM
- `list_corpuses()` - List available corpuses
- `validate_corpus_compatibility()` - Check compatibility
- `is_gpu_available()` - GPU detection

### 3. Batch Indexing

**File:** [batch_indexing.py](batch_indexing.py)

Index multiple files programmatically as separate corpuses or using shell scripting.

**Run it:**
```bash
cd examples
python batch_indexing.py
```

**What it demonstrates:**
- Batch indexing multiple files as separate corpuses (Python approach)
- Shell script approach for batch operations (recommended)
- Progress tracking and error handling
- Corpus organization strategies
- Multi-corpus search patterns

**Use cases:**
- Indexing multiple repositories
- Organizing by project, type, domain, or version
- Independent corpus updates
- Large-scale indexing operations

**Prerequisites:**
Create sample files to see batch indexing in action:
```bash
echo 'Sample backend code' > repo1.txt
echo 'Sample frontend code' > repo2.txt
echo 'Sample documentation' > repo3.txt
```

### 4. Custom Pipeline

**File:** [custom_pipeline.py](custom_pipeline.py)

Use core SEAR functions to build custom workflows with multi-corpus architecture.

**Run it:**
```bash
cd examples
python custom_pipeline.py
```

**What it demonstrates:**
- Custom embedding pipeline with custom chunk sizes
- Custom search pipeline with configurable top-k
- Custom generation pipeline with custom prompts
- Multi-corpus custom search logic
- Combining SEAR core functions for specialized workflows

**Advanced features:**
- Custom chunking (granular/coarse control)
- Custom result ranking and filtering
- Domain-specific prompt templates
- FAISS index manipulation
- GPU acceleration for custom pipelines

**Key functions shown:**
- `ollama_embed()` - Direct embedding access
- `ollama_generate()` - Direct generation access
- `chunk_text()` - Custom chunking
- `validate_input_file()` - Input validation
- `get_corpus_dir()` - Access corpus directories

## Example Output

The SQL examples show both:
1. **SQL syntax** - The query string
2. **Parsed JSON** - How it's converted internally
3. **CLI equivalent** - Alternative way to express the same query

Example output:
```
================================================================================
EXAMPLE 2: Union Query (OR)
================================================================================
SQL: SELECT * FROM search("security") UNION SELECT * FROM search("authentication")

🔍 Parsing SQL query...
   Input: SELECT * FROM search("security") UNION SELECT * FROM search("authentication")
   Found UNION operation
   ...
✓ Parsed successfully

Parsed to JSON: {'operation': 'union', 'queries': ['security', 'authentication']}

Equivalent CLI:
python sear.py search "security, authentication" --union
```

## Quick Start

If you just want to see how SQL queries work without executing them:

```bash
python sql_queries_example.py
```

This will show all SQL syntax examples and their JSON equivalents.

## More Resources

- [BOOLEAN_QUERIES.md](../BOOLEAN_QUERIES.md) - Complete SQL syntax reference
- [test_sql_interface.py](../test_sql_interface.py) - Comprehensive test suite
- [README.md](../README.md) - Main SEAR documentation
- [CHANGELOG.md](../CHANGELOG.md) - Version history and feature releases

## Contributing Examples

Have a useful example? Please contribute!

1. Create a new `.py` file in this directory
2. Add clear comments and docstrings
3. Include usage instructions at the top
4. Update this README with your example

## Example Structure

Good examples should:
- Be self-contained and runnable
- Include clear comments explaining each step
- Show both CLI and programmatic usage
- Provide expected output
- Include error handling
- Be well-documented

```python
#!/usr/bin/env python3
"""
Brief description of what this example demonstrates.

Prerequisites:
    - List any setup steps

Usage:
    python example_name.py
"""

def main():
    """Main function with clear demonstration"""
    # Your example code here
    pass

if __name__ == '__main__':
    main()
```
