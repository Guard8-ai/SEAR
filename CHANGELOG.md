# SEAR Changelog

## v2.3.0 - SQL Query Interface & Project Restructuring (2025-02-06)

### Project Restructuring

**Standard Python src-layout**: SEAR now follows Python packaging best practices with a standard src-layout structure.

**New Structure**:
```
src/
  sear/
    __init__.py      # Public API exports
    cli.py           # CLI interface (from sear.py)
    core.py          # Core functions (from sear_core.py)
  doc_converter/     # Document conversion module
tests/               # All test files
examples/            # Example code
```

**Migration Guide**:
- **CLI**: Use `sear` command (installed via pip) instead of `python sear.py`
- **Imports**: Use `from sear.core import ...` instead of `from sear_core import ...`
- **Entry point**: Configured as `sear.cli:main` in pyproject.toml
- **All tests passing**: 44/44 tests ✅
- **All examples working**: 4/4 example files compile successfully ✅

**Benefits**:
- Standard Python packaging structure
- Cleaner imports and namespace
- Better IDE support and discoverability
- Follows PEP 517/518 modern packaging standards
- Easier to understand for new contributors

### New Features

**SQL Query Interface**

SEAR now supports SQL-like queries for boolean operations, providing a familiar and expressive syntax for complex retrieval filtering. This is a new frontend layer built on top of the existing JSON query engine.

- **SQL Syntax**: Use standard SQL set operations (UNION, EXCEPT, INTERSECT)
- **WHERE Clause**: Filter by corpus, min_score, semantic options
- **Nested Queries**: Support for complex nested subqueries with parentheses
- **CLI Command**: New `sql` command for executing SQL queries
- **Programmatic API**: Import `parse_sql_query()` and `execute_sql_query()` functions

**CLI Syntax**:
```bash
# Simple query
python sear.py sql "SELECT * FROM search(\"authentication\")"

# Union (OR)
python sear.py sql "SELECT * FROM search(\"security\") UNION SELECT * FROM search(\"auth\")"

# Difference (EXCEPT)
python sear.py sql "SELECT * FROM search(\"physics\") EXCEPT SELECT * FROM search(\"mechanics\")"

# Intersect (AND)
python sear.py sql "SELECT * FROM search(\"API\") INTERSECT SELECT * FROM search(\"security\")"

# With WHERE clause
python sear.py sql "SELECT * FROM search(\"security\") WHERE corpus IN ('backend', 'api') AND min_score >= 0.35"

# Nested queries
python sear.py sql "SELECT * FROM (SELECT * FROM search(\"security\") UNION SELECT * FROM search(\"auth\")) EXCEPT SELECT * FROM search(\"deprecated\")"
```

**Programmatic Usage**:
```python
from sear.core import execute_sql_query, parse_sql_query

# Execute SQL query directly
sql = 'SELECT * FROM search("security") EXCEPT SELECT * FROM search("deprecated")'
results = execute_sql_query(sql, verbose=True)

# Parse SQL to JSON for inspection/modification
query_spec = parse_sql_query(sql)
# Returns: {"operation": "difference", "left": {...}, "right": {...}}
```

**Supported Features**:
- **Set Operations**: UNION, EXCEPT (difference), INTERSECT
- **WHERE Clause Options**:
  - `corpus IN ('name1', 'name2')` - Filter by corpus
  - `min_score >= 0.35` - Minimum similarity threshold
  - `semantic = true` - Enable semantic filtering
  - `threshold >= 0.7` - Semantic similarity threshold
- **Nested Queries**: Arbitrary nesting with parentheses
- **Both Modes**: Works with both search (LLM) and extract (file) modes

**Architecture**:
```
SQL Query String
      ↓
parse_sql_query() ← Converts to JSON
      ↓
JSON Query Spec
      ↓
execute_query() ← Existing engine
      ↓
Filtered Chunks
```

The SQL interface is a frontend layer that converts familiar SQL syntax to the existing JSON query format, maintaining full compatibility with all boolean query features introduced in v2.2.0.

**Why This Matters**:
SQL is the universal language for set operations and data filtering. Users familiar with SQL can now express complex boolean queries without learning new syntax. The SQL interface makes SEAR more accessible while maintaining all the power and flexibility of the underlying JSON engine.

**Documentation**: See [BOOLEAN_QUERIES.md - SQL Query Interface](BOOLEAN_QUERIES.md#sql-query-interface) for complete guide with syntax reference, examples, and programmatic usage.

### Files Changed

- [src/sear/core.py](src/sear/core.py) - SQL parser and executor
  - New functions: `parse_sql_query()`, `execute_sql_query()`, `_parse_sql_node()`, `_extract_queries()`
  - Added `__all__` export list for clean public API
  - SQL syntax: SELECT FROM search(), UNION, EXCEPT, INTERSECT, WHERE clause
  - Recursive parsing for nested subqueries

- [src/sear/cli.py](src/sear/cli.py) - CLI support for SQL command
  - Added `sql` command with full option support
  - Modes: search (LLM) and extract (file)
  - Flags: --mode, --output, --temperature, --provider, --api-key, --gpu
  - Updated help text and examples

- [BOOLEAN_QUERIES.md](BOOLEAN_QUERIES.md) - Comprehensive SQL documentation
  - SQL syntax reference with all supported operations
  - CLI usage examples for search and extract modes
  - Programmatic usage guide for Python projects
  - Comparison: SQL vs CLI flags vs JSON
  - Error handling and limitations

- [tests/test_sql_interface.py](tests/test_sql_interface.py) - Comprehensive test suite
  - 12 test cases covering all SQL operations
  - Tests: simple queries, UNION, EXCEPT, INTERSECT, WHERE clause, nesting
  - Error handling validation
  - All tests passing ✅

### Dependencies

- Added `sqlparse` library for SQL parsing (installed automatically with pip)

### Tested Features

All SQL features have been tested and verified:
- ✅ Simple SELECT FROM search() queries
- ✅ UNION operation (multiple queries combined)
- ✅ EXCEPT operation (difference/exclusion)
- ✅ INTERSECT operation (common elements)
- ✅ WHERE clause: corpus IN (...)
- ✅ WHERE clause: min_score >= ...
- ✅ WHERE clause: semantic = true/false
- ✅ WHERE clause: threshold >= ...
- ✅ Nested subqueries with parentheses
- ✅ Multiple UNIONs chained together
- ✅ Case-insensitive SQL keywords
- ✅ Single and double quotes for strings
- ✅ Error handling for invalid SQL

---

## v2.2.0 - Boolean Query Operations (2025-02-05)

### New Features

**Boolean Query Operations**

SEAR now supports SQL-like boolean operations for advanced retrieval filtering, enabling precise control over search results through union, difference, and semantic filtering operations.

- **Union (OR)**: Combine results from multiple targeted queries - `--union` flag
- **Difference (Exclusion)**: Exclude unwanted topics from results - `--exclude` flag
- **Semantic Filtering**: Content-based exclusion using embedding similarity - `--semantic` flag with `--threshold`
- **Complex Queries**: Combine operations for precise filtering - `(A OR B) - C`

**CLI Syntax**:
```bash
# Basic exclusion
python sear.py search "physics lessons" --exclude "mechanics"

# Union of topics
python sear.py search "thermodynamics, quantum, EM" --union

# Complex: (security OR auth) - (deprecated OR legacy)
python sear.py search "security, authentication" --union --exclude "deprecated, legacy"

# Semantic filtering for overlapping topics
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.7
```

**Works for both commands**:
- `search` - Send filtered results to LLM for answer generation
- `extract` - Save filtered content to file without LLM processing

**API Support**:
```python
from sear.core import execute_query

# JSON query format for programmatic use
query_spec = {
    "operation": "difference",
    "left": {"operation": "union", "queries": ["security", "authentication"]},
    "right": {"query": "deprecated"},
    "semantic": True,
    "threshold": 0.7,
    "corpuses": ["backend"],
    "min_score": 0.3
}
results = execute_query(query_spec, verbose=True)
```

**Performance**:
- **Union**: O(n) set-based operations using Python dicts
- **Difference (exact)**: O(n) location-based matching
- **Difference (semantic)**: O(n×m) embedding similarity comparison
- **Document order preservation**: Maintains reading flow, merges adjacent chunks

**Why This Matters**:
Traditional RAG systems retrieve chunks for a single broad query, often including unwanted content. Boolean operations enable precise filtering - get "physics WITHOUT mechanics", combine multiple targeted topics, or exclude deprecated content. This improves answer quality by ensuring only relevant content reaches the LLM.

**Documentation**: See [BOOLEAN_QUERIES.md](BOOLEAN_QUERIES.md) for complete guide with examples, JSON format, and best practices.

### Files Changed

- [src/sear/core.py](src/sear/core.py) - Boolean query operations implementation
  - New functions: `execute_query()`, `_union()`, `_difference()`, `_intersect()`
  - Semantic filtering with embedding similarity
  - Document order preservation and chunk merging

- [src/sear/cli.py](src/sear/cli.py) - CLI parser updates
  - Added flags: `--exclude`, `--union`, `--semantic`, `--threshold`
  - Both `search` and `extract` commands support boolean operations

- [BOOLEAN_QUERIES.md](BOOLEAN_QUERIES.md) - Comprehensive documentation
  - Operation types and use cases
  - CLI syntax and examples
  - JSON query format for programmatic use
  - Performance characteristics and best practices

### Tested Features

All features have been tested and verified:
- ✅ Union operation with multiple queries
- ✅ Difference (exact location matching)
- ✅ Difference (semantic content filtering)
- ✅ Complex nested queries (union + difference)
- ✅ Document order preservation
- ✅ Adjacent chunk merging
- ✅ CLI integration for search and extract
- ✅ JSON query format

---

## v2.1.0 - Quality-Aware Search (2025-01-22)

### New Features

**Empirical Quality Filtering**

SEAR now filters search results based on empirically-determined similarity score thresholds, ensuring only high-quality matches are used for answer generation.

- **Default threshold: 0.30** - Based on empirical testing showing clear separation between relevant (>0.21) and noise (<0.20)
- **Auto-refinement prompts** - When <2 quality results found, suggests query refinement instead of generating low-quality answers
- **Selective context** - Only quality results (score ≥ min_score) are passed to LLM for answer generation
- **Transparent filtering** - Shows how many low-quality results were filtered out

**API Changes**:
```python
search(query, min_score=0.30)  # New parameter with smart default
# Returns: {'answer': str, 'sources': list, 'quality_results': int, 'filtered_count': int}
```

**Why This Matters**:
Traditional RAG systems pass all retrieved chunks to the LLM, including low-quality matches that add noise. SEAR's quality-aware filtering ensures only semantically relevant content is used, improving answer accuracy and reducing hallucinations.

### Empirical Testing Results

Controlled testing on actual corpus revealed clear score patterns:
- **High quality** (0.30-0.54): Direct semantic matches, implementation-specific terms
- **Medium quality** (0.20-0.40): Related concepts, broader topics
- **Low quality** (<0.20): Unrelated content, noise

**Differentiation from Traditional RAG**: Most RAG systems use fixed top-k retrieval without quality assessment. SEAR adds an evidence-based quality gate, making it production-ready for applications requiring high precision.

---

## v2.0.0 - Multi-Corpus Architecture (2025-01-21)

### Major Changes

**Architecture Redesign: From 3-Stage to Multi-Corpus**

This release completely redesigns SEAR's architecture from a fixed 2-index system to a flexible multi-corpus architecture.

### Breaking Changes

- **CLI Commands Changed**:
  - Removed: `index-code`, `index-explain`
  - Added: `index <file> [corpus-name]`, `list`, `delete <corpus-name>`
  - Modified: `search "query"` now accepts `--corpus name1,name2,...` flag

- **API Changes**:
  - `index_file(filepath, corpus_name=None, verbose=True)` - removed `corpus_type` parameter
  - `search(query, corpuses=None, top_k=5, verbose=True)` - replaces `sear_search()`
  - Removed constants: `CODE_INDEX`, `CODE_META`, `EXPLAIN_INDEX`, `EXPLAIN_META`

- **Storage Structure**:
  - Old: `~/.sear/{code,explain}.faiss` and `{code,explain}_meta.pkl`
  - New: `~/.sear/<corpus-name>/{index.faiss, chunks.pkl, metadata.json}`

### New Features

1. **Multi-Corpus Support**
   - Unlimited independent corpuses (vs fixed 2-index system)
   - Each corpus is self-contained with its own directory
   - Corpus names default to filename stem if not specified

2. **Enhanced Metadata** ([src/sear/core.py:259-267](src/sear/core.py#L259-L267))
   - `file_path`, `file_modified`, `file_size` - for cache validation
   - `embedding_model`, `embedding_dim` - for compatibility checks
   - `faiss_index_type`, `normalized` - for search behavior

3. **Corpus Management**
   - `list_corpuses()` - list all available corpuses
   - `get_corpus_info(corpus_name)` - get metadata with chunk count
   - `delete_corpus(corpus_name)` - remove corpus and all files
   - `validate_corpus_compatibility(corpus_names)` - validate embedding compatibility

4. **Flexible Search**
   - Search all corpuses: `search("query")`
   - Search specific corpuses: `search("query", corpuses=["name1", "name2"])`
   - Automatic compatibility validation before searching
   - Results include corpus name, line numbers, and similarity scores

5. **Improved Results Format**
   ```
   📍 SOURCES
   1. [corpus-name] file.txt:10-50 (score: 0.856)
   ```

### Improvements

- **Smart Caching**: Skip re-indexing if file hasn't changed (based on mtime and size)
- **Better Error Messages**: Detailed compatibility errors when mixing incompatible corpuses
- **Parallel Search**: Search multiple corpuses in parallel and aggregate results
- **Corpus Attribution**: Every result shows which corpus it came from

### Migration Guide

No backward compatibility is provided. To migrate:

1. Re-index your files using the new CLI:
   ```bash
   python sear.py index your-code.txt my-corpus
   ```

2. Update your code to use the new API:
   ```python
   # Old (v1.x)
   from sear_core import index_file, sear_search, CODE_INDEX, CODE_META
   index_file("file.txt", CODE_INDEX, CODE_META)
   results = sear_search("query")

   # v2.0 (deprecated structure)
   from sear_core import index_file, search
   index_file("file.txt", "my-corpus")
   results = search("query", corpuses=["my-corpus"])

   # v2.3+ (current)
   from sear.core import index_file, search
   index_file("file.txt", "my-corpus")
   results = search("query", corpuses=["my-corpus"])
   ```

### Files Changed

- [src/sear/core.py](src/sear/core.py) - Complete refactoring (~580 lines, was ~320)
  - Removed legacy constants
  - Added helper functions for corpus management
  - Replaced 3-stage search with 2-stage multi-corpus search
  - Enhanced metadata schema

- [src/sear/cli.py](src/sear/cli.py) - CLI redesign (~165 lines, was ~90)
  - New command structure
  - Added `list` and `delete` commands
  - Updated help text
  - Added `--corpus` flag for search

- [README.md](README.md) - Complete documentation update (~504 lines)
  - Updated architecture diagrams
  - New usage examples
  - Multi-corpus best practices
  - Updated library usage examples

### Technical Details

**Metadata Schema** ([src/sear/core.py:259-267](src/sear/core.py#L259-L267)):
```json
{
  "file_path": "/absolute/path/to/source.txt",
  "file_modified": 1705329600.123,
  "file_size": 52480,
  "embedding_model": "all-minilm",
  "embedding_dim": 384,
  "faiss_index_type": "IndexFlatIP",
  "normalized": true
}
```

**Directory Structure**:
```
~/.sear/
├── corpus-name-1/
│   ├── index.faiss      # FAISS embeddings
│   ├── chunks.pkl       # Chunks + line ranges
│   └── metadata.json    # File + embedding metadata
└── corpus-name-2/
    ├── index.faiss
    ├── chunks.pkl
    └── metadata.json
```

### Tested Features

All features have been tested and verified:
- ✅ Indexing with automatic corpus naming
- ✅ Cache validation (skip re-indexing unchanged files)
- ✅ List command with metadata display
- ✅ Search (implicit - all corpuses)
- ✅ Search (explicit - specific corpuses)
- ✅ Multi-corpus compatibility validation
- ✅ Delete command
- ✅ Corpus attribution in results

### Credits

Implementation: Claude Code (Anthropic)
Architecture Design: Based on requirements in [next_steps.txt](next_steps.txt)
