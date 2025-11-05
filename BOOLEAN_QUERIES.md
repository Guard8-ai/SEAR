# Boolean Query Operations in SEAR

SEAR supports SQL-like boolean operations for advanced retrieval filtering. Combine multiple queries, exclude unwanted topics, and apply semantic filtering for precise content extraction.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Operations](#operations)
- [Semantic Filtering](#semantic-filtering)
- [CLI Syntax](#cli-syntax)
- [JSON Query Format](#json-query-format)
- [Performance](#performance)
- [Best Practices](#best-practices)

---

## Overview

### Why Boolean Queries?

Traditional RAG systems retrieve chunks for a single query. Boolean operations enable:

- **Union**: Combine targeted topic queries → `(thermodynamics OR quantum OR EM)`
- **Difference**: Exclude unwanted topics → `physics - mechanics`
- **Complex filters**: `(security OR authentication) - (deprecated OR legacy)`

### Key Benefits

✅ **Deterministic**: Same query always returns same results
✅ **Efficient**: O(n) set-based operations using Python dicts
✅ **Document-aware**: Preserves reading order, merges adjacent chunks
✅ **Flexible**: Works for both `search` (LLM answers) and `extract` (raw content)

---

## Quick Start

### Basic Exclusion

Exclude specific topics from results:

```bash
# Exclude mechanics from physics results
python sear.py search "physics lessons" --exclude "mechanics"

# Exclude multiple topics
python sear.py search "physics" --exclude "mechanics, optics"
```

### Union of Topics

Combine multiple targeted queries (experiment finding: this often works better than broad queries):

```bash
# Combine three physics topics
python sear.py search "thermodynamics, quantum mechanics, electromagnetism" --union

# Extract union to file
python sear.py extract "security, authentication, authorization" --union --output security_topics.txt
```

### Complex Queries

Combine union and exclusion:

```bash
# (security OR authentication) - (deprecated OR legacy)
python sear.py search "security, authentication" --union --exclude "deprecated, legacy"
```

### Semantic Filtering

Use semantic similarity for more thorough exclusion:

```bash
# Exclude chunks semantically similar to "mechanics" (not just exact location matches)
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.7

# Lower threshold = more aggressive filtering
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.6
```

---

## Operations

### 1. Union (OR)

**Purpose**: Combine results from multiple targeted queries
**Use case**: When you want content about ANY of several related topics

```bash
# CLI syntax
python sear.py search "topic1, topic2, topic3" --union

# Example: Get all async programming patterns
python sear.py extract "promises, async await, callbacks" --union --output async_patterns.txt
```

**How it works**:
- Executes separate FAISS searches for each topic
- Combines results using set union
- Deduplicates by `(corpus, location)` key
- Preserves highest score for duplicates

### 2. Difference (Exclusion)

**Purpose**: Exclude unwanted topics from results
**Use case**: When you want A but not B

```bash
# CLI syntax
python sear.py search "main_query" --exclude "unwanted_topic"

# Example: Physics without mechanics
python sear.py search "physics lessons" --exclude "mechanics"
```

**Two levels of matching**:

**Level 1 - Exact matching (default, fast)**:
- Removes chunks with identical `(corpus, location)`
- Pure set operations, O(n)
- Use for: basic filtering, distinct topics

**Level 2 - Semantic matching (optional, thorough)**:
- Removes chunks semantically similar to exclusion set
- Compares embedding similarity using FAISS
- Use for: overlapping topics (e.g., physics vs mechanics)

```bash
# Exact matching only (default)
python sear.py search "physics" --exclude "mechanics"

# Semantic matching (checks content similarity)
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.7
```

### 3. Intersect (AND)

**Purpose**: Find chunks matching ALL topics
**Use case**: When you need overlapping content

**Note**: Intersect is implemented in the core but not exposed in CLI yet (rare use case). Use JSON queries for now.

---

## Semantic Filtering

### What is Semantic Filtering?

Semantic filtering compares **content similarity** between chunks, not just location:

- **Exact matching**: "Does this chunk appear in both result sets?" (same location)
- **Semantic matching**: "Is this chunk similar to any exclusion chunk?" (similar meaning)

### When to Use Semantic Filtering

✅ **Use semantic filtering when**:
- Topics overlap semantically (physics vs mechanics)
- You want thorough exclusion of related concepts
- Exact matching leaves too many similar results

❌ **Don't use semantic filtering when**:
- Topics are distinct (security vs documentation)
- Performance matters more than thoroughness
- You just want union (no exclusion needed)

### Threshold Selection

The `--threshold` parameter (0.0-1.0) controls similarity cutoff:

```bash
# Conservative: only exclude very similar chunks (>= 0.8 similarity)
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.8

# Balanced: moderate filtering (>= 0.7 similarity) [DEFAULT]
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.7

# Aggressive: exclude loosely related chunks (>= 0.6 similarity)
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.6
```

**Guidelines**:
- `0.8-0.9`: Very similar content only
- `0.7`: Good default (recommended)
- `0.6-0.7`: More aggressive exclusion
- `< 0.6`: May exclude too much

### How It Works

1. Execute exact exclusion first (fast path)
2. Load FAISS indices for involved corpuses
3. Retrieve embeddings using `index.reconstruct()`
4. Calculate cosine similarity between remaining chunks and exclusion set
5. Keep only chunks below threshold

**Performance**: O(n×m) where n=remaining chunks, m=exclusion chunks

---

## CLI Syntax

### Search Command

Send filtered results to LLM for answer generation:

```bash
python sear.py search "query" [OPTIONS]

Options:
  --corpus name1,name2    Search specific corpuses only
  --exclude "query"       Exclude matching topics
  --union                 Combine comma-separated queries
  --semantic              Enable semantic exclusion
  --threshold 0.0-1.0     Semantic similarity threshold (default: 0.7)
  --temperature 0.0-1.0   LLM creativity (default: 0.0)
  --provider ollama|anthropic  LLM provider
  --gpu / --no-gpu        Force GPU on/off
```

### Extract Command

Save filtered chunks to file (no LLM processing):

```bash
python sear.py extract "query" [OPTIONS]

Options:
  --output file.txt       Output file (default: extracted_content.txt)
  --corpus name1,name2    Extract from specific corpuses
  --exclude "query"       Exclude matching topics
  --union                 Combine comma-separated queries
  --semantic              Enable semantic exclusion
  --threshold 0.0-1.0     Semantic similarity threshold (default: 0.7)
  --min-score 0.0-1.0     Similarity threshold (default: 0.3)
  --max-chunks N          Limit total chunks
  --gpu / --no-gpu        Force GPU on/off
```

### Examples

```bash
# Simple exclusion
python sear.py search "authentication" --exclude "deprecated"

# Multiple exclusions
python sear.py search "security" --exclude "legacy, deprecated, outdated"

# Union of topics
python sear.py search "async, promises, callbacks" --union

# Complex: union + exclusion
python sear.py search "security, auth" --union --exclude "deprecated, legacy"

# Semantic filtering
python sear.py search "physics" --exclude "mechanics" --semantic --threshold 0.75

# Extract to file
python sear.py extract "API design" --exclude "deprecated" --output api_guide.txt

# Specific corpus
python sear.py search "testing" --corpus my-project --exclude "unit tests"
```

---

## JSON Query Format

For programmatic use or complex queries, use the JSON format directly:

### Simple Query

```json
{
  "query": "authentication security",
  "corpuses": ["backend", "docs"],
  "min_score": 0.3,
  "sort": true,
  "merge_adjacent": true
}
```

### Union Operation

```json
{
  "operation": "union",
  "queries": ["security", "authentication", "authorization"],
  "corpuses": ["backend"],
  "min_score": 0.35
}
```

### Difference (Exclusion)

```json
{
  "operation": "difference",
  "left": {
    "query": "security patterns"
  },
  "right": {
    "query": "deprecated"
  },
  "semantic": true,
  "threshold": 0.7
}
```

### Complex Nested Query

```json
{
  "operation": "difference",
  "left": {
    "operation": "union",
    "queries": ["security", "authentication", "authorization"]
  },
  "right": {
    "operation": "union",
    "queries": ["deprecated", "legacy", "outdated"]
  },
  "semantic": true,
  "threshold": 0.75,
  "corpuses": ["backend", "docs"],
  "min_score": 0.3,
  "sort": true,
  "merge_adjacent": true
}
```

### Using JSON Queries Programmatically

```python
from sear_core import execute_query

# Define query
query_spec = {
    "operation": "difference",
    "left": {
        "operation": "union",
        "queries": ["security", "authentication"]
    },
    "right": {
        "query": "deprecated"
    },
    "semantic": True,
    "threshold": 0.7,
    "corpuses": ["my-corpus"],
    "min_score": 0.3
}

# Execute
results = execute_query(query_spec, verbose=True)

# Results format: list of dicts with keys:
# - corpus: corpus name
# - location: file:line-line
# - score: similarity score
# - chunk: content text
# - chunk_index: index in FAISS
```

---

## Performance

### Complexity

- **Union**: O(n) where n = total chunks across all queries
- **Difference (exact)**: O(n) where n = chunks in left set
- **Difference (semantic)**: O(n×m) where n = remaining chunks, m = exclusion chunks
- **Intersect**: O(n) where n = chunks in smaller set

### Optimization Tips

1. **Use exact matching by default**: Only enable semantic when needed
2. **Specific corpus filters**: Use `--corpus` to reduce search space
3. **Adjust min-score**: Higher thresholds = fewer chunks to process
4. **Limit chunks**: Use `--max-chunks` for testing or previews
5. **GPU acceleration**: Use `--gpu` for large corpuses (if available)

### Benchmarks

Typical performance on 10,000 chunk corpus:

- Union of 3 queries: ~0.5s
- Difference (exact): ~0.3s
- Difference (semantic, 100 exclusion chunks): ~2s
- Complex query `(A ∪ B ∪ C) - D`: ~1s

---

## Best Practices

### Query Design

✅ **DO**:
- Use targeted, specific queries for union
- Test with exact matching before enabling semantic
- Start with default threshold (0.7) and adjust
- Use `extract` to inspect results before sending to LLM

❌ **DON'T**:
- Use overly broad queries (defeats the purpose)
- Enable semantic filtering unnecessarily (performance cost)
- Set threshold too low (< 0.6) without testing
- Forget to specify corpus if you know which one

### Union vs Broad Queries

**Experiment finding**: Targeted queries joined together often work better than broad queries.

```bash
# Better: Specific targeted topics
python sear.py search "thermodynamics, quantum mechanics, EM" --union

# Worse: Single broad query
python sear.py search "physics topics"
```

### Exclusion Strategy

**For distinct topics**: Use exact matching (default)
```bash
python sear.py search "security" --exclude "documentation"
```

**For overlapping topics**: Use semantic matching
```bash
python sear.py search "physics" --exclude "mechanics" --semantic
```

### Testing Workflow

1. Start with simple query to understand corpus content
2. Add exclusions with exact matching
3. Inspect results using `extract` command
4. Enable semantic filtering if needed
5. Adjust threshold based on results
6. Use in `search` command for LLM answers

### Corpus Organization

Boolean queries work best when:
- Related content is in the same corpus
- Corpuses are organized by topic/project
- Documents maintain logical structure

---

## Common Patterns

### Pattern 1: Filter Out Deprecated Content

```bash
# Get current best practices only
python sear.py search "API design patterns" --exclude "deprecated, legacy, outdated"
```

### Pattern 2: Combine Multiple Technologies

```bash
# Get all async programming content
python sear.py extract "promises, async await, callbacks, event loop" --union --output async_complete.txt
```

### Pattern 3: Topic Without Subtopic

```bash
# Physics without mechanics (semantically)
python sear.py search "physics lessons" --exclude "mechanics" --semantic --threshold 0.7
```

### Pattern 4: Multiple Sources, Filtered

```bash
# Security content from multiple codebases, no tests
python sear.py search "authentication security" --corpus backend,frontend,docs --exclude "test, testing, mock"
```

### Pattern 5: Exploratory Analysis

```bash
# Extract all security-related content for manual review
python sear.py extract "security, authentication, authorization, encryption" --union --output security_audit.txt
```

---

## Troubleshooting

### No Results After Exclusion

**Problem**: Query returns 0 chunks after exclusion

**Solutions**:
1. Check if exclusion is too broad
2. Try without semantic filtering first
3. Lower similarity threshold
4. Verify chunks exist for main query

```bash
# Debug: Check main query first
python sear.py extract "my query" --max-chunks 10

# Then add exclusion
python sear.py extract "my query" --exclude "topic" --max-chunks 10
```

### Semantic Filtering Not Working

**Problem**: Same results with/without `--semantic`

**Possible causes**:
1. No semantic overlap between query and exclusion
2. Threshold too high
3. Exact matching already removed all overlapping chunks

**Debug**:
```bash
# Try lower threshold
python sear.py search "query" --exclude "topic" --semantic --threshold 0.6
```

### Too Many Results Excluded

**Problem**: Semantic filtering removes too much

**Solutions**:
1. Increase threshold (0.7 → 0.8)
2. Make exclusion query more specific
3. Use exact matching instead

### Performance Issues

**Problem**: Queries taking too long

**Solutions**:
1. Disable semantic filtering if not needed
2. Use `--corpus` to limit search space
3. Increase `--min-score` to reduce chunks
4. Enable GPU acceleration: `--gpu`

---

## Architecture Notes

### Design Principles

1. **JSON as source of truth**: CLI is a convenience wrapper
2. **Set-based operations**: O(n) efficiency using Python dicts
3. **Document order preservation**: Maintains reading flow
4. **Embedding reuse**: Leverages existing FAISS indices

### Two-Level Matching

**Level 1 (Exact)**: Compare `(corpus, location)` keys
- Fast: O(n) set operations
- Use for: distinct topics, union operations

**Level 2 (Semantic)**: Compare embedding similarity
- Thorough: catches semantically related content
- Use for: overlapping topics, careful exclusion

### Why This Works

1. **Deterministic retrieval**: FAISS always returns same results for same query
2. **Line-level tracking**: Precise overlap detection via location strings
3. **Multi-corpus support**: Already handles parallel queries
4. **Normalized embeddings**: Direct cosine similarity via inner product

---

## Related Documentation

- [README.md](README.md) - Main SEAR documentation
- [EXTRACTION_GUIDE.md](EXTRACTION_GUIDE.md) - Content extraction guide
- [GPU_SUPPORT.md](GPU_SUPPORT.md) - GPU acceleration setup

---

## Feedback & Support

Found a bug or have a feature request? Open an issue on GitHub.

**Design rationale**: See `~/Downloads/SEAR_BOOLEAN_LOGIC_DESIGN.md` for detailed architecture decisions and implementation notes.
