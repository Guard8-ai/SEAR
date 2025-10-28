# SEAR Extraction Guide

**Extract ALL relevant content from your corpus without LLM processing**

## Table of Contents
- [Overview](#overview)
- [When to Use Extract vs Search](#when-to-use-extract-vs-search)
- [Basic Usage](#basic-usage)
- [Advanced Options](#advanced-options)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)
- [Output Format](#output-format)
- [Threshold Tuning](#threshold-tuning)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Overview

The `extract` command retrieves ALL chunks above a similarity threshold and saves them to a file **without sending to an LLM**. This provides:

- **No hallucination**: Raw content from your corpus, no AI interpretation
- **Complete context**: Get ALL relevant chunks, not just top-k
- **Document order**: Content reconstructed in original sequence
- **Audit trail**: Know exactly what was extracted and from where
- **Reduced cognitive load**: Review structured extraction vs parsing LLM output

```bash
python sear.py extract "your query keywords" --output extracted.txt
```

## When to Use Extract vs Search

### Use `extract` when you want to:
- ✅ Review ALL relevant content on a topic
- ✅ Create focused documentation from corpus
- ✅ Export content for external analysis
- ✅ Avoid LLM hallucination (get raw source content)
- ✅ Build custom reports or summaries manually
- ✅ Audit what information exists on a topic

### Use `search` when you want to:
- ✅ Get a quick AI-generated answer
- ✅ Synthesize information from multiple sources
- ✅ Ask specific questions requiring reasoning
- ✅ Save time by letting LLM summarize findings

## Basic Usage

### Simple Extraction
```bash
# Extract all relevant content with default settings
python sear.py extract "authentication security"
# Output: extracted_content.txt (default)
```

### Specify Output File
```bash
python sear.py extract "testing methodologies" --output testing_guide.txt
```

### Extract from Specific Corpuses
```bash
python sear.py extract "API endpoints" --corpus backend,frontend
```

### Extract with GPU Acceleration
```bash
python sear.py extract "large topic query" --gpu
```

## Advanced Options

### All Command-Line Options

```bash
python sear.py extract "query" \
  --output file.txt \              # Output file (default: extracted_content.txt)
  --corpus name1,name2 \            # Specific corpuses (default: all)
  --min-score 0.3 \                 # Similarity threshold (default: 0.3)
  --max-chunks 100 \                # Limit total chunks (default: unlimited)
  --gpu                             # Force GPU acceleration
  --no-gpu                          # Force CPU mode
```

### Programmatic Usage

```python
from sear_core import extract_relevant_content

result = extract_relevant_content(
    query="authentication security validation",
    corpuses=["backend", "docs"],  # None = all corpuses
    output_file="security_audit.txt",
    min_score=0.3,                 # Similarity threshold
    max_chunks=200,                # Optional limit
    use_gpu=True,                  # GPU acceleration
    verbose=True                   # Print progress
)

# Access results
print(f"Total chunks: {result['total_chunks']}")
print(f"Score range: {result['score_range']}")
print(f"Output file: {result['output_file']}")
print(f"Sources: {result['sources']}")
```

## Use Cases

### 1. Security Audit Documentation
Extract all security-related content for review:

```bash
python sear.py extract "authentication authorization security validation" \
  --corpus backend,api \
  --output security_audit.txt \
  --min-score 0.35
```

**Use case**: Preparing for security audit, need complete picture of security implementations.

### 2. Testing Documentation
Gather all testing-related content:

```bash
python sear.py extract "testing methodologies unit tests integration tests" \
  --corpus codebase,docs \
  --output testing_guide.txt \
  --min-score 0.3
```

**Use case**: Creating comprehensive testing documentation for new team members.

### 3. Feature Research
Research how a feature is implemented across codebase:

```bash
python sear.py extract "payment processing stripe webhook" \
  --corpus backend \
  --output payment_research.txt \
  --max-chunks 50
```

**Use case**: Understanding payment integration before making changes.

### 4. API Documentation Export
Extract all API endpoint documentation:

```bash
python sear.py extract "API endpoint route handler REST" \
  --corpus backend,api_docs \
  --output api_reference.txt \
  --min-score 0.4
```

**Use case**: Creating external API documentation for clients.

### 5. Machine Learning Research
Extract ML/AI related content:

```bash
python sear.py extract "neural network transformer attention mechanism model training" \
  --corpus ml_papers,ml_code \
  --output ml_research.txt \
  --min-score 0.35
```

**Use case**: Literature review or understanding ML implementations.

### 6. Compliance Documentation
Extract regulatory compliance content:

```bash
python sear.py extract "GDPR CCPA privacy data protection compliance" \
  --corpus legal,backend \
  --output compliance_audit.txt
```

**Use case**: Preparing for compliance audit or policy review.

## Best Practices

### 1. Choosing Keywords
**Good keywords**: Specific, relevant terms that appear in target content
```bash
# Good: Specific technical terms
python sear.py extract "OAuth2 JWT authentication token validation"

# Less effective: Too generic
python sear.py extract "code"
```

### 2. Balancing Query Breadth
- **Broad queries** (3-5 keywords): Get comprehensive coverage
- **Narrow queries** (1-2 keywords): Get highly focused results

```bash
# Broad: Get all authentication-related content
python sear.py extract "authentication authorization security validation session"

# Narrow: Get OAuth-specific content only
python sear.py extract "OAuth2"
```

### 3. Using Multiple Extractions
For complex topics, do multiple targeted extractions:

```bash
# Extract authentication first
python sear.py extract "authentication login password" --output auth.txt

# Then extract authorization
python sear.py extract "authorization permissions roles" --output authz.txt

# Finally extract security
python sear.py extract "security validation encryption" --output security.txt
```

### 4. Iterative Threshold Tuning
Start with default (0.3), adjust based on results:

```bash
# First pass with default
python sear.py extract "topic" --min-score 0.3

# If too many low-quality results, increase
python sear.py extract "topic" --min-score 0.4

# If missing relevant content, decrease
python sear.py extract "topic" --min-score 0.25
```

## Output Format

### File Structure

```
================================================================================
SEAR EXTRACTED CONTENT
================================================================================
Query: "authentication security validation"
Corpus: backend, api_docs
Min Score: 0.3
Total Chunks: 42
Score Range: 0.312 - 0.847
Date: 2025-10-23 14:32:15

================================================================================
SOURCE: /path/to/source/file.txt (15 chunks)
================================================================================

[Reconstructed text from file.txt, lines 10-50]
...
[Reconstructed text from file.txt, lines 120-180]
...

================================================================================
SOURCE: /path/to/another/file.txt (27 chunks)
================================================================================

[Reconstructed text from another file.txt, lines 5-30]
...

================================================================================
END OF EXTRACTION
================================================================================
```

### Key Features
- **Header**: Query info, corpus names, statistics
- **Source sections**: Grouped by source file
- **Document order**: Content appears in original sequence
- **Merged chunks**: Consecutive/overlapping chunks are merged
- **Footer**: Clear end marker

## Threshold Tuning

### Understanding Similarity Scores

Similarity scores range from 0.0 (no match) to 1.0 (perfect match):

| Score Range | Quality | Description |
|-------------|---------|-------------|
| 0.7 - 1.0 | Excellent | Highly relevant, exact topic match |
| 0.5 - 0.7 | Very Good | Strong relevance, on-topic |
| 0.3 - 0.5 | Good | Relevant, may have some tangential content |
| 0.2 - 0.3 | Fair | Marginal relevance, review carefully |
| < 0.2 | Poor | Likely off-topic or noise |

### Recommended Thresholds by Use Case

```bash
# High precision: Critical compliance/security
python sear.py extract "query" --min-score 0.4  # Fewer, higher quality results

# Balanced (default): General documentation
python sear.py extract "query" --min-score 0.3  # Recommended default

# High recall: Exploratory research
python sear.py extract "query" --min-score 0.2  # More results, review needed

# Maximum recall: Comprehensive sweep
python sear.py extract "query" --min-score 0.15 # Many results, careful review
```

### Empirical Threshold Testing

Test different thresholds to find optimal value:

```bash
# Test with high threshold
python sear.py extract "topic" --min-score 0.5 --output high.txt

# Test with medium threshold
python sear.py extract "topic" --min-score 0.3 --output medium.txt

# Test with low threshold
python sear.py extract "topic" --min-score 0.2 --output low.txt

# Compare file sizes and review quality
ls -lh high.txt medium.txt low.txt
```

## Performance

### Extraction Speed

Performance depends on:
- **Corpus size**: Larger indices take longer to search
- **Number of matches**: More chunks = more reconstruction work
- **GPU availability**: GPU can speed up large searches
- **File I/O**: Reading source files for reconstruction

**Typical performance** (with GPU):
- Small corpus (< 100 chunks): < 1 second
- Medium corpus (100-1000 chunks): 1-5 seconds
- Large corpus (1000-10000 chunks): 5-30 seconds
- Very large corpus (10000+ chunks): 30-60 seconds

### Optimization Tips

1. **Use GPU for large corpuses**:
```bash
python sear.py extract "query" --gpu
```

2. **Limit chunk count for large result sets**:
```bash
python sear.py extract "broad query" --max-chunks 100
```

3. **Extract from specific corpuses** (avoid searching all):
```bash
python sear.py extract "query" --corpus specific_corpus
```

4. **Use higher thresholds** to reduce result size:
```bash
python sear.py extract "query" --min-score 0.4
```

### Performance Benchmarks

From test results on 42-chunk corpus (gitingest-style content):

| Operation | Chunks | Time | Throughput |
|-----------|--------|------|------------|
| Wide query (17 chunks) | 17 | 0.01s | ~1,700 chunks/s |
| Specific query (3 chunks) | 3 | 0.01s | ~300 chunks/s |
| Limited extraction (15 chunks) | 15 | 0.01s | ~1,500 chunks/s |
| Performance test (25 chunks) | 25 | 0.01s | ~2,200 chunks/s |

**Hardware**: RTX 4060, CUDA 13.0, faiss-gpu 1.12.0

## Troubleshooting

### No Results Found

**Problem**: Extraction returns 0 chunks

**Solutions**:
1. Lower the threshold:
   ```bash
   python sear.py extract "query" --min-score 0.2
   ```

2. Broaden your query with more keywords:
   ```bash
   # Instead of: "OAuth"
   # Try: "OAuth authentication authorization"
   ```

3. Check corpus contains relevant content:
   ```bash
   python sear.py list  # Verify corpus exists
   ```

### Too Many Results

**Problem**: Extraction returns hundreds of chunks

**Solutions**:
1. Increase threshold for higher quality:
   ```bash
   python sear.py extract "query" --min-score 0.4
   ```

2. Limit chunk count:
   ```bash
   python sear.py extract "query" --max-chunks 50
   ```

3. Use more specific keywords:
   ```bash
   # Instead of: "security"
   # Try: "SQL injection XSS CSRF prevention"
   ```

### Low Quality Results

**Problem**: Results contain irrelevant content

**Solutions**:
1. Increase minimum score threshold:
   ```bash
   python sear.py extract "query" --min-score 0.4
   ```

2. Refine keywords to be more specific:
   ```bash
   # Instead of: "testing"
   # Try: "unit testing pytest fixtures mocking"
   ```

3. Extract from specific relevant corpuses:
   ```bash
   python sear.py extract "query" --corpus relevant_corpus
   ```

### Missing Source Files

**Problem**: Warning about missing source files

**Cause**: Original files moved/deleted after indexing

**Solutions**:
1. Re-index with current files:
   ```bash
   python sear.py index updated_file.txt corpus_name
   ```

2. Extraction will continue with available files

### Performance Issues

**Problem**: Extraction is slow

**Solutions**:
1. Enable GPU acceleration:
   ```bash
   python sear.py gpu-info  # Check availability
   python sear.py extract "query" --gpu
   ```

2. Reduce result size with higher threshold:
   ```bash
   python sear.py extract "query" --min-score 0.4
   ```

3. Limit chunk count:
   ```bash
   python sear.py extract "query" --max-chunks 100
   ```

## Examples

### Complete Workflow Example

```bash
# 1. Index your codebase
python sear.py index backend_code.txt backend
python sear.py index api_docs.txt api_docs

# 2. Extract security content for audit
python sear.py extract "security authentication authorization validation" \
  --corpus backend,api_docs \
  --output security_audit.txt \
  --min-score 0.35

# 3. Review extracted content
cat security_audit.txt

# 4. If needed, refine with higher threshold
python sear.py extract "security authentication authorization" \
  --corpus backend,api_docs \
  --output security_audit_refined.txt \
  --min-score 0.45

# 5. Extract specific subtopic
python sear.py extract "JWT token validation expiration" \
  --corpus backend \
  --output jwt_details.txt \
  --min-score 0.4
```

### Research Workflow

```bash
# Phase 1: Broad exploration
python sear.py extract "machine learning neural networks" \
  --min-score 0.25 \
  --output ml_broad.txt

# Phase 2: Focused extraction on interesting topics
python sear.py extract "transformer attention mechanism" \
  --min-score 0.4 \
  --output transformers.txt

# Phase 3: Deep dive on specific implementation
python sear.py extract "multi-head self-attention implementation" \
  --min-score 0.45 \
  --output attention_impl.txt \
  --max-chunks 20
```

### Extract-and-Refine Workflow

**For mixed-content documents**: Extract → Re-index → Search focused corpus

```bash
# Scenario: Large repository with mixed backend/frontend/DevOps content
# Goal: Create focused backend corpus for cleaner searches

# Step 1: Extract all backend-related content
python sear.py extract "backend API database models controllers services" \
  --corpus full_repo \
  --output backend_extracted.txt \
  --min-score 0.3

# Step 2: Index extracted content as new focused corpus
python sear.py index backend_extracted.txt backend_only

# Step 3: Now search focused corpus (no noise from frontend/DevOps)
python sear.py search "authentication middleware" --corpus backend_only

# Step 4: Can also extract from the focused corpus
python sear.py extract "authentication security" --corpus backend_only

# Benefits:
# - Higher quality results (no irrelevant chunks)
# - Faster searches (smaller index)
# - Better similarity scores (more focused embeddings)
# - Cleaner extraction outputs
```

**When to use this pattern:**
- Large mixed repositories (monorepos)
- Documentation with multiple unrelated topics
- Research papers collection (extract specific domain)
- Multi-language codebases (extract single language)
- Historical data (extract recent period only)

---

## Tips for Success

1. **Start broad, then narrow**: Begin with general queries, then drill down into specific topics

2. **Use the right threshold**: Default (0.3) works well for most cases; adjust based on results quality

3. **Combine extraction with search**: Use extract for comprehensive review, search for quick answers

4. **Review and iterate**: Check extraction quality and refine query/threshold as needed

5. **Organize outputs**: Use descriptive output file names for easy reference

6. **Document your process**: Keep notes on successful queries and thresholds for future reference

7. **Create focused corpuses from extractions**: For mixed-content documents, extract relevant sections and re-index as a new corpus for cleaner, more targeted searches

   ```bash
   # Step 1: Extract security content from large mixed document
   python sear.py extract "security authentication authorization" \
     --corpus large_mixed_doc \
     --output security_only.txt \
     --min-score 0.35

   # Step 2: Index extracted content as focused corpus
   python sear.py index security_only.txt security_corpus

   # Step 3: Now search only security-relevant content
   python sear.py search "SQL injection prevention" --corpus security_corpus
   ```

---

**Happy extracting! For questions or issues, see the main [README.md](README.md).**
