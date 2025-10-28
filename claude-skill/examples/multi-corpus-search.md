# Multi-Corpus Search Example

## Scenario: Search Across Multiple Knowledge Sources

You have different types of documents (code, docs, research papers) and want to search all of them simultaneously.

## Why Multi-Corpus?

**Problem with single corpus:**
- Mixing different document types reduces retrieval quality
- Code, documentation, and research have different structures
- Hard to track which source type provided each result

**Multi-corpus solution:**
- Keep document types separate
- Search multiple corpuses in parallel
- Get source-labeled results: `[codebase] file.py:42` vs `[docs] guide.md:156`

## Step-by-Step

### 1. Prepare Different Document Types

```bash
# Collect codebase
find ./src -name "*.py" -exec cat {} \; > codebase.txt

# Collect documentation
cat docs/*.md > documentation.txt

# Convert research papers
sear convert papers/*.pdf --output-dir research/
cat research/*.md > research_papers.txt
```

### 2. Index Each Corpus Separately

```bash
# Index codebase
sear index codebase.txt code_corpus
# Created 2,341 chunks

# Index documentation
sear index documentation.txt docs_corpus
# Created 567 chunks

# Index research papers
sear index research_papers.txt papers_corpus
# Created 892 chunks
```

**Result:** Three independent FAISS indices:
```
faiss_indices/
├── code_corpus/
├── docs_corpus/
└── papers_corpus/
```

### 3. Search Across All Corpuses

```bash
# Multi-corpus search
sear search "how does authentication work?" \
  --corpus code_corpus \
  --corpus docs_corpus \
  --corpus papers_corpus

# Output:
# 🔍 Searching 3 corpuses: code_corpus, docs_corpus, papers_corpus
# Query: how does authentication work?
#
# Retrieved 12 chunks total (similarity > 0.30)
#   - code_corpus: 5 chunks
#   - docs_corpus: 4 chunks
#   - papers_corpus: 3 chunks
#
# Answer:
# Authentication is implemented using JWT tokens. The login endpoint
# validates credentials and generates a signed token.
# [code_corpus] auth/login.py:142-156
#
# The documentation explains that tokens expire after 24 hours and
# can be refreshed using the /refresh endpoint.
# [docs_corpus] api_guide.md:89-102
#
# Research shows that JWT-based authentication provides better
# scalability than session-based approaches.
# [papers_corpus] security_patterns.md:456-470
```

**Key benefits:**
- Results from all sources combined
- Clear source labels for each chunk
- LLM synthesizes answer from diverse sources

### 4. Search Specific Subset

```bash
# Search only code and docs (skip research)
sear search "implementation details" \
  --corpus code_corpus \
  --corpus docs_corpus

# Search only research papers
sear search "theoretical background" \
  --corpus papers_corpus
```

## Advanced Patterns

### Prioritized Multi-Corpus Search

```bash
# Primary source: documentation
# Secondary source: code (for examples)
sear search "API usage guide" \
  --corpus docs_corpus \
  --corpus code_corpus \
  --min-score 0.35  # Higher threshold for quality
```

### Domain-Specific Corpuses

```bash
# Organize by domain/project
sear index frontend_code.txt frontend_corpus
sear index backend_code.txt backend_corpus
sear index mobile_code.txt mobile_corpus

# Search specific domain
sear search "state management" --corpus frontend_corpus

# Or search all domains
sear search "API integration" \
  --corpus frontend_corpus \
  --corpus backend_corpus \
  --corpus mobile_corpus
```

### Version-Specific Corpuses

```bash
# Index different versions separately
sear index docs_v1.txt docs_v1_corpus
sear index docs_v2.txt docs_v2_corpus
sear index docs_v3.txt docs_v3_corpus

# Compare implementations across versions
sear search "authentication method" \
  --corpus docs_v1_corpus \
  --corpus docs_v2_corpus \
  --corpus docs_v3_corpus
```

## Corpus Management

### List All Corpuses

```bash
sear list

# Output:
# Available corpuses:
#   - code_corpus (2,341 chunks, 4.2 MB, GPU)
#   - docs_corpus (567 chunks, 1.1 MB, CPU)
#   - papers_corpus (892 chunks, 2.8 MB, CPU)
#   - frontend_corpus (1,234 chunks, 2.1 MB, GPU)
#   - backend_corpus (1,567 chunks, 3.4 MB, GPU)
```

### Delete Unused Corpuses

```bash
# Remove old corpus
sear delete old_corpus

# Re-index with updated content
sear index updated_docs.txt docs_corpus  # Overwrites existing
```

### Update Corpuses

```bash
# When documents change, re-index
cat docs/*.md > documentation.txt
sear index documentation.txt docs_corpus  # Updates existing corpus
```

## Best Practices

### 1. Organize by Document Type

```bash
# Good: Separate by type
✓ code_corpus (Python files)
✓ docs_corpus (Markdown docs)
✓ api_corpus (API specifications)
✓ papers_corpus (Research papers)

# Bad: Mixed types
✗ everything_corpus (code + docs + papers)
```

### 2. Use Meaningful Names

```bash
# Good: Clear, descriptive
✓ auth_module_code
✓ user_facing_docs
✓ internal_specs

# Bad: Generic names
✗ corpus1
✗ data
✗ temp
```

### 3. Size Considerations

```bash
# Small corpuses (<500 chunks): Use CPU
sear index small.txt small_corpus --no-gpu

# Large corpuses (>1000 chunks): Use GPU
sear index large.txt large_corpus --gpu

# Let SEAR decide
sear index medium.txt medium_corpus  # Auto-selects
```

### 4. Threshold Tuning

```bash
# Multi-corpus searches may need higher thresholds
sear search "query" \
  --corpus corpus1 \
  --corpus corpus2 \
  --corpus corpus3 \
  --min-score 0.35  # Slightly higher than default 0.30
```

## Performance Tips

### Parallel Indexing (for speed)

```bash
# Index corpuses in parallel
sear index code.txt code_corpus --gpu &
sear index docs.txt docs_corpus --no-gpu &
sear index papers.txt papers_corpus --gpu &
wait

echo "All corpuses indexed!"
```

### Selective Searching

```bash
# Don't always search everything
# If question is code-specific:
sear search "implementation" --corpus code_corpus

# If question is conceptual:
sear search "theory" --corpus papers_corpus

# If question is broad:
sear search "overview" \
  --corpus code_corpus \
  --corpus docs_corpus \
  --corpus papers_corpus
```

## Example: Complete Multi-Source Setup

```bash
#!/bin/bash
# setup_knowledge_base.sh

echo "Building multi-corpus knowledge base..."

# 1. Collect sources
echo "Collecting sources..."
find ./src -name "*.py" > codebase.txt
cat docs/*.md > documentation.txt
sear convert papers/*.pdf --output-dir research/
cat research/*.md > research.txt

# 2. Index corpuses
echo "Indexing corpuses..."
sear index codebase.txt code --gpu
sear index documentation.txt docs --no-gpu
sear index research.txt papers --gpu

# 3. Verify
echo "Setup complete!"
sear list

echo ""
echo "Try searching:"
echo "  sear search 'your question' --corpus code --corpus docs --corpus papers"
```

## Expected Results

**Search Speed:**
- Single corpus: ~0.8s retrieval
- Multi-corpus (3): ~2.4s retrieval (linear scaling)
- Generation: Same (~5s with Ollama)

**Quality Benefits:**
- Diverse perspectives in answers
- Clear source attribution
- Better coverage of topics

**Management:**
- Easy to update individual corpuses
- Clear organization
- Flexible search strategies

## Next Steps

- See [Basic Workflow](basic-workflow.md)
- Try [PDF Conversion](pdf-conversion.md)
- Read main docs: [README.md](../../README.md)
