# Basic SEAR Workflow Example

## Scenario: Search Technical Documentation

You have technical documentation in markdown files and want to search them semantically.

## Step-by-Step

### 1. Prepare Your Documents

```bash
# Example: Concatenate documentation files
cat docs/*.md > documentation.txt

# Or use existing text files directly
ls -lh documentation.txt
# -rw-r--r-- 1 user user 2.4M Oct 28 2025 documentation.txt
```

### 2. Index the Documentation

```bash
# Create a searchable index
sear index documentation.txt tech_docs

# Output:
# Indexing documentation.txt into corpus 'tech_docs'...
# Created 1,234 chunks from 1 file
# Indexed successfully (CPU mode, 8.2 seconds)
# Corpus 'tech_docs' is ready for search
```

**What happened:**
- SEAR split the document into semantic chunks
- Each chunk was embedded using all-minilm (384 dimensions)
- FAISS index created at `faiss_indices/tech_docs/`

### 3. Search with Questions

```bash
# Basic semantic search
sear search "how do I configure authentication?" --corpus tech_docs

# Output:
# 🔍 Searching corpus: tech_docs
# Query: how do I configure authentication?
#
# Retrieved 5 chunks (similarity > 0.30)
#
# Answer:
# To configure authentication, you need to set up the auth module...
# [tech_docs] documentation.txt:142-156
#
# The authentication system supports OAuth2 and JWT tokens...
# [tech_docs] documentation.txt:892-905
```

**What happened:**
- Query was embedded using the same model
- FAISS found the 5 most similar chunks (above threshold 0.30)
- Local LLM (qwen2.5:0.5b) synthesized an answer
- Citations show exact file and line numbers

### 4. Extract Content Without LLM

```bash
# Get raw chunks for manual review
sear extract "security best practices" --corpus tech_docs

# Output:
# 🔍 Extracting from corpus: tech_docs
# Query: security best practices
#
# === Chunk 1 (score: 0.82) ===
# Source: [tech_docs] documentation.txt:1024-1038
#
# Security Best Practices
# =======================
#
# 1. Always use HTTPS in production
# 2. Enable rate limiting on all API endpoints
# 3. Implement proper input validation
# ...
#
# === Chunk 2 (score: 0.76) ===
# Source: [tech_docs] documentation.txt:2134-2150
# ...
```

**What happened:**
- Same retrieval as search, but no LLM synthesis
- Just raw chunks with scores and sources
- Useful for manual analysis or further processing

## Tips

### Adjust Quality Threshold

```bash
# Stricter matching (higher threshold)
sear search "query" --corpus tech_docs --min-score 0.40

# Broader matching (lower threshold)
sear search "query" --corpus tech_docs --min-score 0.20
```

### Use Better LLM for Critical Questions

```bash
# Use Anthropic Claude for higher quality
export ANTHROPIC_API_KEY=sk-ant-xxx
sear search "complex question" --corpus tech_docs --provider anthropic
```

### Check Available Corpuses

```bash
# List all indexed corpuses
sear list

# Output:
# Available corpuses:
#   - tech_docs (1,234 chunks, 2.4 MB)
#   - api_docs (567 chunks, 1.1 MB)
```

## Expected Results

**Speed:**
- Indexing: ~8-10 seconds for 2.4MB file (CPU)
- Search: ~5-6 seconds total (0.8s retrieval + 4-5s generation)

**Quality:**
- Retrieval: 100% deterministic (same sources every time)
- Generation: ~95% consistent with temp=0

**Cost:**
- $0 (fully local with Ollama)

## Next Steps

- Try [PDF Conversion Example](pdf-conversion.md)
- Learn [Multi-Corpus Search](multi-corpus-search.md)
- See main docs: [README.md](../../README.md)
