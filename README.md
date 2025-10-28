# SEAR: Summarization-Enhanced Augmented Retrieval

**Multi-corpus RAG with FAISS retrieval → LLM summarization. 100% deterministic, line-level source tracking, zero cost.**

```
Query → Embed → Parallel FAISS Search → Top Chunks → LLM Synthesis → Answer + Sources
                (multi-corpus, ~1s)                  (Ollama, ~23s)   [corpus] file:lines
```

## Why SEAR vs Traditional RAG?

| Feature | Traditional RAG | SEAR |
|---------|-----------------|------|
| **Cost** | $0.50/query (46k tokens) | **$0.00** (local Ollama) |
| **Determinism** | ❌ Non-deterministic | ⚠️ **Retrieval: 100%**, Generation: ~95% |
| **Quality Filtering** | ❌ Uses all top-k results | ✅ **Empirical score thresholds** (default: 0.30) |
| **Source Tracking** | Chunk-level | ✅ **Line-level** + corpus names |
| **Multi-Corpus** | Single index | ✅ Parallel independent indices |
| **Offline** | API-dependent | ✅ Fully local |
| **Token Efficiency** | Sends everything | **99% reduction** (retrieval-first) |
| **Content Extraction** | ❌ Not available | ✅ **Extract ALL relevant chunks without LLM** |

**Quality-Aware RAG**: SEAR filters results using empirically-determined thresholds (default: 0.30). When quality is insufficient (<2 results), it prompts for query refinement instead of generating answers from noise. Traditional RAG blindly uses all top-k results, including low-quality matches.

**Note on Determinism**: FAISS retrieval is 100% deterministic (identical sources every time). LLM generation with `temp=0, seed=42` is highly consistent (~95%) but not perfectly deterministic due to floating-point arithmetic and model internals.

## Quick Start

### 1. Install

**Option A: Install as Python Package (Recommended)**
```bash
python3 -m venv venv && source venv/bin/activate

# CPU installation (default)
pip install -e .
ollama pull all-minilm qwen2.5:0.5b  # 45MB + 397MB

# Or with GPU support (requires CUDA toolkit)
pip install -e ".[gpu]"
ollama pull all-minilm qwen2.5:0.5b

# Or with Anthropic Claude support
pip install -e ".[anthropic]"

# Or with document conversion support (PDF/DOCX to markdown)
pip install -e ".[converter]"

# Or install everything
pip install -e ".[all]"

# Check GPU availability
sear gpu-info
```

**Option B: Manual Installation**
```bash
python3 -m venv venv && source venv/bin/activate

# CPU installation
pip install faiss-cpu numpy
ollama pull all-minilm qwen2.5:0.5b

# GPU installation (requires CUDA toolkit)
pip install faiss-gpu numpy
ollama pull all-minilm qwen2.5:0.5b

# Check GPU availability
python sear.py gpu-info
```

### 2. Prepare Corpuses

**SEAR operates on `.txt` or `.md` files.** Prepare your data:

```bash
# Option A: Convert documents (PDF/DOCX) to markdown
sear convert document.pdf                    # Converts to converted_md/document.md
sear convert report.docx --output-dir docs/  # Custom output directory
sear convert scanned.pdf --force-ocr --lang heb+eng  # OCR with language hint

# Option B: Git repos (requires gitingest)
pip install gitingest
gitingest https://github.com/user/repo -o repo.txt

# Option C: Local code/docs
cat docs/*.md > documentation.txt
find ./src -name "*.py" -exec cat {} \; > codebase.txt
```

**Document Conversion Features:**
- **Multi-format**: PDF (with OCR fallback) and DOCX (Microsoft Word)
- **Smart OCR**: Auto-detects text layer, uses fast extraction or OCR
- **Language support**: Hebrew, English, mixed content (auto-detected)
- **LLM-optimized**: Removes niqqud, styling, formatting that waste tokens
- **RAG-ready**: Metadata headers + page separators for citations

**Installation:**
```bash
# Option A: Using pyproject.toml extras (recommended)
pip install -e ".[converter]"

# Option B: Using requirements file
pip install -r requirements-converter.txt

# Tesseract OCR (required for PDF OCR support)
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-heb tesseract-ocr-eng
```

### 3. Index and Search
```bash
# If installed as package, use 'sear' command directly:
sear index converted_md/document.md technical-docs
sear index repo.txt backend
sear list  # Show all corpuses

# Or use python sear.py if not installed as package:
# python sear.py index converted_md/document.md

# Search (all corpuses by default)
sear search "authentication patterns"

# Search with GPU acceleration (auto-detected if available)
sear search "authentication patterns" --gpu

# Search specific corpuses
sear search "API endpoints" --corpus backend,documentation

# Force CPU mode
sear search "authentication patterns" --no-gpu

# Output with line-level sources:
# 📍 SOURCES
# 1. [backend] repo.txt:557-659 (score: 0.856)
# 2. [documentation] docs.txt:3016-3162 (score: 0.742)
```

### 4. Extract Relevant Content (New!)
**Extract ALL matching content without LLM processing** - reduces cognitive load:

```bash
# Extract all relevant content above threshold
sear extract "authentication security validation"

# Save to specific file
sear extract "testing methodologies" --output testing_guide.txt

# Extract from specific corpuses with custom threshold
sear extract "neural networks transformers" --corpus ml-docs --min-score 0.4

# Limit extraction size
sear extract "API security" --max-chunks 50

# Use GPU for large extractions
sear extract "security audit" --max-chunks 200 --gpu
```

**Why Extract vs Search?**
- **No LLM hallucination**: Raw extracted content, no AI interpretation
- **Complete context**: Get ALL relevant chunks (not just top-k)
- **Human-readable**: Review full sections in original document order
- **Exportable**: Create focused documents for specific topics
- **Reduced cognitive load**: Structured extraction vs parsing LLM output
- **Audit trail**: Know exactly what was extracted and from where

**Pro Tip**: For mixed-content documents, extract relevant sections first, then re-index as a new focused corpus for cleaner search results.

See [EXTRACTION_GUIDE.md](EXTRACTION_GUIDE.md) for detailed usage patterns.

**Organize corpuses by:** project, type (code/docs/specs), domain (backend/frontend), or version (v1/v2).

## Architecture

**Two-Stage RAG Pipeline:**

1. **Retrieval** (~1s CPU, ~0.1-0.3s GPU): Query → Embedding → Parallel FAISS search → Top K chunks (ranked by similarity)
2. **Summarization** (~23s): Retrieved chunks → Ollama qwen2.5:0.5b (`temp=0, seed=42`) → Answer

**Key Innovations:**
- **Line-level tracking**: Each chunk stores `(start_line, end_line)` for verification
- **Corpus attribution**: Results show `[corpus-name] file.txt:557-659 (score: 0.856)`
- **High determinism**: FAISS retrieval is 100% deterministic; LLM generation (~95% consistent with `temp=0, seed=42`)
- **Compatibility validation**: Auto-checks embedding model, dimension, normalization across corpuses
- **GPU acceleration**: Automatic GPU detection and usage when CUDA + faiss-gpu available

## Document Conversion Workflow

**Complete PDF/DOCX to RAG pipeline:**

```bash
# 1. Convert documents to markdown
sear convert financial-report.pdf
sear convert legal-contract.docx
sear convert mixed-content.pdf --lang heb+eng

# 2. Index the converted markdown
sear index converted_md/financial-report.md finance
sear index converted_md/legal-contract.md legal
sear index converted_md/mixed-content.md docs

# 3. Search across all documents
sear search "What are the financial obligations?"
sear search "תנאים משפטיים"  # Hebrew queries supported
```

**Conversion Options:**
- `--output-dir DIR`: Specify output directory (default: `converted_md/`)
- `--no-normalize`: Keep original formatting (niqqud, styling)
- `--force-ocr`: Force OCR even if text layer exists (PDF only)
- `--lang heb|eng|heb+eng`: Specify OCR language (PDF only, auto-detected by default)

**Performance:**
- PDF with text layer: ~14 pages/s (fast extraction)
- Scanned PDF: ~0.16 pages/s (OCR)
- DOCX: ~50 pages/s (native extraction)

## Use Cases

**Ideal for:** Code analysis, security scanning, IDE tools, compliance systems, privacy-sensitive environments, cost-conscious applications, **document search (PDFs/DOCX)**

**Not ideal for:** Creative content generation, conversational AI (no memory), complex multi-source synthesis

## Customization

```python
# Embedding model (in sear_core.py)
payload = {"model": "all-minilm", "prompt": text}  # Current: 45MB, fast
# payload = {"model": "nomic-embed-text", "prompt": text}  # Better quality
# payload = {"model": "mxbai-embed-large", "prompt": text}  # Highest quality

# Similarity metric
index = faiss.IndexFlatIP(dimension)  # Cosine similarity (current)
# index = faiss.IndexFlatL2(dimension)  # L2 distance

# Chunk size: Auto-detected from embedding model's context length
```

## Performance

**CPU Mode** (Default):
- **Indexing** (99k token corpus): ~2 minutes (chunking <1s, embedding 1186 chunks ~120s)
- **Query** (end-to-end): ~24s (retrieval ~1s + LLM generation ~23s)
- **Memory**: ~450MB total (Ollama models: 442MB, indices: ~2MB/corpus)

**GPU Mode** (With CUDA + faiss-gpu):
- **Small corpuses (<500 chunks)**: CPU faster due to GPU overhead
- **Medium corpuses (1k-10k chunks)**: 2-5x faster indexing/search
- **Large corpuses (10k+ chunks)**: 3-10x faster, especially for repeated searches
- **Auto-detection**: GPU automatically used when available (see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md))
- **Tested on**: RTX 4060 with CUDA 13.0, faiss-gpu 1.12.0

**Cost Savings** (1000 queries):
- Traditional RAG: 46k tokens/query × $0.50 = **$500**
- SEAR: Local Ollama = **$0** (100% savings)

## Advanced

### Optimization

**GPU Acceleration:**
```bash
# Check GPU availability
sear gpu-info

# Force GPU usage (skip auto-detection)
sear index large-repo.txt --gpu
sear search "query" --gpu

# Disable GPU (use CPU only)
sear search "query" --no-gpu
```

**LLM Provider Options:**
```bash
# Local Ollama (default, fastest)
sear search "query"  # Uses qwen2.5:0.5b (~5s)

# Anthropic Claude 3.5 Sonnet 4.5 (higher quality, slower)
export ANTHROPIC_API_KEY=sk-ant-xxx
sear search "query" --provider anthropic  # ~10s, better reasoning

# Alternative local models
ollama pull llama3.2:1b     # 1.3GB, faster than qwen2.5:0.5b
ollama pull phi3:mini       # 2.3GB, better quality/speed
```

**Scale to 100k+ chunks (approximate search):**
```python
# In sear_core.py for very large indices
index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist=100)
```

### Batch Indexing
```bash
for repo in repo1 repo2 repo3; do
    gitingest https://github.com/$repo -o ${repo}.txt
    sear index ${repo}.txt
done
```

## Implementation

**File Structure:**
- `sear.py` - CLI interface (~335 lines)
- `sear_core.py` - Core library (~1,280 lines)
- `~/.sear/<corpus>/` - Indices (index.faiss, chunks.pkl, metadata.json)

**Line Tracking:**
1. Chunking: Text → chunks with `(start_line, end_line)` metadata
2. Retrieval: FAISS returns chunk index → lookup line range
3. Output: Display `file.txt:557-659` for verification

**Why This Works:**
- FAISS retrieval: 99% token reduction (46k → 1k), 100% deterministic
- `temperature=0, seed=42`: ~95% consistent generation (verified empirically)
- Result: Same quality, $0 cost, highly reproducible

**Determinism Verification** (tested 2025-10-22):
- Query: "how does gitingest handle file exclusion patterns"
- Run 1 vs Run 2: Identical sources (5/5 chunks), similar but not identical answers
- Retrieval stage: 100% deterministic (same chunks, scores, line numbers)
- Generation stage: ~95% consistent (key facts preserved, phrasing varies slightly)

## Library Usage

```python
from sear_core import (
    index_file, search, extract_relevant_content, list_corpuses,
    validate_corpus_compatibility, is_gpu_available
)

# Check GPU availability
if is_gpu_available():
    print("GPU acceleration available!")

# Index and search with GPU support
index_file("code.txt", corpus_name="backend", use_gpu=True)
results = search("authentication", corpuses=["backend"],
                verbose=False, use_gpu=True)

# Access search results
print(results['answer'])
for src in results['sources']:
    print(f"[{src['corpus']}] {src['location']} (score: {src['score']})")

# Extract relevant content (no LLM processing)
extraction = extract_relevant_content(
    query="security testing authentication",
    corpuses=["backend"],
    output_file="security_content.txt",
    min_score=0.3,
    max_chunks=100,
    use_gpu=True,
    verbose=False
)

print(f"Extracted {extraction['total_chunks']} chunks")
print(f"Score range: {extraction['score_range']}")
print(f"Saved to: {extraction['output_file']}")

# Validate compatibility
validate_corpus_compatibility(["backend", "frontend"])
```

See [examples/](examples/) for more patterns (library_usage.py, batch_indexing.py, custom_pipeline.py).

## References

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [FAISS by Meta AI](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.ai/)
- [gitingest](https://github.com/coderamp-labs/gitingest)

---

**SEAR: Because not everything needs generation.**
