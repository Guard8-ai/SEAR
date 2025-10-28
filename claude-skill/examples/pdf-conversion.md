# PDF Conversion and Search Example

## Scenario: Research Paper Analysis

You have research papers in PDF format and want to search them semantically.

## Step-by-Step

### 1. Convert PDFs to Markdown

```bash
# Single PDF conversion
sear convert research_paper.pdf

# Output:
# Converting research_paper.pdf...
# Detected text layer, using fast extraction
# Processing 24 pages...
# Saved to: converted_md/research_paper.md
# Conversion complete (3.2 seconds)
```

**What happened:**
- doc-converter detected the PDF has text (not scanned)
- Extracted text with pdfplumber (fast method)
- Converted to LLM-optimized markdown
- Added metadata headers and page separators

**Output file:** `converted_md/research_paper.md`
```markdown
# Document Metadata
- Source: research_paper.pdf
- Pages: 24
- Converted: 2025-10-28

---PAGE 1---

# Introduction

Machine learning approaches to natural language processing...

---PAGE 2---

## Related Work

Previous research has shown that transformer architectures...
```

### 2. Convert Multiple PDFs

```bash
# Convert all PDFs in a directory
for pdf in papers/*.pdf; do
    sear convert "$pdf"
done

# Or use custom output directory
sear convert papers/*.pdf --output-dir research_md/
```

### 3. Handle Scanned PDFs (OCR)

```bash
# For scanned documents without text layer
sear convert scanned_paper.pdf --force-ocr

# With language hints (Hebrew + English)
sear convert hebrew_document.pdf --force-ocr --lang heb+eng

# Output:
# Converting scanned_paper.pdf...
# No text layer detected, using OCR...
# Processing 15 pages with Tesseract...
# Saved to: converted_md/scanned_paper.md
# Conversion complete (45.3 seconds)
```

**OCR Notes:**
- Much slower than text extraction (15-20 seconds per page)
- Quality depends on scan quality
- Language hints improve accuracy

### 4. Index Converted Papers

```bash
# Index single paper
sear index converted_md/research_paper.md papers_corpus

# Or index all converted papers
cat converted_md/*.md > all_papers.txt
sear index all_papers.txt papers_corpus

# Output:
# Indexing all_papers.txt into corpus 'papers_corpus'...
# Created 456 chunks from 1 file (3 papers)
# Indexed successfully (CPU mode, 5.1 seconds)
```

### 5. Search Across Papers

```bash
# Ask questions about the research
sear search "what were the main findings?" --corpus papers_corpus

# Output:
# 🔍 Searching corpus: papers_corpus
# Query: what were the main findings?
#
# Retrieved 6 chunks (similarity > 0.30)
#
# Answer:
# The main findings indicate that transformer-based models
# outperform traditional approaches by 15-20% on benchmark tasks.
# [papers_corpus] research_paper.md:142-156 (Page 8)
#
# Additionally, the study found that fine-tuning on domain-specific
# data improves accuracy by an additional 8%.
# [papers_corpus] advanced_methods.md:89-102 (Page 4)
```

**Note:** Citations include page numbers from metadata!

### 6. Extract Specific Sections

```bash
# Extract methodology sections
sear extract "methodology experimental setup" --corpus papers_corpus

# Extract results and analysis
sear extract "results conclusions" --corpus papers_corpus
```

## Advanced Options

### Hebrew Document Processing

```bash
# Convert Hebrew PDF with normalization
sear convert hebrew_doc.pdf

# Keep original formatting (niqqud, etc.)
sear convert hebrew_doc.pdf --no-normalize

# Mixed Hebrew/English with OCR
sear convert mixed_doc.pdf --force-ocr --lang heb+eng
```

### Batch Processing Script

```bash
#!/bin/bash
# batch_process.sh - Convert and index multiple papers

echo "Converting PDFs..."
for pdf in papers/*.pdf; do
    echo "Processing: $pdf"
    sear convert "$pdf" --output-dir converted/
done

echo "Combining converted files..."
cat converted/*.md > all_research.txt

echo "Indexing corpus..."
sear index all_research.txt research_corpus --gpu

echo "Done! Ready to search:"
echo "  sear search 'your question' --corpus research_corpus"
```

### GPU Acceleration for Large Batches

```bash
# For 100+ papers, use GPU
sear index large_corpus.txt papers --gpu

# Check GPU is available first
sear gpu-info

# Output:
# GPU: NVIDIA RTX 4060 (8188 MB)
# CUDA: 13.0
# faiss-gpu: 1.12.0 ✓
```

## Conversion Quality Tips

### High-Quality PDFs (Digital)
- Fast extraction (3-5 seconds per document)
- Near-perfect text quality
- Preserves structure and formatting

### Scanned PDFs (OCR Required)
- Slower processing (15-20 seconds per page)
- Quality varies with scan quality
- Use language hints for better accuracy

### Mixed Documents
```bash
# Auto-detect: tries text extraction first, falls back to OCR
sear convert mixed_document.pdf

# Force OCR even if text layer exists (for better quality)
sear convert problematic.pdf --force-ocr
```

## Expected Results

**Conversion Speed:**
- Digital PDF: ~3-5 seconds per document
- Scanned PDF (OCR): ~15-20 seconds per page

**Search Quality:**
- Page-level citations in results
- Metadata preserved in output
- Structure maintained for better retrieval

**Cost:**
- $0 (fully local processing)

## Common Issues

### "No text layer detected"
**Solution:** Use `--force-ocr` for scanned documents

### OCR produces garbage text
**Solution:** Check scan quality, try `--lang` hints

### Hebrew text appears corrupted
**Solution:** Use `--no-normalize` to keep original formatting

## Next Steps

- Try [Multi-Corpus Search](multi-corpus-search.md)
- See [Basic Workflow](basic-workflow.md)
- Read main docs: [README.md](../../README.md)
