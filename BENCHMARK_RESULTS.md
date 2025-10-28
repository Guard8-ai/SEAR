# SEAR GPU Benchmark Results

## Test Environment

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **GPU Memory**: 8188 MiB
- **Compute Capability**: 8.9
- **CUDA**: Version 13.0 (V13.0.88)
- **Driver**: Version 580.95.05
- **CPU**: AMD/Intel x86_64
- **OS**: Linux (Arch Linux based)
- **faiss-gpu**: Version 1.12.0 (built from source)
- **Python**: 3.13.7

## Benchmark Methodology

Tests were conducted using the `test_timing.py` script which:
1. Creates a synthetic test corpus (100 sections, ~600 lines, 134 chunks)
2. Measures indexing time for both CPU and GPU
3. Measures search time for both CPU and GPU
4. Uses the same query for both modes: "testing methodologies"

## Small Corpus Results (134 chunks, ~600 lines)

| Operation | CPU Time | GPU Time | Speedup | Notes |
|-----------|----------|----------|---------|-------|
| **Indexing** | 5.53s | 5.37s | 1.03x | Marginal GPU benefit |
| **Search** | 2.95s | 4.32s | 0.68x | GPU slower due to overhead |

### Analysis

For small corpuses (< 200 chunks):
- **Indexing**: Minimal GPU benefit (~3% faster)
- **Search**: GPU is actually slower due to transfer overhead
- **Recommendation**: Use CPU mode (`--no-gpu`) for small datasets

### Why Small Datasets Don't Benefit from GPU

1. **Data Transfer Overhead**: Moving small amounts of data to/from GPU memory takes time
2. **Initialization Cost**: GPU initialization and kernel launch overhead
3. **Parallelization Inefficiency**: Small datasets don't saturate GPU cores
4. **Memory Latency**: GPU memory access patterns not optimized for tiny datasets

## Expected Performance at Scale

Based on FAISS documentation and GPU architecture:

### Medium Corpus (1,000-10,000 chunks)
- **Expected Indexing Speedup**: 2-3x faster
- **Expected Search Speedup**: 3-5x faster
- **Sweet Spot**: 5,000+ chunks

### Large Corpus (10,000+ chunks)
- **Expected Indexing Speedup**: 3-5x faster
- **Expected Search Speedup**: 5-10x faster
- **Optimal Performance**: 50,000+ chunks

## GPU Acceleration Characteristics

### When GPU Provides Benefits:
- Large vector batches (1000+ vectors)
- High-dimensional embeddings (384+ dimensions in our case)
- Repeated searches on the same index
- Multiple concurrent queries

### When CPU is Better:
- Small corpuses (< 500 chunks)
- One-off operations
- Limited GPU memory scenarios
- Single query operations

## Technical Details

### Build Configuration
- Compiled from source (facebook/faiss main branch)
- CUDA architecture: 89 (RTX 4060)
- GPU support: Enabled with CUDA 13.0
- BLAS: System BLAS (/usr/lib/libblas.so)
- LAPACK: System LAPACK (/usr/lib/liblapack.so)

### GPU Utilization
The NVIDIA RTX 4060 Laptop GPU has:
- **CUDA Cores**: ~3072
- **Tensor Cores**: Available for mixed precision
- **Memory Bandwidth**: ~192 GB/s
- **L2 Cache**: 32 MB

For the small test corpus, GPU utilization was minimal (<10%), indicating that the dataset size is insufficient to leverage the GPU's parallel processing capabilities.

## Recommendations

### For Development/Testing:
```bash
# Use CPU for small test corpuses
python sear.py index test.txt test_corpus --no-gpu
python sear.py search "query" --corpus test_corpus --no-gpu
```

### For Production/Large Datasets:
```bash
# Let SEAR automatically use GPU when beneficial
python sear.py index large_doc.txt production_corpus

# Or explicitly enable GPU
python sear.py index large_doc.txt production_corpus --gpu
python sear.py search "query" --corpus production_corpus --gpu
```

### Best Practices:
1. **Automatic Mode**: Don't specify `--gpu` or `--no-gpu`, let SEAR decide
2. **Batch Operations**: Index multiple files in one operation for better GPU utilization
3. **Memory Considerations**: GPU has limited memory (8GB), very large corpuses may need CPU
4. **Query Batching**: Multiple queries benefit more from GPU than single queries

## Future Improvements

Potential optimizations for better GPU performance:

1. **Index Caching**: Keep frequently-used indices on GPU memory
2. **Batch Query Support**: Allow multiple simultaneous queries
3. **Mixed Mode**: Use GPU for search, CPU for indexing on small datasets
4. **Adaptive Switching**: Automatically switch between CPU/GPU based on dataset size

## LLM Provider Benchmarks

Tested on `gitingest` corpus (1186 chunks) with query: "how does gitingest handle file exclusion patterns"

| Provider | Model | Time | Quality | Cost | Notes |
|----------|-------|------|---------|------|-------|
| **Ollama** (default) | qwen2.5:0.5b | **5.3s** | Good | $0 | Local, fast, adequate quality |
| **Anthropic** | Claude 3.5 Sonnet 4.5 | **10.5s** | Excellent | ~$0.01/query | Better reasoning, structured output |

### Analysis

- **Retrieval phase**: Identical (~0.8s with GPU)
- **Generation phase**: Ollama ~4.5s vs Anthropic ~9.7s
- **Quality difference**: Anthropic provides more structured analysis, acknowledges limitations, better synthesis
- **Use case**:
  - Ollama: Fast iteration, development, cost-sensitive
  - Anthropic: Production, critical analysis, higher quality requirements

### Provider Selection

```bash
# Local Ollama (default)
python sear.py search "query"

# Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-xxx
python sear.py search "query" --provider anthropic

# Or pass API key directly
python sear.py search "query" --provider anthropic --api-key sk-ant-xxx
```

## Verification Commands

To reproduce these benchmarks:

```bash
# Check GPU availability
python sear.py gpu-info

# Run timing tests
python test_timing.py

# Test LLM providers
time python sear.py search "query" --provider ollama
time python sear.py search "query" --provider anthropic

# Manual testing
python sear.py index test.txt test_cpu --no-gpu
python sear.py index test.txt test_gpu --gpu
time python sear.py search "query" --corpus test_cpu --no-gpu
time python sear.py search "query" --corpus test_gpu --gpu
```

---

**Last Updated**: 2025-10-23
**SEAR Version**: 1.1.0
**Test Date**: October 23, 2025
