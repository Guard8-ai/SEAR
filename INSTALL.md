# SEAR Installation Guide

This guide covers different installation methods for SEAR (Summarization-Enhanced Augmented Retrieval).

## Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) installed and running
- (Optional) CUDA toolkit for GPU acceleration

## Installation Methods

### Method 1: Install as Python Package (Recommended)

This method installs SEAR as a proper Python package with the `sear` command available system-wide.

#### Basic Installation (CPU)

```bash
# Clone the repository
git clone https://github.com/Guard8ai/sear.git
cd sear

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Download Ollama models
ollama pull all-minilm      # 45MB - embedding model
ollama pull qwen2.5:0.5b    # 397MB - generation model
```

After installation, you can use the `sear` command directly:

```bash
sear list
sear gpu-info
sear index myfile.txt corpus-name
sear search "your query"
```

#### Installation with GPU Support

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with GPU support (requires CUDA toolkit)
pip install -e ".[gpu]"

# Download Ollama models
ollama pull all-minilm qwen2.5:0.5b

# Verify GPU is available
sear gpu-info
```

#### Installation with Anthropic Claude Support

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with Anthropic support
pip install -e ".[anthropic]"

# Set API key
export ANTHROPIC_API_KEY=sk-ant-xxx

# Use with Anthropic provider
sear search "query" --provider anthropic
```

#### Installation with All Features

```bash
# Install with both GPU and Anthropic support
pip install -e ".[gpu,anthropic]"
```

### Method 2: Manual Installation

If you prefer not to install SEAR as a package, you can install dependencies manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy>=1.24.0 faiss-cpu>=1.7.4

# For GPU support
pip install numpy>=1.24.0 faiss-gpu>=1.7.4

# For Anthropic support
pip install anthropic>=0.39.0

# Download Ollama models
ollama pull all-minilm qwen2.5:0.5b
```

With manual installation, use the Python scripts directly:

```bash
python sear.py list
python sear.py index myfile.txt corpus-name
python sear.py search "your query"
```

### Method 3: Development Installation

For contributing to SEAR or development:

```bash
# Clone repository
git clone https://github.com/Guard8ai/sear.git
cd sear

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black .
ruff check .
```

## Verifying Installation

### Check Package Installation

```bash
# Verify sear is installed
pip show sear

# Check command availability
which sear
sear --help
```

### Check Dependencies

```bash
# Verify Python packages
python -c "import numpy, faiss; print('✓ Dependencies OK')"
python -c "from sear_core import search, index_file; print('✓ SEAR core OK')"
```

### Check Ollama

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check if models are installed
ollama list
```

Should show:
```
NAME                ID              SIZE    MODIFIED
all-minilm:latest   <id>            45 MB   <date>
qwen2.5:0.5b        <id>            397 MB  <date>
```

### Check GPU Support (Optional)

```bash
# Check FAISS GPU support
python -c "import faiss; print(f'GPU available: {hasattr(faiss, \"StandardGpuResources\")}')"
python -c "import faiss; print(f'Number of GPUs: {faiss.get_num_gpus() if hasattr(faiss, \"get_num_gpus\") else 0}')"

# Or use SEAR's built-in check
sear gpu-info
```

## Post-Installation

### Create Your First Corpus

```bash
# Index a text file
echo "Sample content for testing" > test.txt
sear index test.txt test-corpus

# List corpuses
sear list

# Search
sear search "testing"
```

### Optional: Install gitingest

For extracting repositories into text files:

```bash
pip install gitingest

# Extract a repository
gitingest https://github.com/user/repo -o repo.txt
sear index repo.txt repo-corpus
```

## Troubleshooting

### Issue: `sear: command not found`

**Solution**: Make sure you're in the virtual environment and installed with `pip install -e .`

```bash
source venv/bin/activate
which sear  # Should show path in venv
```

If still not found, use `python sear.py` instead.

### Issue: `ModuleNotFoundError: No module named 'faiss'`

**Solution**: Install FAISS

```bash
pip install faiss-cpu  # or faiss-gpu
```

### Issue: GPU not detected

**Solution**: Check CUDA installation and FAISS version

```bash
# Check CUDA
nvcc --version

# Uninstall CPU version and install GPU version
pip uninstall faiss-cpu
pip install faiss-gpu

# Verify
sear gpu-info
```

### Issue: Ollama connection error

**Solution**: Ensure Ollama is running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve

# Pull required models
ollama pull all-minilm
ollama pull qwen2.5:0.5b
```

### Issue: `anthropic` module not found

**Solution**: Install Anthropic SDK

```bash
pip install anthropic
# or
pip install -e ".[anthropic]"
```

## Uninstallation

### If Installed as Package

```bash
pip uninstall sear
```

### Complete Cleanup

```bash
# Remove package
pip uninstall sear

# Remove corpus data
rm -rf ~/.sear/

# Remove virtual environment
deactivate
rm -rf venv/
```

## Upgrading

### Upgrade Package Installation

```bash
source venv/bin/activate
pip install -e . --upgrade --upgrade-strategy eager
```

### Update Ollama Models

```bash
ollama pull all-minilm
ollama pull qwen2.5:0.5b
```

## Next Steps

- Read the [README.md](README.md) for usage examples
- Check [examples/](examples/) for programmatic usage
- See [EXTRACTION_GUIDE.md](EXTRACTION_GUIDE.md) for extraction patterns
- Review [GPU_SUPPORT.md](GPU_SUPPORT.md) for GPU optimization
- Check [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for performance data

## Getting Help

- **Issues**: https://github.com/Guard8ai/sear/issues
- **Documentation**: https://github.com/Guard8ai/sear#readme
