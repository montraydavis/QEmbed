# Installation Guide

This guide will help you install QEmbed and set up your development environment.

## 📋 Prerequisites

Before installing QEmbed, make sure you have:

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **PyTorch 1.9 or higher** (PyTorch 2.0+ recommended)
- **CUDA 11.0 or higher** (optional, for GPU acceleration)
- **Git** (for development installation)

## 🚀 Quick Installation

### From PyPI (Recommended)

The easiest way to install QEmbed is using pip:

```bash
# Install the latest stable version
pip install qembed

# Install with development dependencies
pip install qembed[dev]

# Install with examples and notebooks
pip install qembed[examples]
```

### From Source

If you want to install from source or contribute to development:

```bash
# Clone the repository
git clone https://github.com/qembed/qembed.git
cd qembed

# Install in development mode
pip install -e ".[dev]"
```

## 🔧 Detailed Installation Steps

### 1. Python Environment Setup

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv qembed_env

# Activate virtual environment
# On Windows:
qembed_env\Scripts\activate
# On macOS/Linux:
source qembed_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. PyTorch Installation

Install PyTorch according to your system:

```bash
# CPU only (recommended for beginners)
pip install torch torchvision torchaudio

# CUDA 11.8 (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (for latest GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Note**: For the latest PyTorch installation commands, visit [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Install QEmbed

```bash
# Install QEmbed
pip install qembed

# Verify installation
python -c "import qembed; print(qembed.__version__)"
```

## 📦 Optional Dependencies

### Development Dependencies

```bash
# Install development tools
pip install qembed[dev]

# This includes:
# - pytest (testing)
# - black (code formatting)
# - isort (import sorting)
# - mypy (type checking)
# - coverage (test coverage)
```

### Example Dependencies

```bash
# Install examples and notebooks
pip install qembed[examples]

# This includes:
# - jupyter (notebooks)
# - matplotlib (plotting)
# - seaborn (statistical plots)
# - pandas (data manipulation)
```

### Full Installation

```bash
# Install everything
pip install qembed[dev,examples]

# Or install all optional dependencies
pip install qembed[all]
```

## 🐳 Docker Installation

### Using Docker

```bash
# Pull the official QEmbed image
docker pull qembed/qembed:latest

# Run a container
docker run -it --rm qembed/qembed:latest python -c "import qembed; print('QEmbed installed successfully!')"
```

### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  qembed:
    image: qembed/qembed:latest
    ports:
      - "8888:8888"
    volumes:
      - ./data:/data
      - ./notebooks:/notebooks
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then run:

```bash
docker-compose up -d
```

## 🔍 Verification

### Basic Verification

```python
# Test basic import
import qembed
print(f"QEmbed version: {qembed.__version__}")

# Test core components
from qembed.core.quantum_embeddings import QuantumEmbeddings
from qembed.models.quantum_bert import QuantumBERT
from qembed.training.quantum_trainer import QuantumTrainer

print("All core components imported successfully!")
```

### Test Suite

Run the test suite to verify everything works:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_quantum_embeddings.py
pytest tests/test_models.py
pytest tests/test_training.py

# Run with coverage
pytest --cov=qembed tests/
```

### Example Execution

```bash
# Run basic examples
python examples/basic_usage.py

# Run benchmarks
python examples/benchmarks/word_sense_disambiguation.py
```

## 🛠️ Development Setup

### For Contributors

If you plan to contribute to QEmbed:

```bash
# Clone the repository
git clone https://github.com/qembed/qembed.git
cd qembed

# Create a new branch
git checkout -b feature/your-feature-name

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests before committing
pytest tests/
```

### Code Quality Tools

```bash
# Format code
black qembed/
isort qembed/

# Type checking
mypy qembed/

# Linting
flake8 qembed/
```

## 🚨 Troubleshooting

### Common Issues

#### 1. PyTorch Installation Problems

```bash
# If you get CUDA errors, try CPU-only version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Import Errors

```bash
# Check if QEmbed is installed correctly
pip list | grep qembed

# Reinstall if necessary
pip uninstall qembed
pip install qembed
```

#### 3. CUDA Issues

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# Visit pytorch.org for the correct command
```

#### 4. Memory Issues

```bash
# Reduce batch size in examples
# Use smaller models for testing
# Consider using CPU if GPU memory is insufficient
```

### Getting Help

If you encounter issues:

1. **Check the error message** - Look for specific error details
2. **Verify your environment** - Python version, PyTorch version, CUDA version
3. **Search existing issues** - Check GitHub issues for similar problems
4. **Open a new issue** - Provide detailed error information and environment details

## 🔄 Updating QEmbed

### Update from PyPI

```bash
# Update to latest version
pip install --upgrade qembed

# Update specific version
pip install qembed==0.2.0
```

### Update from Source

```bash
# Pull latest changes
git pull origin main

# Reinstall
pip install -e ".[dev]"
```

## 📊 System Requirements

### Minimum Requirements

- **CPU**: 2+ cores
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **CPU**: 4+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 10GB+ free space
- **OS**: Ubuntu 20.04+, macOS 11+, Windows 10+

### GPU Requirements

- **CUDA**: 11.0+ (11.8+ recommended)
- **GPU**: NVIDIA GPU with compute capability 6.0+
- **VRAM**: 4GB+ (8GB+ recommended for large models)

## 🌟 Next Steps

After successful installation:

1. **Read the [Quick Start Tutorial](quickstart.md)** - Learn the basics
2. **Try the [Examples](../examples/)** - Run working code
3. **Explore the [API Reference](api_reference.md)** - Understand the full API
4. **Join the Community** - Contribute and get help

---

**Installation complete!** 🎉 Now you're ready to explore the quantum world of NLP with QEmbed.
