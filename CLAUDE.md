# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project focused on document loading and processing with custom text splitting and search capabilities. The project supports reading various document formats (TXT, PDF, URLs) and provides advanced text processing features including multiple splitting strategies and search modes.

## Project Structure

```text
Document-Loader/
├── main.py                 # Main application entry point
├── text_splitter.py        # Text splitting module with various strategies
├── embeddings.py           # Vector embeddings and search functionality
├── README.md              # Project documentation (in Chinese)
├── LICENSE                # MIT License
├── CLAUDE.md              # This file
└── .venv/                 # Virtual environment (managed by uv)
```

### Core Modules

#### `main.py` - Application Entry Point

- Command-line interface using argparse
- Document loading from TXT, PDF, and URL sources
- Text splitting and search functionality
- Import organization with clear separation of standard libraries, third-party packages, and custom modules

#### `text_splitter.py` - Text Processing Module

- **TextSplitter**: Base class for all text splitters
- **CharacterTextSplitter**: Character-based splitting with separator support
- **RecursiveCharacterTextSplitter**: Iterative splitting using multiple separators
- **StreamingTextSplitter**: Memory-efficient streaming splitter for large files
- **TokenTextSplitter**: Word/token-based splitting
- **SemanticTextSplitter**: Sentence boundary-aware splitting
- **create_text_splitter()**: Factory function for splitter creation

#### `embeddings.py` - Search and Embedding Module

- **SimpleEmbeddings**: Vector embedding generation and semantic search
- **HybridSearch**: Combined keyword and semantic search
- **simple_text_search()**: Basic keyword-based search functionality

## Development Setup

This project uses **uv** as the package manager and **pre-commit** for code quality assurance. To set up the development environment:

1. Initialize the project with uv (if not already done):

   ```bash
   uv init
   ```

2. Add required dependencies:

   ```bash
   # For PDF processing
   uv add pymupdf

   # For embeddings and search (if needed)
   uv add numpy
   uv add scikit-learn
   uv add sentence-transformers
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate  # uv automatically creates and manages this
   ```

4. Install and set up pre-commit:

   ```bash
   # Install pre-commit
   uv add pre-commit

   # Install git hooks
   pre-commit install
   ```

5. Run the project:

   ```bash
   uv run python main.py document.txt
   ```

### Common uv Commands

- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv sync` - Sync dependencies
- `uv run <command>` - Run a command in the virtual environment
- `uv lock` - Update lockfile
- `uv tree` - Show dependency tree

### Pre-commit Commands

- `pre-commit run` - Run checks on staged files
- `pre-commit run --all-files` - Run checks on all files
- `pre-commit autoupdate` - Update hook versions
- `pre-commit install` - Install git hooks

## Usage Examples

### Basic Document Reading

```bash
# Read text file
uv run python main.py document.txt

# Read PDF file
uv run python main.py document.pdf

# Read from URL
uv run python main.py https://example.com
```

### Text Splitting

```bash
# Split text using recursive splitter
uv run python main.py large_file.txt --split --chunk-size 500 --splitter recursive

# Use different splitting strategies
uv run python main.py document.txt --split --splitter semantic --chunk-size 300
```

### Search Functionality

```bash
# Keyword search
uv run python main.py document.txt --search-mode keyword --search-query "Python programming"

# Semantic search
uv run python main.py document.txt --search-mode semantic --search-query "machine learning"

# Hybrid search (keyword + semantic)
uv run python main.py document.txt --search-mode hybrid --search-query "data algorithms"

# Combined splitting and search
uv run python main.py large_file.txt --split --search-mode semantic --search-query "artificial intelligence"
```

## Code Organization Guidelines

### Import Organization

Follow the established import structure in all Python files:

```python
# ===== 标准库导入 =====
import sys
import argparse
from pathlib import Path

# ===== 第三方库导入 =====
try:
    import fitz  # PyMuPDF for PDF reading
except ImportError:
    # Handle missing dependencies
    pass

# ===== 项目自定义模块导入 =====
# Module descriptions
from text_splitter import create_text_splitter
from embeddings import SimpleEmbeddings, HybridSearch, simple_text_search
```

### Module Structure

- Each module should have a clear, single responsibility
- Use factory functions for creating complex objects
- Provide comprehensive docstrings in Chinese (project's primary language)
- Follow Python naming conventions (snake_case for functions, CamelCase for classes)
- Type hints should be used for better code maintainability

### Pre-commit Integration

All code changes must pass pre-commit checks before committing. The configured hooks ensure:

- Code formatting consistency (black, isort)
- Code quality standards (flake8, mypy)
- Security best practices (bandit, safety)
- Documentation completeness (interrogate)
- File formatting standards (YAML, Markdown)

## Development Notes

- This is a learning project focused on document processing and text analysis
- Primary language for documentation and comments is Chinese
- Uses **uv** as the package manager with automatic virtual environment management
- Modular architecture allows for easy extension of new text splitting strategies
- Custom implementation rather than LangChain dependencies for better control and learning
- Code is structured to be educational while maintaining production-quality standards
- Pre-commit hooks ensure consistent code quality and formatting across the project
- See `PRECOMMIT.md` for detailed pre-commit configuration and usage instructions
