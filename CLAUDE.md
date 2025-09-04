# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project focused on document loading and processing with custom text splitting and search capabilities. The project supports reading various document formats (TXT, PDF, URLs) and provides advanced text processing features including multiple splitting strategies and search modes.

## Project Structure

```text
Document-Loader/
├── main.py                    # Main application entry point with CLI
├── text_splitter.py           # Text splitting module with 5 strategies
├── embeddings.py              # Vector embeddings and search functionality
├── search_engine.py           # Web search engine integration
├── vector_store.py            # Vector storage and retrieval
├── retrieval_qa.py            # Retrieval QA system
├── advanced_retrieval_qa.py   # Advanced retrieval implementations
├── README.md                  # Project documentation (in Chinese)
├── LICENSE                    # MIT License
├── CLAUDE.md                  # This file
├── PRECOMMIT.md               # Pre-commit configuration
├── pyproject.toml             # Project dependencies and configuration
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
└── .venv/                     # Virtual environment (managed by uv)
```

### Core Modules

#### `main.py` - Application Entry Point

- **Command-line interface**: Comprehensive argparse implementation with multiple modes
- **Document loading**: TXT, PDF, and URL sources with encoding detection
- **Text processing**: Integration with all text splitting strategies
- **Search functionality**: Keyword, semantic, and hybrid search modes
- **Web search**: Integration with multiple search engines (DuckDuckGo, Bing, SerpApi)
- **Import organization**: Clear separation of 标准库, 第三方库, and 项目自定义模块

#### `text_splitter.py` - Text Processing Module

- **TextSplitter**: Base class with common chunking logic
- **CharacterTextSplitter**: Character-based splitting with configurable separators
- **RecursiveCharacterTextSplitter**: Iterative splitting using hierarchical separators (paragraphs → sentences → words → spaces)
- **StreamingTextSplitter**: Memory-efficient generator-based splitter for large files
- **TokenTextSplitter**: Word/token-based splitting using regex tokenization
- **SemanticTextSplitter**: Sentence boundary-aware splitting for Chinese and English
- **create_text_splitter()**: Factory function for dynamic splitter creation

#### `embeddings.py` - Search and Embedding Module

- **SimpleEmbeddings**: TF-IDF based vector embeddings with cosine similarity
- **HybridSearch**: Combined keyword matching and semantic similarity with configurable weights
- **simple_text_search()**: Fast keyword-based search without training requirements
- **Language support**: Chinese character handling and English word tokenization
- **No external dependencies**: Pure Python implementation for learning purposes

## Development Setup

This project uses **uv** as the package manager and **pre-commit** for code quality assurance. To set up the development environment:

## Setup Commands

```bash
# Initialize project (if not already done)
uv init

# Install dependencies from pyproject.toml
uv sync

# Activate virtual environment
source .venv/bin/activate  # uv automatically creates and manages this

# Install and set up pre-commit hooks
uv add pre-commit
pre-commit install

# Run the application
uv run python main.py document.txt
```

## Common Commands

### Development

```bash
# Run any command with uv
uv run python main.py --help

# Add new dependencies
uv add <package-name>

# Remove dependencies
uv remove <package-name>

# Update lockfile
uv lock

# Show dependency tree
uv tree
```

### Code Quality

```bash
# Run pre-commit on staged files
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files

# Update pre-commit hook versions
pre-commit autoupdate

# Install git hooks (if not already installed)
pre-commit install
```

### Pre-commit Configuration

The project uses comprehensive pre-commit hooks configured in `.pre-commit-config.yaml`:

- **Code formatting**: Black (line-length: 180) and isort
- **Code quality**: Flake8 (line-length: 180, ignore E203)
- **Security**: Bandit for security scanning
- **File checks**: YAML, JSON, AST, merge conflicts, large files
- **Best practices**: Debug statements, private key detection

### Web Search Commands

```bash
# Basic web search
uv run python main.py --web-search "Python programming"

# Search with specific engine and results count
uv run python main.py --web-search "machine learning" --engine web --results 5

# Use API-based search engines (requires API keys)
uv run python main.py --web-search "AI research" --engine bing --bing-api-key YOUR_KEY
uv run python main.py --web-search "data science" --engine serpapi --serpapi-key YOUR_KEY
```

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

## Architecture Overview

### Design Patterns

- **Factory Pattern**: `create_text_splitter()` for dynamic splitter creation
- **Strategy Pattern**: Different text splitting strategies with common interface
- **Template Method**: Base classes with common functionality and specialized implementations
- **Generator Pattern**: StreamingTextSplitter for memory-efficient processing

### Import Organization Standard

All Python files follow this import structure:

```python
# ===== 标准库导入 =====
import argparse
import sys
from pathlib import Path

# ===== 第三方库导入 =====
try:
    import fitz  # PyMuPDF for PDF reading
except ImportError:
    # Handle missing dependencies gracefully
    pass

# ===== 项目自定义模块导入 =====
# Module descriptions grouped by functionality
from text_splitter import create_text_splitter
from embeddings import SimpleEmbeddings, HybridSearch, simple_text_search
```

### Module Design Principles

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extensible through factory functions and base classes
- **Dependency Management**: Graceful handling of missing optional dependencies
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Documentation**: Chinese docstrings with clear usage examples
- **Error Handling**: Robust error handling with fallback mechanisms

## Development Notes

### Project Context

- **Learning Focus**: Educational project for understanding document processing and text analysis
- **Language**: Primary documentation and comments in Chinese
- **Package Manager**: Uses **uv** with automatic virtual environment management
- **Dependencies**: Minimal external dependencies; custom implementations for learning

### Architecture Decisions

- **Custom vs External**: Built custom text splitting and embedding implementations instead of using LangChain for better understanding
- **Modular Design**: Each splitting strategy is implemented separately for clarity and extensibility
- **Performance**: Memory-efficient streaming processing for large documents
- **Language Support**: Comprehensive Chinese and English text processing capabilities

### Quality Assurance

- **Pre-commit Integration**: Automated code quality checks before every commit
- **Testing**: Includes test files for core functionality (e.g., `test_search_engine.py`)
- **Type Safety**: Full type hint coverage for better maintainability
- **Error Handling**: Graceful degradation and fallback mechanisms throughout

### Key Technical Details

- **Text Encoding**: Multiple encoding support (UTF-8, GBK, Latin-1) for robust file reading
- **URL Validation**: Secure URL handling with scheme validation and timeout protection
- **Memory Management**: Generator-based processing for large file handling
- **Search Flexibility**: Multiple search modes with configurable parameters
