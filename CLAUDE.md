# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project focused on document loading and processing using LangChain. The project is designed to facilitate file reading and retrieval operations for document processing workflows.

## Project Structure

Current project structure:
- `README.md` - Project documentation (in Chinese)
- `LICENSE` - MIT License
- `main.py` - Main application with document loading and text splitting
- `CLAUDE.md` - This guidance file
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependencies

## Development Setup

This project uses **uv** as the package manager. To set up the development environment:

1. Initialize the project with uv (if not already done):
   ```bash
   uv init
   ```

2. Add LangChain and other dependencies:
   ```bash
   uv add langchain
   uv add langchain-community  # For document loaders
   uv add langchain-core       # Core LangChain functionality
   uv add pymupdf              # For PDF processing
   # Add other document processing dependencies as needed
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # uv automatically creates and manages this
   ```

4. Run the project:
   ```bash
   # Basic document reading
   uv run python main.py document.txt
   uv run python main.py document.pdf
   uv run python main.py https://example.com
   
   # With text splitting
   uv run python main.py document.txt --split --chunk-size 1000 --splitter recursive
   uv run python main.py large_file.txt --split --splitter streaming
   ```

### Common uv Commands

- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv sync` - Sync dependencies
- `uv run <command>` - Run a command in the virtual environment
- `uv lock` - Update lockfile
- `uv tree` - Show dependency tree

## Core Features

### Document Loading
- **Text Files**: Supports `.txt` files with multiple encoding options (UTF-8, GBK, Latin-1)
- **PDF Files**: Full PDF text extraction using PyMuPDF with page-by-page processing
- **Web Content**: URL content fetching with HTML tag stripping and encoding handling

### Text Splitting
Multiple text splitting strategies for different use cases:

- **CharacterTextSplitter**: Simple character-based splitting with custom separators
- **RecursiveCharacterTextSplitter**: Memory-efficient iterative splitting using paragraph/sentence/word boundaries
- **StreamingTextSplitter**: Generator-based streaming splitter for very large files
- **TokenTextSplitter**: Token-based splitting for English text
- **SemanticTextSplitter**: Sentence boundary-aware splitting for semantic coherence

### Command Line Interface
Comprehensive CLI with flexible options:
- Multiple input sources (files, URLs)
- Configurable chunk sizes and overlap
- Multiple splitting strategies
- Encoding options for different file formats

## Notes for Development

- This is a learning project for LangChain document processing
- The README indicates the project is for educational purposes
- Primary language appears to be Chinese based on the README
- Uses **uv** as the package manager (modern Python package management)
- Memory-efficient text splitting implementation avoids recursion issues
- Supports both Chinese and English text processing
- Designed for scalability with streaming capabilities for large files
- uv automatically manages virtual environments in `.venv/` directory

## Key Design Decisions

- **Memory Safety**: Iterative text splitters prevent stack overflow with large documents
- **Flexibility**: Multiple splitting strategies for different use cases
- **Performance**: Streaming capabilities for handling very large files
- **Internationalization**: Full Unicode support with multiple encoding options
- **Extensibility**: Clean class hierarchy for adding new splitter types