# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project focused on document loading and processing using LangChain. The project is designed to facilitate file reading and retrieval operations for document processing workflows.

## Project Structure

Currently, this is a minimal project with:
- `README.md` - Project documentation (in Chinese)
- `LICENSE` - MIT License
- No Python source code files yet
- No dependency management files present

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
   # Add other document processing dependencies as needed
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # uv automatically creates and manages this
   ```

4. Run the project:
   ```bash
   uv run python main.py  # or whatever the main file is named
   ```

### Common uv Commands

- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv sync` - Sync dependencies
- `uv run <command>` - Run a command in the virtual environment
- `uv lock` - Update lockfile
- `uv tree` - Show dependency tree

## Intended Architecture

Based on the README description, this project will focus on:
- Document loading capabilities using LangChain
- File reading and processing
- Document retrieval functionality
- Support for various document formats

## Notes for Development

- This is a learning project for LangChain document processing
- The README indicates the project is for educational purposes
- Primary language appears to be Chinese based on the README
- Uses **uv** as the package manager (modern Python package management)
- No existing code patterns or conventions established yet
- Will need to establish proper Python project structure as development progresses
- uv automatically manages virtual environments in `.venv/` directory