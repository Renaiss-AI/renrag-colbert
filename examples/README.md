# RenRAG ColBERT Examples

This directory contains example scripts demonstrating various features of the RenRAG ColBERT library.

## Running the Examples

To run any example:

```bash
python examples/basic_search.py
```

Make sure you have RenRAG ColBERT installed:

```bash
pip install renrag-colbert
```

## Examples Overview

### 1. Basic Search (`basic_search.py`)

Demonstrates the core functionality of indexing documents and searching them:
- Creating a simple index with documents
- Performing a basic search
- Displaying search results

### 2. Metadata Search (`metadata_search.py`)

Shows how to work with document metadata:
- Creating an index with documents and metadata
- Searching the index
- Displaying search results with associated metadata

### 3. Filtered Search (`filtered_search.py`)

Demonstrates various filtering techniques:
- Threshold filtering to get only highly relevant results
- File ID filtering to restrict search to specific documents
- Combining multiple filtering techniques

### 4. Incremental Indexing (`incremental_indexing.py`)

Shows how to build and modify indices incrementally:
- Creating an empty index
- Adding documents one at a time
- Adding multiple documents at once
- Removing documents by doc_id and file_id
- Searching after each modification

### 5. Advanced Search (`advanced_search.py`)

A more sophisticated example showing:
- Working with a larger dataset (academic paper abstracts)
- Creating and reusing persistent indices
- Performance timing
- Complex queries across different topics
- Category-based filtering

## Notes

- All examples use `device="cpu"` for compatibility, but you can change to `"cuda"` if you have a GPU
- The advanced example creates an `examples_data` directory to store the index
- You may need to adjust paths based on your working directory