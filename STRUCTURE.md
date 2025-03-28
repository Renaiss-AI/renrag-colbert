# Renrag ColBERT Library Structure

This document provides an overview of the Renrag ColBERT library architecture and code organization.

## Project Structure

```
renrag-colbert/
   src/
      renrag_colbert/           # Main package directory
          __init__.py           # Package exports
          renrag_indexer.py     # Document indexing functionality
          renrag_searcher.py    # Document searching functionality
          tests/                # Test directory
              __init__.py
              test_indexer.py   # Tests for indexer
              test_searcher.py  # Tests for searcher
   example.py                    # Example usage file
   setup.py                      # Package setup configuration
   README.md                     # Documentation
   CHANGELOG.md                  # Version history
   CONTRIBUTING.md               # Contribution guidelines
   LICENSE                       # MIT License
   .gitignore                    # Git ignore file
   .pre-commit-config.yaml       # Pre-commit hooks configuration
```

## Architecture Overview

### Core Components

1. **ColbertIndexer**: Handles document indexing and index management 
   - Creates and manages index files
   - Adds documents to indices
   - Removes documents from indices
   - Manages document metadata, file IDs, and document IDs

2. **ColbertSearcher**: Enables semantic search over indexed documents
   - Loads index data
   - Handles query encoding
   - Computes document similarity scores
   - Filters results by threshold and file IDs
   - Formats and ranks search results

### Data Flow

1. **Indexing Process**:
   - User provides documents, optional metadata, and IDs
   - Documents are encoded using the ColBERT model
   - Embeddings and metadata are saved to the index directory
   - Index can be created with documents or empty

2. **Search Process**:
   - User provides a natural language query
   - Query is encoded using the ColBERT model
   - Similarity scores are computed between query and documents
   - Results are filtered by threshold and optional file IDs
   - Top k results are returned with metadata and scores

### Index Structure

A Renrag ColBERT index contains the following files:

- `metadata.pt`: General index metadata (model name, dimensions, etc.)
- `doc_mapping.pt`: Mapping of internal indices to document text
- `doc_metadata.pt`: Document metadata dictionary
- `doc_file_ids.pt`: Mapping of indices to file IDs
- `document_ids.pt`: Mapping of indices to document IDs
- `doc_*.pt`: Document embedding files (one per document)

## Key Features

1. **Semantic Search**: Uses neural embeddings for better search relevance
2. **Document Management**: Add, remove, and organize documents
3. **Metadata Support**: Store and retrieve rich document metadata
4. **ID Management**: Track documents by custom IDs or file IDs
5. **Empty Index Creation**: Build indices incrementally 
6. **Search Filtering**: By threshold and file IDs
7. **Normalized Scoring**: Four-decimal precision between 0.0 and 1.0

## Function Call Flow

1. Initialize indexer
2. Create index (with or without documents)
3. Add/remove documents as needed
4. Initialize searcher with index path
5. Search documents with queries
6. Process results from search

## Extension Points

- Add more metadata filtering capabilities
- Implement more sophisticated ranking algorithms
- Add document chunk support for long documents
- Add data type converters for different document formats
- Implement caching mechanisms for frequently accessed documents