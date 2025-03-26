# RenRAG ColBERT

A Python library for semantic document indexing and searching using the [ColBERT](https://huggingface.co/colbert-ir/colbertv2.0) retrieval model. RenRAG ColBERT provides a simplified, high-performance interface for building and querying semantic search indices.

## Features

- **Semantic Search** - Uses ColBERT's contextualized late interaction approach for more accurate search results
- **File and Document Tracking** - Associate documents with file IDs for efficient organization
- **Custom Document IDs** - Use your own IDs or automatic UUID generation
- **Rich Metadata Support** - Store and retrieve custom metadata with each document
- **Flexible Index Management** - Create empty indices, add/remove documents, and filter search results
- **Precise Similarity Scores** - Four-decimal precision for better result ranking
- **Threshold Filtering** - Filter search results by minimum similarity score
- **File Filtering** - Target searches to specific documents by file ID

## Installation

```bash
pip install renrag-colbert
```

Requirements:
- Python 3.7+
- PyTorch
- Transformers
- ColBERT

## Quick Start

```python
from renrag_colbert import ColbertIndexer, ColbertSearcher

# Initialize indexer and index some documents
indexer = ColbertIndexer(device="cpu")
documents = [
    "ColBERT uses neural representations for efficient retrieval",
    "The model enables effective semantic search across documents",
    "Renrag ColBERT makes document indexing and search easy"
]

# Create an index
index_path = indexer.index(
    documents=documents,
    index_name="my_first_index"
)

# Search the index
searcher = ColbertSearcher(index_path=index_path, device="cpu")
results = searcher.search("semantic search", k=2)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. [score: {result['score']}] {result['text']}")
```

For more comprehensive examples, check out the [`examples/`](examples/) directory, which contains:

- **Basic Search**: Simple document indexing and searching
- **Metadata Search**: Working with document metadata
- **Filtered Search**: Filtering by threshold and file IDs
- **Incremental Indexing**: Building and modifying indices over time
- **Advanced Search**: Complex search patterns with a larger dataset

## Usage

### Creating and Indexing Documents

#### Creating an Empty Index

```python
from renrag_colbert import ColbertIndexer

# Initialize the indexer
indexer = ColbertIndexer(model_name="colbert-ir/colbertv2.0")

# Create an empty index
empty_index_path = indexer.create_index(
    index_name="my_empty_index",
    index_dir=".renrag/index",  # Optional - default is .renrag/index
    overwrite=False  # Optional - by default, will raise an error if index already exists
)
print(f"Empty index created at: {empty_index_path}")

# Add documents to the empty index later
doc_id = indexer.add_document(
    index_path=empty_index_path,
    document="This is a document added to an initially empty index.",
    doc_id="first-doc",  # Optional
    file_id="document.pdf",  # Optional
    metadata={"author": "John Smith", "year": 2023}  # Optional
)
```

#### Indexing Documents at Creation Time

```python
# Documents to index
documents = [
    "ColBERT is a fast and accurate retrieval model.",
    "It uses contextualized late interaction to efficiently score query-document pairs.",
    "The model was developed by researchers at Stanford University.",
    "ColBERT enables per-term interaction between queries and documents."
]

# Optional: Add metadata for each document
document_metadata = [
    {
        "author": "John Smith",
        "year": 2020,
        "category": "research",
        "keywords": ["information retrieval", "neural networks"]
    },
    {
        "author": "Jane Doe",
        "year": 2021,
        "category": "explanation",
        "keywords": ["contextualized", "late interaction"]
    },
    {
        "author": "Alice Johnson",
        "year": 2019,
        "category": "history",
        "keywords": ["Stanford", "development"]
    },
    {
        "author": "Bob Williams",
        "year": 2022,
        "category": "technical",
        "keywords": ["interaction", "tokens", "per-term"]
    }
]

# Optional: Add file IDs for each document
file_ids = [
    "introduction.pdf",
    "methodology.pdf",
    "background.pdf",
    "architecture.pdf"
]

# Optional: Add custom document IDs
# If not provided, UUIDs will be generated automatically
doc_ids = [
    "doc-001",
    "doc-002",
    None,  # Will be assigned a UUID
    "doc-004"
]

# Index the documents with metadata, file IDs, and document IDs
index_path = indexer.index(
    documents=documents, 
    document_metadata=document_metadata,  # Optional
    file_ids=file_ids,  # Optional
    doc_ids=doc_ids,  # Optional - if not provided, UUIDs will be generated
    index_name="my_colbert_index",
    index_dir=".renrag/index",  # Optional - default is .renrag/index
    overwrite=False  # Optional - by default, will raise an error if index already exists
)
print(f"Index created at: {index_path}")

# To overwrite an existing index, set overwrite=True
index_path = indexer.index(
    documents=documents,
    index_name="my_colbert_index",
    overwrite=True  # Will replace an existing index with the same name
)
```

### Searching Documents

```python
from renrag_colbert import ColbertSearcher

# Initialize the searcher with the path to your index
searcher = ColbertSearcher(index_path=".renrag/index/my_colbert_index")

# Basic search - returns top k results
results = searcher.search("How does ColBERT work?", k=3)

# Print results with all information
for result in results:
    print(f"Document ID: {result['id']}")  # Unique ID (UUID or custom)
    print(f"Index: {result['index']}")     # Internal position index
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print(f"File: {result['file_id']}")
    
    # Access metadata fields
    if result['metadata']:
        print(f"Author: {result['metadata'].get('author')}")
        print(f"Year: {result['metadata'].get('year')}")
        print(f"Category: {result['metadata'].get('category')}")
        print(f"Keywords: {', '.join(result['metadata'].get('keywords', []))}")
    print("-" * 50)

# Search with similarity threshold
# Only returns documents with similarity scores >= 0.3 (scale of 0.0 to 1.0)
filtered_results = searcher.search("How does ColBERT work?", k=5, threshold=0.3)

# Search with a higher threshold for more precise results
precise_results = searcher.search("How does ColBERT work?", k=5, threshold=0.7)

# Search with file filtering
# Filter to documents with a specific file ID
single_file_results = searcher.search(
    "How does ColBERT work?", 
    k=5, 
    filter_by_files="methodology.pdf"
)

# Filter to documents with any of the specified file IDs
multi_file_results = searcher.search(
    "How does ColBERT work?", 
    k=5, 
    filter_by_files=["introduction.pdf", "architecture.pdf"]
)
```

The similarity scores are normalized between 0.0 and 1.0, with a precision of four decimal places. Scores are relative to the best match in the result set, ensuring better score distribution with more differentiation between results. Use the threshold parameter to filter results by minimum similarity score. You can also filter search results by file ID using the filter_by_files parameter to narrow down the search to specific documents or document groups.

### Adding Documents to an Existing Index

```python
from renrag_colbert import ColbertIndexer, ColbertSearcher

# Initialize indexer
indexer = ColbertIndexer()

# Add a single document
doc_id = indexer.add_document(
    index_path=".renrag/index/my_colbert_index",
    document="ColBERT is a powerful retrieval model for efficient search.",
    doc_id="custom-id-123",  # Optional, UUID generated if None
    file_id="document.pdf",  # Optional
    metadata={               # Optional
        "author": "John Smith",
        "year": 2023,
        "keywords": ["colbert", "retrieval", "search"]
    }
)
print(f"Added document with ID: {doc_id}")

# Add multiple documents at once for better performance
doc_ids = indexer.add_documents(
    index_path=".renrag/index/my_colbert_index",
    documents=[
        "First document to add",
        "Second document to add"
    ],
    doc_ids=["doc-1", "doc-2"],  # Optional, UUIDs generated for None values
    file_ids=["file1.pdf", "file2.pdf"],  # Optional
    metadata_list=[  # Optional
        {"author": "Author 1", "year": 2023},
        {"author": "Author 2", "year": 2024}
    ]
)
print(f"Added document IDs: {doc_ids}")

# After addition, reload the searcher index
searcher.reload_index(".renrag/index/my_colbert_index")
```

### Removing Documents

```python
from renrag_colbert import ColbertIndexer, ColbertSearcher

# Initialize indexer
indexer = ColbertIndexer()

# Remove document by document ID
removed_ids = indexer.remove_documents(
    index_path=".renrag/index/my_colbert_index",
    doc_id="doc-001"
)
print(f"Removed document IDs: {removed_ids}")

# Remove documents by file ID
removed_ids = indexer.remove_documents(
    index_path=".renrag/index/my_colbert_index",
    file_id="document.pdf"  # Removes all documents with this file ID
)
print(f"Removed document IDs: {removed_ids}")

# After removal, reload the searcher index
searcher = ColbertSearcher(index_path=".renrag/index/my_colbert_index")
# Or if searcher already exists:
searcher.reload_index(".renrag/index/my_colbert_index")
```

# API Reference

## ColbertIndexer

The main class for creating and managing search indices.

### Methods

#### `__init__(model_name="colbert-ir/colbertv2.0", device="cuda"/"cpu")`
- Initializes the indexer with the specified ColBERT model and device

#### `create_index(index_name, index_dir=None, overwrite=False)`
- Creates an empty index structure without documents
- Returns the path to the created index

#### `index(documents, index_name, document_metadata=None, file_ids=None, doc_ids=None, index_dir=None, overwrite=False)`
- Creates a new index with the provided documents and metadata
- Returns the path to the created index

#### `add_document(index_path, document, doc_id=None, file_id=None, metadata=None)`
- Adds a single document to an existing index
- Returns the document ID (either provided or generated)

#### `add_documents(index_path, documents, doc_ids=None, file_ids=None, metadata_list=None)`
- Adds multiple documents to an existing index
- Returns a list of document IDs

#### `remove_documents(index_path, doc_id=None, file_id=None)`
- Removes documents matching the specified doc_id or file_id
- Returns a list of removed document IDs

## ColbertSearcher

The main class for searching indexed documents.

### Methods

#### `__init__(index_path, model_name=None, device="cuda"/"cpu")`
- Initializes the searcher with the specified index and device

#### `reload_index(index_path, model_name=None, device=None)`
- Reloads the index data (useful after adding or removing documents)

#### `search(query, k=10, threshold=0.0, filter_by_files=None)`
- Searches the index with the provided query
- Parameters:
  - `query`: The search query text
  - `k`: Maximum number of results to return
  - `threshold`: Minimum similarity score (0.0 to 1.0)
  - `filter_by_files`: File ID or list of file IDs to filter results by
- Returns a list of document dictionaries containing:
  - `id`: Document ID
  - `index`: Internal index position
  - `text`: Document text
  - `score`: Similarity score (0.0 to 1.0)
  - `metadata`: Document metadata
  - `file_id`: File ID

## License

MIT
