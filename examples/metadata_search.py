"""
Example of using metadata with RenRAG ColBERT.

This example demonstrates:
- Creating an index with documents and metadata
- Searching the index
- Displaying search results with metadata
"""

from renrag_colbert import ColbertIndexer, ColbertSearcher

def main():
    print("RenRAG ColBERT Metadata Example")
    print("-" * 50)
    
    # Initialize the indexer
    print("Initializing indexer...")
    indexer = ColbertIndexer(device="cpu")
    
    # Sample documents with metadata
    documents = [
        "Deep learning has revolutionized computer vision tasks.",
        "Natural language processing benefits from transformer models.",
        "Graph neural networks are powerful for analyzing network data.",
        "Reinforcement learning enables agents to make sequential decisions.",
        "Self-supervised learning reduces the need for labeled data."
    ]
    
    # Metadata for each document
    metadata = [
        {
            "author": "Alex Johnson",
            "year": 2020,
            "category": "Computer Vision",
            "keywords": ["deep learning", "CNN", "vision"]
        },
        {
            "author": "Sarah Lee",
            "year": 2021,
            "category": "NLP",
            "keywords": ["transformers", "BERT", "language models"]
        },
        {
            "author": "Michael Chen",
            "year": 2019,
            "category": "Graph Learning",
            "keywords": ["GNN", "networks", "graph algorithms"]
        },
        {
            "author": "Emma Davis",
            "year": 2022,
            "category": "Reinforcement Learning",
            "keywords": ["RL", "agents", "decision making"]
        },
        {
            "author": "David Wilson",
            "year": 2023,
            "category": "Self-Supervised Learning",
            "keywords": ["SSL", "unsupervised", "representation learning"]
        }
    ]
    
    # Create an index with documents and metadata
    print("Creating index with documents and metadata...")
    index_path = indexer.index(
        documents=documents,
        document_metadata=metadata,
        index_name="metadata_example",
        overwrite=True
    )
    print(f"Index created at: {index_path}")
    
    # Initialize the searcher
    print("\nInitializing searcher...")
    searcher = ColbertSearcher(index_path=index_path, device="cpu")
    
    # Search the index
    query = "machine learning techniques"
    print(f"\nSearching for: \"{query}\"")
    
    results = searcher.search(query, k=5)
    
    # Display results with metadata
    print("\nSearch Results with Metadata:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result['score']}] {result['text']}")
        print(f"   Author: {result['metadata'].get('author')}")
        print(f"   Year: {result['metadata'].get('year')}")
        print(f"   Category: {result['metadata'].get('category')}")
        print(f"   Keywords: {', '.join(result['metadata'].get('keywords', []))}")
        print("-" * 50)
    
    print("\nDone!")

if __name__ == "__main__":
    main()