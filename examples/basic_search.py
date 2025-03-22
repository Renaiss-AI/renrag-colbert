"""
Basic example of document indexing and searching with RenRAG ColBERT.

This example demonstrates:
- Creating an index with documents
- Searching the index with a query
- Displaying search results
"""

from renrag_colbert import ColbertIndexer, ColbertSearcher

def main():
    print("RenRAG ColBERT Basic Search Example")
    print("-" * 50)
    
    # Initialize the indexer with the ColBERT model
    print("Initializing indexer...")
    indexer = ColbertIndexer(device="cpu")
    
    # Sample documents to index
    documents = [
        "ColBERT is a fast and accurate neural retrieval model.",
        "It uses contextualized late interaction to efficiently score query-document pairs.",
        "The model was developed by researchers at Stanford University.",
        "ColBERT enables per-term interaction between queries and documents.",
        "RenRAG ColBERT simplifies document indexing and semantic search."
    ]
    
    # Create an index with the documents
    print("Creating index with documents...")
    index_path = indexer.index(
        documents=documents,
        index_name="basic_example",
        overwrite=True
    )
    print(f"Index created at: {index_path}")
    
    # Initialize the searcher with the index path
    print("\nInitializing searcher...")
    searcher = ColbertSearcher(index_path=index_path, device="cpu")
    
    # Define a search query
    query = "How does ColBERT work?"
    print(f"\nSearching for: \"{query}\"")
    
    # Search the index
    results = searcher.search(query, k=3)
    
    # Display results
    print("\nSearch Results:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result['score']}] {result['text']}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()