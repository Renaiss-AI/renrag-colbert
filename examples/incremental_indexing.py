"""
Example of incremental document indexing with RenRAG ColBERT.

This example demonstrates:
- Creating an empty index
- Adding documents incrementally
- Removing documents
- Searching after each modification
"""

from renrag_colbert import ColbertIndexer, ColbertSearcher

def main():
    print("RenRAG ColBERT Incremental Indexing Example")
    print("-" * 60)
    
    # Initialize the indexer
    print("Initializing indexer...")
    indexer = ColbertIndexer(device="cpu")
    
    # 1. Create an empty index
    print("\n1. Creating an empty index...")
    index_path = indexer.create_index(
        index_name="incremental_example",
        overwrite=True
    )
    print(f"Empty index created at: {index_path}")
    
    # Initialize the searcher with the empty index
    searcher = ColbertSearcher(index_path=index_path, device="cpu")
    
    # Search the empty index (should return no results)
    print("\nSearching empty index...")
    results = searcher.search("test query", k=5)
    print(f"Found {len(results)} results in empty index (expected: 0)")
    
    # 2. Add a single document
    print("\n2. Adding a single document...")
    doc_id = indexer.add_document(
        index_path=index_path,
        document="Artificial intelligence is transforming industries across the globe.",
        doc_id="doc-001",
        file_id="ai_impact.pdf",
        metadata={
            "author": "Jane Smith",
            "year": 2023,
            "category": "AI"
        }
    )
    print(f"Added document with ID: {doc_id}")
    
    # Reload searcher and search
    print("\nSearching after adding one document...")
    searcher.reload_index(index_path)
    results = searcher.search("artificial intelligence", k=5)
    print_results(results, "Results after adding one document")
    
    # 3. Add multiple documents at once
    print("\n3. Adding multiple documents at once...")
    new_docs = [
        "Machine learning models require large amounts of training data.",
        "Deep learning has achieved remarkable results in image recognition."
    ]
    new_doc_ids = ["doc-002", "doc-003"]
    new_file_ids = ["ml_data.pdf", "deep_learning.pdf"]
    new_metadata = [
        {"author": "Alex Johnson", "year": 2022, "category": "ML"},
        {"author": "Maria Garcia", "year": 2021, "category": "DL"}
    ]
    
    doc_ids = indexer.add_documents(
        index_path=index_path,
        documents=new_docs,
        doc_ids=new_doc_ids,
        file_ids=new_file_ids,
        metadata_list=new_metadata
    )
    print(f"Added document IDs: {doc_ids}")
    
    # Reload searcher and search again
    print("\nSearching after adding multiple documents...")
    searcher.reload_index(index_path)
    results = searcher.search("machine learning", k=5)
    print_results(results, "Results after adding multiple documents")
    
    # 4. Remove a document by doc_id
    print("\n4. Removing a document by doc_id...")
    removed_ids = indexer.remove_documents(
        index_path=index_path,
        doc_id="doc-002"
    )
    print(f"Removed document IDs: {removed_ids}")
    
    # Reload searcher and search again
    print("\nSearching after removing document...")
    searcher.reload_index(index_path)
    results = searcher.search("machine learning", k=5)
    print_results(results, "Results after removing doc-002")
    
    # 5. Remove a document by file_id
    print("\n5. Removing a document by file_id...")
    removed_ids = indexer.remove_documents(
        index_path=index_path,
        file_id="deep_learning.pdf"
    )
    print(f"Removed document IDs: {removed_ids}")
    
    # Final search
    print("\nFinal search after all modifications...")
    searcher.reload_index(index_path)
    results = searcher.search("artificial intelligence", k=5)
    print_results(results, "Final results")

def print_results(results, title):
    print(f"\n{title}:")
    print("-" * 60)
    if not results:
        print("No results found")
    else:
        for i, result in enumerate(results, 1):
            print(f"{i}. [Score: {result['score']}] {result['text']}")
            print(f"   Document ID: {result['id']}")
            print(f"   File: {result['file_id']}")
            if result['metadata']:
                print(f"   Author: {result['metadata'].get('author')}")
                print(f"   Year: {result['metadata'].get('year')}")
                print(f"   Category: {result['metadata'].get('category')}")
            print()
        print(f"Total results: {len(results)}")

if __name__ == "__main__":
    main()