from renrag_colbert import ColbertIndexer, ColbertSearcher

def main():
    # Sample documents to index
    documents = [
        "ColBERT is a fast and accurate retrieval model.",
        "It uses contextualized late interaction to efficiently score query-document pairs.",
        "The model was developed by researchers at Stanford University.",
        "ColBERT enables per-term interaction between queries and documents.",
        "It uses late interaction, computing interactions between query and document representations."
    ]
    
    # Sample metadata for each document
    document_metadata = [
        {
            "author": "John Smith",
            "year": 2020,
            "category": "research",
            "keywords": ["information retrieval", "neural networks", "transformers"]
        },
        {
            "author": "Jane Doe",
            "year": 2021,
            "category": "explanation",
            "keywords": ["contextualized", "late interaction", "efficiency"]
        },
        {
            "author": "Alice Johnson",
            "year": 2019,
            "category": "history",
            "keywords": ["Stanford", "development", "research team"]
        },
        {
            "author": "Bob Williams",
            "year": 2022,
            "category": "technical",
            "keywords": ["interaction", "tokens", "per-term"]
        },
        {
            "author": "Emily Brown",
            "year": 2022,
            "category": "technical",
            "keywords": ["representations", "interaction", "efficiency"]
        }
    ]
    
    # Sample file IDs for each document
    file_ids = [
        "intro.pdf",
        "methods.pdf",
        "background.pdf",
        "architecture.pdf", 
        "details.pdf"
    ]
    
    # Sample custom document IDs for each document
    # For some documents we'll use custom IDs, for others we'll let the system generate UUIDs
    doc_ids = [
        "doc-001",
        "doc-002",
        None,  # Will generate UUID
        "doc-004",
        None   # Will generate UUID
    ]
    
    import os
    # Create a specific directory for our indices to make it more predictable
    index_dir = os.path.join(os.getcwd(), "indices")
    os.makedirs(index_dir, exist_ok=True)
    
    # Initialize the indexer
    print("Initializing the indexer...")
    indexer = ColbertIndexer(device="cpu")
    
    # Index the documents
    print("Indexing documents...")
    try:
        index_path = indexer.index(
            documents=documents,
            document_metadata=document_metadata,
            file_ids=file_ids,
            doc_ids=doc_ids,
            index_name="example_index",
            collection_dir=index_dir,
            overwrite=True  # Overwrite the index if it already exists
        )
        print(f"Index created at: {index_path}")
    except Exception as e:
        print(f"Error during indexing: {e}")
        
    # Demonstrate create_index and overwrite behavior
    print("\nDemonstrating create_index and overwrite behavior:")
    
    print("\n1. Creating an empty index...")
    try:
        # Create an empty index
        empty_index_path = indexer.create_index(
            index_name="empty_index",
            index_dir=index_dir
        )
        print(f"  Empty index created at: {empty_index_path}")
        
        # Add a document to the empty index
        print("  Adding a document to the empty index...")
        doc_id = indexer.add_document(
            index_path=empty_index_path,
            document="This document was added to an initially empty index.",
            doc_id="first-doc",
            file_id="empty-test.pdf",
            metadata={"author": "Empty Index Tester", "year": 2023}
        )
        print(f"  Added document with ID: {doc_id}")
        
        # Search the index with the added document
        print("  Searching the previously empty index:")
        empty_searcher = ColbertSearcher(index_path=empty_index_path, device="cpu")
        results = empty_searcher.search("empty index", k=1)
        
        if results:
            print("  Results from previously empty index:")
            print(f"  1. [score: {results[0]['score']}] {results[0]['text']}")
            print(f"     Author: {results[0]['metadata'].get('author')}")
            print(f"     Year: {results[0]['metadata'].get('year')}")
        else:
            print("  No results found")
    except Exception as e:
        print(f"  Error testing empty index: {e}")
    
    print("\n2. Attempting to create the same index without overwrite...")
    try:
        # Try to create the same empty index without overwrite
        indexer.create_index(
            index_name="empty_index",
            index_dir=index_dir,
            overwrite=False  # Don't overwrite
        )
        print("  Index created (this shouldn't happen)")
    except ValueError as e:
        print(f"  Expected error: {e}")
        
    print("\n3. Creating the same index with overwrite=True...")
    try:
        # Create the same index with overwrite=True
        new_docs = ["This is a completely new document that overwrites the previous index"]
        new_metadata = [{"author": "Overwrite Author", "year": 2025}]
        index_path = indexer.index(
            documents=new_docs,
            document_metadata=new_metadata,
            index_name="example_index",
            collection_dir=index_dir,
            overwrite=True  # Overwrite the index
        )
        print(f"  Index overwritten at: {index_path}")
        
        # Verify the overwrite by searching
        print("\n  Verifying overwrite by searching the new index:")
        searcher = ColbertSearcher(index_path=index_path, device="cpu")
        results = searcher.search("new document", k=1)
        
        if results:
            print("  Results from overwritten index:")
            print(f"  1. [score: {results[0]['score']}] {results[0]['text']}")
            print(f"     Author: {results[0]['metadata'].get('author')}")
            print(f"     Year: {results[0]['metadata'].get('year')}")
        else:
            print("  No results found (this shouldn't happen)")
    except Exception as e:
        print(f"  Error during overwrite test: {e}")
        
    # Recreate the original index for the rest of the example
    print("\n4. Recreating the original index for the remaining examples...")
    try:
        # Recreate the original index
        index_path = indexer.index(
            documents=documents,
            document_metadata=document_metadata,
            file_ids=file_ids,
            doc_ids=doc_ids,
            index_name="example_index",
            collection_dir=index_dir,
            overwrite=True
        )
        print(f"  Original index recreated at: {index_path}")
    except Exception as e:
        print(f"  Error recreating original index: {e}")
    
    # Initialize the searcher and search only if indexing was successful
    if 'index_path' in locals():
        try:
            print("Initializing the searcher...")
            searcher = ColbertSearcher(
                index_path=index_path,
                device="cpu"
            )
            
            # Search for documents with different thresholds
            print("Searching documents...")
            queries = [
                "How does ColBERT work?",
                "Who developed ColBERT?",
                "What is late interaction?"
            ]
            
            thresholds = [0.0, 0.3, 0.7]
            
            for query in queries:
                print(f"\nQuery: {query}")
                
                for threshold in thresholds:
                    print(f"\n  With threshold = {threshold}:")
                    try:
                        results = searcher.search(query, k=5, threshold=threshold)
                        
                        if results:
                            print("  Results:")
                            for i, result in enumerate(results, 1):
                                print(f"  {i}. [score: {result['score']}] {result['text']}")
                                print(f"     Document ID: {result['id']}")
                                print(f"     Index: {result['index']}")
                                print(f"     File ID: {result['file_id']}")
                                print(f"     Metadata: Author: {result['metadata'].get('author')}, " 
                                      f"Year: {result['metadata'].get('year')}, "
                                      f"Category: {result['metadata'].get('category')}")
                                print(f"     Keywords: {', '.join(result['metadata'].get('keywords', []))}")
                        else:
                            print("  No results found above this threshold.")
                    except Exception as e:
                        print(f"  Error during search: {e}")
                        
            # Demonstrate document removal feature
            print("\nDemonstrating document removal:")
            
            # First, remove a document by doc_id
            print("\n1. Removing document with ID 'doc-001'...")
            try:
                removed_ids = indexer.remove_documents(index_path, doc_id="doc-001")
                print(f"Removed document IDs: {removed_ids}")
                
                # Reload the searcher and search again
                print("\nAfter removal, searching again:")
                searcher.reload_index(index_path)
                results = searcher.search("How does ColBERT work?", k=5)
                
                if results:
                    print("  Results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. [score: {result['score']}] {result['text']}")
                        print(f"     Document ID: {result['id']}")
                        print(f"     File ID: {result['file_id']}")
            except Exception as e:
                print(f"Error during doc_id removal: {e}")
                
            # Next, remove a document by file_id
            print("\n2. Removing document with file_id 'methods.pdf'...")
            try:
                removed_ids = indexer.remove_documents(index_path, file_id="methods.pdf")
                print(f"Removed document IDs: {removed_ids}")
                
                # Reload the searcher and search again
                print("\nAfter second removal, searching again:")
                searcher.reload_index(index_path)
                results = searcher.search("What is late interaction?", k=5)
                
                if results:
                    print("  Results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. [score: {result['score']}] {result['text']}")
                        print(f"     Document ID: {result['id']}")
                        print(f"     File ID: {result['file_id']}")
            except Exception as e:
                print(f"Error during file_id removal: {e}")
                
            # Now, add new documents
            print("\n3. Adding a single new document...")
            try:
                # Add a single document
                new_doc = "ColBERT leverages BERT-based representations for efficient and effective passage retrieval."
                new_metadata = {
                    "author": "New Author",
                    "year": 2023,
                    "category": "explanation",
                    "keywords": ["colbert", "retrieval", "overview"]
                }
                
                doc_id = indexer.add_document(
                    index_path=index_path,
                    document=new_doc,
                    doc_id="new-doc-1",
                    file_id="overview.pdf",
                    metadata=new_metadata
                )
                print(f"Added document with ID: {doc_id}")
                
                # Reload the searcher and search for the new document
                print("\nAfter addition, searching for the new document:")
                searcher.reload_index(index_path)
                results = searcher.search("ColBERT retrieval", k=5)
                
                if results:
                    print("  Results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. [score: {result['score']}] {result['text']}")
                        print(f"     Document ID: {result['id']}")
                        print(f"     File ID: {result['file_id']}")
            except Exception as e:
                print(f"Error during document addition: {e}")
                
            # Add multiple documents at once
            print("\n4. Adding multiple documents at once...")
            try:
                new_docs = [
                    "ColBERT has been extended to support various languages beyond English.",
                    "The architecture of ColBERT allows for efficient scaling with large document collections."
                ]
                
                new_metadata_list = [
                    {
                        "author": "Multilingual Team",
                        "year": 2023,
                        "category": "multilingual",
                        "keywords": ["colbert", "multilingual", "languages"]
                    },
                    {
                        "author": "Scaling Team",
                        "year": 2024,
                        "category": "performance",
                        "keywords": ["scaling", "performance", "large collections"]
                    }
                ]
                
                doc_ids = indexer.add_documents(
                    index_path=index_path,
                    documents=new_docs,
                    doc_ids=["multi-doc", "scaling-doc"],
                    file_ids=["multilingual.pdf", "scaling.pdf"],
                    metadata_list=new_metadata_list
                )
                print(f"Added document IDs: {doc_ids}")
                
                # Reload the searcher and search for all documents
                print("\nAfter adding multiple documents, searching all documents:")
                searcher.reload_index(index_path)
                results = searcher.search("ColBERT", k=10)
                
                if results:
                    print("  Results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. [score: {result['score']}] {result['text']}")
                        print(f"     Document ID: {result['id']}")
                        print(f"     File ID: {result['file_id']}")
                
                # Demonstrate file_id filtering
                print("\n5. Demonstrating file_id filtering:")
                
                # Single file filtering
                print("\nSearching with single file filter (multilingual.pdf):")
                results = searcher.search("ColBERT", k=10, filter_by_files="multilingual.pdf")
                
                if results:
                    print("  Results (filtered by file=multilingual.pdf):")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. [score: {result['score']}] {result['text']}")
                        print(f"     Document ID: {result['id']}")
                        print(f"     File ID: {result['file_id']}")
                else:
                    print("  No results found with file=multilingual.pdf")
                
                # Multiple files filtering
                print("\nSearching with multiple files filter (overview.pdf, scaling.pdf):")
                results = searcher.search("ColBERT", k=10, filter_by_files=["overview.pdf", "scaling.pdf"])
                
                if results:
                    print("  Results (filtered by file_ids=[overview.pdf, scaling.pdf]):")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. [score: {result['score']}] {result['text']}")
                        print(f"     Document ID: {result['id']}")
                        print(f"     File ID: {result['file_id']}")
                else:
                    print("  No results found with file_ids=[overview.pdf, scaling.pdf]")
            except Exception as e:
                print(f"Error during multiple document addition: {e}")
                
        except Exception as e:
            print(f"Error initializing searcher: {e}")
    else:
        print("Skipping search because indexing failed")


if __name__ == "__main__":
    main()