"""
Example of filtered search with RenRAG ColBERT.

This example demonstrates:
- Creating an index with documents and file IDs
- Searching with threshold filtering
- Searching with file ID filtering
- Combining both filtering techniques
"""

from renrag_colbert import ColbertIndexer, ColbertSearcher

def main():
    print("RenRAG ColBERT Filtered Search Example")
    print("-" * 60)
    
    # Initialize the indexer
    print("Initializing indexer...")
    indexer = ColbertIndexer(device="cpu")
    
    # Sample documents with file IDs
    documents = [
        "Python is a high-level programming language known for its readability.",
        "JavaScript is the primary language for web development.",
        "Java is widely used for enterprise application development.",
        "C++ provides low-level memory manipulation and high performance.",
        "Rust offers memory safety without garbage collection."
    ]
    
    # Associate each document with a file ID
    file_ids = [
        "python_intro.pdf",
        "javascript_basics.pdf",
        "java_enterprise.pdf",
        "cpp_performance.pdf",
        "rust_memory_safety.pdf"
    ]
    
    # Create an index
    print("Creating index with documents and file IDs...")
    index_path = indexer.index(
        documents=documents,
        file_ids=file_ids,
        index_name="filtered_example",
        overwrite=True
    )
    print(f"Index created at: {index_path}")
    
    # Initialize the searcher
    print("\nInitializing searcher...")
    searcher = ColbertSearcher(index_path=index_path, device="cpu")
    
    # Define a query
    query = "programming language features"
    
    # 1. Basic search without filtering
    print(f"\n1. Basic search for: \"{query}\"")
    results = searcher.search(query, k=5)
    print_results(results, "Basic Search Results")
    
    # 2. Search with threshold filtering
    threshold = 0.3
    print(f"\n2. Search with threshold {threshold} for: \"{query}\"")
    threshold_results = searcher.search(query, k=5, threshold=threshold)
    print_results(threshold_results, f"Results with threshold >= {threshold}")
    
    # 3. Search with file ID filtering (single file)
    file_filter = "python_intro.pdf"
    print(f"\n3. Search filtered by file: {file_filter}")
    file_results = searcher.search(
        query, 
        k=5, 
        filter_by_files=file_filter
    )
    print_results(file_results, f"Results filtered by file: {file_filter}")
    
    # 4. Search with file ID filtering (multiple files)
    multi_file_filter = ["javascript_basics.pdf", "rust_memory_safety.pdf"]
    print(f"\n4. Search filtered by files: {', '.join(multi_file_filter)}")
    multi_file_results = searcher.search(
        query, 
        k=5, 
        filter_by_files=multi_file_filter
    )
    print_results(multi_file_results, "Results filtered by multiple files")
    
    # 5. Combine threshold and file ID filtering
    print(f"\n5. Search with threshold {threshold} and filtered by files")
    combined_results = searcher.search(
        query, 
        k=5, 
        threshold=threshold,
        filter_by_files=multi_file_filter
    )
    print_results(combined_results, "Results with combined filtering")

def print_results(results, title):
    print(f"\n{title}:")
    print("-" * 60)
    if not results:
        print("No results found")
    else:
        for i, result in enumerate(results, 1):
            print(f"{i}. [Score: {result['score']}] {result['text']}")
            print(f"   File: {result['file_id']}")
        print(f"Total results: {len(results)}")

if __name__ == "__main__":
    main()