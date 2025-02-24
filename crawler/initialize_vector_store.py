from utils.vector_store import initialize_vector_store
import os

def main():
    """Initialize or update the vector store with PDF documents."""
    # Get the base directory (project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use tests/downloads directory instead of data
    data_dir = os.path.join(base_dir, "tests", "downloads")
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
        
    print(f"Using PDF directory: {data_dir}")
    print("Initializing vector store...")
    vector_store = initialize_vector_store(base_dir, data_dir)
    
    # Test some searches
    print("\nTesting vector store searches...")
    
    print("\nSearching for emissions data:")
    emissions_results = vector_store.search_emissions_data(k=2)
    for i, doc in enumerate(emissions_results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Type: {doc.metadata.get('chunk_type', 'Unknown')}")
        print(f"Preview: {doc.page_content[:200]}...")
    
    print("\nSearching for targets and commitments:")
    target_results = vector_store.search_targets_and_commitments(k=2)
    for i, doc in enumerate(target_results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Type: {doc.metadata.get('chunk_type', 'Unknown')}")
        print(f"Preview: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 