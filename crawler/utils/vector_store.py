from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import shutil
import json

load_dotenv()

class ESGVectorStore:
    def __init__(self, base_dir: str):
        """Initialize ESG Vector Store."""
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.db_dir = os.path.join(base_dir, "chroma_db")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectordb = None
        
    def load_and_process_pdfs(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Load PDFs from directory and split into chunks with metadata."""
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Enhance chunks with ESG-specific metadata
        enhanced_chunks = []
        for chunk in chunks:
            # Extract potential ESG metrics from chunk
            metrics = self._extract_esg_metrics(chunk.page_content)
            
            # Add metadata
            chunk.metadata.update({
                'metrics': metrics,
                'chunk_type': self._determine_chunk_type(chunk.page_content),
                'processed_date': os.path.getmtime(chunk.metadata.get('source', '')),
            })
            enhanced_chunks.append(chunk)
            
        return enhanced_chunks
    
    def _extract_esg_metrics(self, text: str) -> Dict:
        """Extract potential ESG metrics from text."""
        metrics = {
            'has_emissions_data': bool(any(x in text.lower() for x in ['scope 1', 'scope 2', 'scope 3', 'ghg emissions'])),
            'has_targets': bool(any(x in text.lower() for x in ['target', 'goal', 'commitment'])),
            'has_financial_data': bool(any(x in text.lower() for x in ['$', 'usd', 'million', 'billion'])),
            'has_renewable_energy': bool('renewable' in text.lower()),
        }
        return metrics
    
    def _determine_chunk_type(self, text: str) -> str:
        """Determine the type of content in the chunk."""
        text_lower = text.lower()
        if any(x in text_lower for x in ['table', 'figure']):
            return 'table_or_figure'
        elif any(x in text_lower for x in ['scope 1', 'scope 2', 'scope 3', 'ghg emissions']):
            return 'emissions_data'
        elif any(x in text_lower for x in ['target', 'goal', 'commitment']):
            return 'targets'
        elif any(x in text_lower for x in ['$', 'usd', 'million', 'billion']):
            return 'financial'
        else:
            return 'general'
    
    def create_or_update_vector_store(self, chunks: List[Document], force_refresh: bool = False):
        """Create or update Chroma vector store."""
        if force_refresh and os.path.exists(self.db_dir):
            print(f"Clearing existing vector store at {self.db_dir}")
            shutil.rmtree(self.db_dir)
        
        if not os.path.exists(self.db_dir):
            print("Creating new vector store...")
            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
        else:
            print("Loading existing vector store...")
            self.vectordb = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
            # Add new documents
            print("Updating vector store with new documents...")
            self.vectordb.add_documents(chunks)
        
        self.vectordb.persist()
        print(f"Vector store {'created' if force_refresh else 'updated'} at {self.db_dir}")
    
    def search_emissions_data(self, company_name: Optional[str] = None, year: Optional[int] = None, k: int = 5) -> List[Document]:
        """Search for emissions-related data with optional filters."""
        query = "emissions data scope 1 scope 2 scope 3 greenhouse gas"
        filter_dict = {}
        
        if company_name:
            filter_dict["company"] = company_name
        if year:
            filter_dict["year"] = year
            
        results = self.vectordb.similarity_search(
            query,
            k=k,
            filter=filter_dict if filter_dict else None
        )
        return results
    
    def search_targets_and_commitments(self, k: int = 5) -> List[Document]:
        """Search for targets and commitments."""
        query = "emission reduction targets net zero commitments renewable energy goals"
        return self.vectordb.similarity_search(query, k=k)
    
    def search_financial_metrics(self, k: int = 5) -> List[Document]:
        """Search for financial metrics and investments."""
        query = "sustainable finance green investments climate-related financial metrics"
        return self.vectordb.similarity_search(query, k=k)

def initialize_vector_store(base_dir: str, data_dir: str = None) -> ESGVectorStore:
    """Initialize and return an ESG Vector Store instance."""
    vector_store = ESGVectorStore(base_dir)
    # Override data directory if provided
    if data_dir:
        vector_store.data_dir = data_dir
    chunks = vector_store.load_and_process_pdfs()
    vector_store.create_or_update_vector_store(chunks)
    return vector_store

def load_and_process_pdfs(data_dir: str):
    """Load PDFs from directory and split into chunks."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory: str):
    """Create and persist Chroma vector store."""
    # Clear existing vector store if it exists
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and persist Chroma vector store
    print("Creating new vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def main():
    # Define directories
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # Process PDFs
    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Created {len(chunks)} chunks from PDFs")
    
    # Create vector store
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

if __name__ == "__main__":
    main()