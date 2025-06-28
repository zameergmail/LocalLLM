"""
Document ingestion module for processing and storing documents in the vector database.
"""

import os
import logging
from typing import List, Optional
from langchain.schema import Document
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

from utils import (
    load_document, 
    split_documents, 
    get_embedding_model,
    create_chroma_client,
    get_collection_name
)

# Set environment variable to fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles document ingestion and vector store management."""
    
    def __init__(self, persist_directory: str = "chroma"):
        """
        Initialize the document ingester.
        
        Args:
            persist_directory: Directory to persist the vector store
        """
        self.persist_directory = persist_directory
        self.embedding_model = get_embedding_model()
        self.chroma_client = create_chroma_client(persist_directory)
        
    def ingest_document(self, file_path: str, 
                       chunk_size: int = 1024, 
                       chunk_overlap: int = 200) -> bool:
        """
        Ingest a single document into the vector store.
        
        Args:
            file_path: Path to the document file
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading document: {file_path}")
            
            # Load the document
            documents = load_document(file_path)
            logger.info(f"Loaded {len(documents)} pages/sections")
            
            # Split into chunks
            chunks = split_documents(documents, chunk_size, chunk_overlap)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Create collection name
            collection_name = get_collection_name(file_path)
            
            # Create or get collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"source": file_path}
            )
            
            # Prepare documents for storage
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
            
            # Add to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {len(chunks)} chunks into collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}")
            return False
    
    def get_vector_store(self, collection_name: Optional[str] = None):
        """
        Get a LangChain vector store instance.
        
        Args:
            collection_name: Name of the collection to use
            
        Returns:
            Chroma vector store instance
        """
        try:
            if collection_name:
                return Chroma(
                    client=self.chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embedding_model
                )
            else:
                return Chroma(
                    client=self.chroma_client,
                    embedding_function=self.embedding_model
                )
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            return [col.name for col in self.chroma_client.list_collections()]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False
    
    def clear_all_collections(self) -> bool:
        """
        Clear all collections.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            collections = self.list_collections()
            for collection_name in collections:
                self.delete_collection(collection_name)
            logger.info("Cleared all collections")
            return True
        except Exception as e:
            logger.error(f"Error clearing collections: {str(e)}")
            return False


def ingest_documents_from_directory(directory: str = "docs", 
                                   chunk_size: int = 1024, 
                                   chunk_overlap: int = 200) -> List[str]:
    """
    Ingest all documents from a directory.
    
    Args:
        directory: Directory containing documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of successfully ingested file paths
    """
    ingester = DocumentIngester()
    successful_files = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return successful_files
    
    supported_extensions = {'.pdf', '.md', '.txt'}
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            if ingester.ingest_document(file_path, chunk_size, chunk_overlap):
                successful_files.append(file_path)
    
    return successful_files


if __name__ == "__main__":
    # Example usage
    successful_files = ingest_documents_from_directory()
    print(f"Successfully ingested {len(successful_files)} files")
    for file_path in successful_files:
        print(f"  - {file_path}") 