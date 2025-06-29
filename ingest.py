"""
Document ingestion script for the RAG application.
"""

import os
import sys
from typing import List
import logging
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import OllamaEmbeddings

from utils import (
    load_document, 
    split_documents, 
    get_embedding_model,
    create_chroma_client,
    get_collection_name,
    check_multimodal_capabilities,
    get_image_analysis_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_document(file_path: str, 
                   chunk_size: int = 1024, 
                   chunk_overlap: int = 200,
                   enable_multimodal: bool = True,
                   use_ollama_embeddings: bool = False) -> bool:
    """
    Ingest a document into the vector store.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        enable_multimodal: Whether to enable image extraction for PDFs
        use_ollama_embeddings: Whether to use Ollama embeddings instead of FastEmbed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Starting ingestion of: {file_path}")
        
        # Check multimodal capabilities
        if enable_multimodal and not check_multimodal_capabilities():
            logger.warning("Multimodal capabilities not available. Falling back to text-only processing.")
            enable_multimodal = False
        
        # Load document with multimodal processing if enabled
        documents = load_document(file_path, enable_multimodal=enable_multimodal)
        logger.info(f"Loaded {len(documents)} document(s)")
        
        # Split documents into chunks
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Get image analysis summary for PDFs
        if enable_multimodal and file_path.endswith('.pdf'):
            image_summary = get_image_analysis_summary(file_path)
            if image_summary:
                logger.info("Image analysis completed")
                logger.info(f"Image summary: {image_summary[:200]}...")
        
        # Choose embedding model
        if use_ollama_embeddings:
            embedding_model = OllamaEmbeddings(model="nomic-embed-text")
            logger.info("Using Ollama embeddings")
        else:
            embedding_model = get_embedding_model()
            logger.info("Using FastEmbed embeddings")
        
        # Create ChromaDB client
        client = create_chroma_client()
        
        # Get collection name
        collection_name = get_collection_name(file_path)
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare documents for storage
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
        
        # Add documents to collection
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully ingested {len(chunks)} chunks into collection: {collection_name}")
        
        # Log multimodal information
        if enable_multimodal:
            multimodal_chunks = [chunk for chunk in chunks if chunk.metadata.get('multimodal', False)]
            if multimodal_chunks:
                logger.info(f"Processed {len(multimodal_chunks)} multimodal chunks with image content")
        
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting document {file_path}: {e}")
        return False


def ingest_directory(directory_path: str, 
                    chunk_size: int = 1024, 
                    chunk_overlap: int = 200,
                    enable_multimodal: bool = True,
                    use_ollama_embeddings: bool = False) -> dict:
    """
    Ingest all supported documents in a directory.
    
    Args:
        directory_path: Path to the directory containing documents
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        enable_multimodal: Whether to enable image extraction for PDFs
        use_ollama_embeddings: Whether to use Ollama embeddings instead of FastEmbed
        
    Returns:
        Dictionary with ingestion results
    """
    supported_extensions = {'.pdf', '.md', '.txt'}
    results = {
        'successful': [],
        'failed': [],
        'total_files': 0,
        'total_chunks': 0
    }
    
    logger.info(f"Starting directory ingestion: {directory_path}")
    
    # Check multimodal capabilities
    if enable_multimodal:
        if check_multimodal_capabilities():
            logger.info("Multimodal capabilities available")
        else:
            logger.warning("Multimodal capabilities not available. Falling back to text-only processing.")
            enable_multimodal = False
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            results['total_files'] += 1
            
            logger.info(f"Processing file: {filename}")
            
            if ingest_document(
                file_path, 
                chunk_size, 
                chunk_overlap, 
                enable_multimodal,
                use_ollama_embeddings
            ):
                results['successful'].append(filename)
                
                # Count chunks for this file
                try:
                    documents = load_document(file_path, enable_multimodal)
                    chunks = split_documents(documents, chunk_size, chunk_overlap)
                    results['total_chunks'] += len(chunks)
                except:
                    pass
            else:
                results['failed'].append(filename)
    
    logger.info(f"Ingestion complete. Successful: {len(results['successful'])}, Failed: {len(results['failed'])}")
    return results


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_or_directory_path> [chunk_size] [chunk_overlap] [enable_multimodal] [use_ollama_embeddings]")
        print("Example: python ingest.py docs/ 1024 200 true false")
        sys.exit(1)
    
    path = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    chunk_overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    enable_multimodal = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
    use_ollama_embeddings = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else False
    
    if os.path.isfile(path):
        success = ingest_document(path, chunk_size, chunk_overlap, enable_multimodal, use_ollama_embeddings)
        if success:
            print(f"Successfully ingested: {path}")
        else:
            print(f"Failed to ingest: {path}")
            sys.exit(1)
    elif os.path.isdir(path):
        results = ingest_directory(path, chunk_size, chunk_overlap, enable_multimodal, use_ollama_embeddings)
        print(f"Ingestion results:")
        print(f"  Total files: {results['total_files']}")
        print(f"  Successful: {len(results['successful'])}")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Total chunks: {results['total_chunks']}")
        
        if results['failed']:
            print(f"Failed files: {', '.join(results['failed'])}")
            sys.exit(1)
    else:
        print(f"Path does not exist: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main() 