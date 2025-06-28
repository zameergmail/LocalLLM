"""
Utility functions for document processing, chunking, and embeddings.
"""

import os
from typing import List, Optional
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings

# Set environment variable to fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FastEmbedWrapper:
    """Wrapper for FastEmbed to make it compatible with LangChain."""
    
    def __init__(self):
        self.embedding_model = TextEmbedding()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = list(self.embedding_model.embed(texts))
        return [embedding.tolist() for embedding in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = list(self.embedding_model.embed([text]))[0]
        return embedding.tolist()


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from file path.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.md') or file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return loader.load()


def split_documents(documents: List[Document], 
                   chunk_size: int = 1024, 
                   chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks for better processing.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)


def get_embedding_model():
    """
    Get the FastEmbed embedding model wrapped for LangChain compatibility.
    
    Returns:
        FastEmbedWrapper instance
    """
    return FastEmbedWrapper()


def create_chroma_client(persist_directory: str = "chroma"):
    """
    Create and configure ChromaDB client.
    
    Args:
        persist_directory: Directory to persist the vector store
        
    Returns:
        ChromaDB client instance
    """
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
    )


def get_collection_name(file_path: str) -> str:
    """
    Generate a collection name from file path.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Collection name string
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return f"doc_{base_name.lower().replace(' ', '_')}"


def save_uploaded_file(uploaded_file, docs_dir: str = "docs") -> str:
    """
    Save an uploaded file to the docs directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        docs_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Create docs directory if it doesn't exist
    os.makedirs(docs_dir, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(docs_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path 