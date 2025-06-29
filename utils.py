"""
Utility functions for document processing, chunking, and embeddings.
"""

import os
import logging
from typing import List, Dict, Any
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings

# Set environment variables to reduce warnings and noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Configure logging
logger = logging.getLogger(__name__)


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


def load_document(file_path: str, enable_multimodal: bool = True) -> List[Document]:
    """Load document from file path."""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Handle image files
        if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']:
            if enable_multimodal:
                return load_image_document(file_path)
            else:
                # If multimodal is disabled, create a basic document with filename
                return [Document(
                    page_content=f"Image file: {os.path.basename(file_path)}",
                    metadata={"source": file_path, "type": "image", "multimodal": False}
                )]
        
        # Handle PDF files
        elif file_extension == '.pdf':
            if enable_multimodal:
                return load_pdf_with_images(file_path)
            else:
                return load_pdf_text_only(file_path)
        
        # Handle text files
        elif file_extension in ['.txt', '.md']:
            return load_text_document(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        return [Document(
            page_content=f"Error loading document: {str(e)}",
            metadata={"source": file_path, "error": str(e)}
        )]


def load_image_document(file_path: str) -> List[Document]:
    """Load and analyze an image file."""
    try:
        from PIL import Image
        from multimodal_utils import get_image_analysis_summary
        
        # Load image
        image = Image.open(file_path)
        
        # Get basic image information
        image_info = {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'size_bytes': os.path.getsize(file_path)
        }
        
        # Analyze image content (lightweight mode by default)
        analysis_summary = get_image_analysis_summary(file_path, use_heavy_models=False)
        
        # Create document content
        content = f"""
Image Analysis:
- File: {os.path.basename(file_path)}
- Dimensions: {image_info['width']}x{image_info['height']} pixels
- Format: {image_info['format']} ({image_info['mode']})
- Size: {image_info['size_bytes']} bytes

Content Analysis:
{analysis_summary}
        """.strip()
        
        return [Document(
            page_content=content,
            metadata={
                "source": file_path,
                "type": "image",
                "multimodal": True,
                "image_info": image_info
            }
        )]
        
    except Exception as e:
        logger.error(f"Error loading image document {file_path}: {e}")
        return [Document(
            page_content=f"Error analyzing image: {str(e)}",
            metadata={"source": file_path, "type": "image", "error": str(e)}
        )]


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


def check_multimodal_capabilities() -> bool:
    """
    Check if multimodal capabilities are available.
    
    Returns:
        True if multimodal processing is available
    """
    try:
        import transformers
        import torch
        import PIL
        return True
    except ImportError:
        return False


def get_image_analysis_summary(file_path: str) -> str:
    """
    Get a summary of image analysis for a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Summary string of image content
    """
    if not file_path.endswith('.pdf'):
        return ""
    
    try:
        from multimodal_utils import MultimodalDocumentProcessor
        processor = MultimodalDocumentProcessor()
        multimodal_data = processor.process_pdf_with_images(file_path)
        return multimodal_data.get('multimodal_summary', "")
    except ImportError:
        return "Multimodal processing not available"
    except Exception as e:
        return f"Error processing images: {str(e)}"


def load_pdf_with_images(file_path: str) -> List[Document]:
    """Load PDF with image extraction and analysis."""
    try:
        # Load text content
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add multimodal processing
        try:
            from multimodal_utils import MultimodalDocumentProcessor
            processor = MultimodalDocumentProcessor()
            documents = processor.create_multimodal_documents(file_path, documents)
        except ImportError:
            # Fallback to text-only if multimodal utils not available
            pass
        
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF with images {file_path}: {e}")
        return [Document(
            page_content=f"Error loading PDF: {str(e)}",
            metadata={"source": file_path, "error": str(e)}
        )]


def load_pdf_text_only(file_path: str) -> List[Document]:
    """Load PDF text content only."""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading PDF text {file_path}: {e}")
        return [Document(
            page_content=f"Error loading PDF: {str(e)}",
            metadata={"source": file_path, "error": str(e)}
        )]


def load_text_document(file_path: str) -> List[Document]:
    """Load text document (txt, md)."""
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading text document {file_path}: {e}")
        return [Document(
            page_content=f"Error loading text document: {str(e)}",
            metadata={"source": file_path, "error": str(e)}
        )] 