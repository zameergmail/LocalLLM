"""
Main Streamlit application for the RAG system with multimodal capabilities.
"""

import streamlit as st
import os
import tempfile
from typing import List, Optional, Dict, Any
import logging
import shutil

from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from utils import (
    load_document,
    split_documents,
    get_embedding_model,
    create_chroma_client,
    get_collection_name,
    save_uploaded_file,
    check_multimodal_capabilities,
    get_image_analysis_summary
)

# Set environment variable to fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable ChromaDB telemetry to reduce log noise
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Local RAG with Multimodal Capabilities",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .multimodal-info {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
    .image-analysis {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'collections' not in st.session_state:
        st.session_state.collections = []
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'multimodal_enabled' not in st.session_state:
        st.session_state.multimodal_enabled = check_multimodal_capabilities()


def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        import ollama
        response = ollama.list()
        
        # The API returns a ListResponse object with a 'models' attribute
        if hasattr(response, 'models'):
            models = response.models
        else:
            models = response
        
        # Extract model names
        model_names = []
        for model in models:
            if hasattr(model, 'model'):
                model_names.append(model.model)
            elif hasattr(model, 'name'):
                model_names.append(model.name)
            elif isinstance(model, dict) and 'name' in model:
                model_names.append(model['name'])
            else:
                # Fallback: try to get the name from the model object
                model_names.append(str(model))
        
        return model_names if model_names else ["mistral", "llama2", "deepseek-coder"]
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        return ["mistral", "llama2", "deepseek-coder"]


def create_qa_chain(collection_name: str, model_name: str, use_ollama_embeddings: bool = False):
    """Create a QA chain for the specified collection."""
    try:
        # Create ChromaDB client
        client = create_chroma_client()
        
        # Get collection
        collection = client.get_collection(collection_name)
        
        # Choose embedding model
        if use_ollama_embeddings:
            embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        else:
            embedding_model = get_embedding_model()
        
        # Create vector store
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model
        )
        
        # Create LLM
        llm = OllamaLLM(model=model_name, temperature=0.1)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant. Use the following context to answer the question.
            If the context contains image analysis, consider both text and visual information.
            
            Context: {context}
            
            Question: {question}
            
            Answer: """
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        return None


def process_uploaded_file(uploaded_file, enable_multimodal: bool = True, use_ollama_embeddings: bool = False, use_heavy_models: bool = False):
    """Process an uploaded file and add it to the vector store."""
    try:
        # Save uploaded file
        file_path = save_uploaded_file(uploaded_file)
        
        # Load and process document
        documents = load_document(file_path, enable_multimodal)
        chunks = split_documents(documents)
        
        # Choose embedding model
        if use_ollama_embeddings:
            embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        else:
            embedding_model = get_embedding_model()
        
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
        
        # Get image analysis summary for PDFs
        image_summary = ""
        if enable_multimodal and file_path.endswith('.pdf'):
            image_summary = get_image_analysis_summary(file_path, use_heavy_models)
        
        return {
            'success': True,
            'collection_name': collection_name,
            'chunks': len(chunks),
            'image_summary': image_summary,
            'multimodal': any(chunk.metadata.get('multimodal', False) for chunk in chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Local RAG with Multimodal Capabilities</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select Ollama Model",
            available_models,
            key="model_selector"
        )
        
        # Embedding model selection
        use_ollama_embeddings = st.checkbox(
            "Use Ollama Embeddings (slower but more accurate)",
            value=False,
            key="embedding_selector"
        )
        
        # Multimodal settings
        st.header("🖼️ Multimodal Settings")
        
        if st.session_state.multimodal_enabled:
            multimodal_enabled = st.checkbox(
                "Enable Image Analysis",
                value=True,
                help="Extract and analyze images from PDFs and standalone image files",
                key="multimodal_checkbox"
            )
            
            # Add lightweight mode option
            use_heavy_models = st.checkbox(
                "Use Heavy Vision Models (may cause memory issues)",
                value=False,
                help="Enable advanced image analysis with heavy models. May cause memory issues on some systems.",
                key="heavy_models_checkbox"
            )
            
            st.markdown("""
            <div class="multimodal-info">
                <strong>Multimodal Features:</strong>
                <ul>
                    <li>Image extraction from PDFs</li>
                    <li>Direct image file analysis (PNG, JPG, GIF, etc.)</li>
                    <li>Basic image analysis (size, format, orientation)</li>
                    <li>Advanced analysis with heavy models (optional)</li>
                    <li>Combined text and image understanding</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Multimodal capabilities not available")
            st.info("Install required packages: transformers, torch, PIL")
            multimodal_enabled = False
            use_heavy_models = False
        
        # Chunking settings
        st.header("📄 Document Processing")
        chunk_size = st.slider("Chunk Size", 512, 2048, 1024, step=128)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, step=50)
        
        # Collection management
        st.header("🗂️ Collections")
        
        # Get existing collections
        try:
            client = create_chroma_client()
            collections = [col.name for col in client.list_collections()]
            st.session_state.collections = collections
            
            if collections:
                selected_collection = st.selectbox(
                    "Select Collection",
                    collections,
                    key="collection_selector"
                )
                st.session_state.current_collection = selected_collection
                
                # Delete collection button
                if st.button("🗑️ Delete Selected Collection", key="delete_collection"):
                    try:
                        client.delete_collection(selected_collection)
                        st.success(f"Deleted collection: {selected_collection}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting collection: {e}")
            else:
                st.info("No collections found. Upload documents to create collections.")
                
        except Exception as e:
            st.error(f"Error loading collections: {e}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Documents")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'md', 'txt', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} file(s)")
            
            # Process files
            if st.button("🚀 Process Documents", key="process_button"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    result = process_uploaded_file(
                        uploaded_file, 
                        multimodal_enabled, 
                        use_ollama_embeddings,
                        use_heavy_models
                    )
                    results.append(result)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                
                # Display results
                for i, (uploaded_file, result) in enumerate(zip(uploaded_files, results)):
                    if result['success']:
                        st.success(f"✅ {uploaded_file.name}")
                        st.write(f"  - Collection: {result['collection_name']}")
                        st.write(f"  - Chunks: {result['chunks']}")
                        
                        if result.get('multimodal'):
                            st.write("  - 🖼️ Multimodal content detected")
                        
                        if result.get('image_summary'):
                            with st.expander("🖼️ Image Analysis Summary"):
                                st.text(result['image_summary'])
                    else:
                        st.error(f"❌ {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                
                st.rerun()
    
    with col2:
        st.header("💬 Chat Interface")
        
        if st.session_state.current_collection:
            # Create QA chain
            qa_chain = create_qa_chain(
                st.session_state.current_collection,
                selected_model,
                use_ollama_embeddings
            )
            
            if qa_chain:
                # Chat input
                user_question = st.text_input(
                    "Ask a question about your documents:",
                    key="user_question"
                )
                
                if user_question:
                    if st.button("🔍 Search", key="search_button"):
                        with st.spinner("Searching..."):
                            try:
                                # Get answer using invoke instead of run
                                answer = qa_chain.invoke({"query": user_question})
                                
                                # Add to chat history
                                st.session_state.chat_history.append({
                                    'question': user_question,
                                    'answer': answer
                                })
                                
                                # Display answer
                                st.markdown("### Answer:")
                                st.write(answer)
                                
                                # Display chat history
                                if st.session_state.chat_history:
                                    st.markdown("### Chat History:")
                                    for i, chat in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                                        with st.expander(f"Q: {chat['question'][:50]}..."):
                                            st.write(f"**Question:** {chat['question']}")
                                            st.write(f"**Answer:** {chat['answer']}")
                                
                            except Exception as e:
                                st.error(f"Error getting answer: {e}")
            else:
                st.error("Failed to create QA chain")
        else:
            st.info("Please upload and process documents first, then select a collection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with Streamlit, LangChain, Ollama, and ChromaDB</p>
        <p>Supports PDF, Markdown, text files, and images (PNG, JPG, GIF, etc.) with multimodal analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 