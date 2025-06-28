"""
Local RAG Web App with Ollama, LangChain, and Streamlit
A beginner-friendly web app for chatting with your PDF and Markdown files.
"""

import streamlit as st
import os
import time
from typing import List, Optional
import logging

# LangChain imports
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.schema import Document

# Local imports
from ingest import DocumentIngester
from utils import save_uploaded_file, get_collection_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable to fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Page configuration
st.set_page_config(
    page_title="🧠 Local RAG Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #764ba2;
    }
    .file-upload-area {
        border: 2px dashed #667eea;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ingester' not in st.session_state:
        st.session_state.ingester = DocumentIngester()
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None


def check_ollama_status():
    """Check if Ollama is running and available."""
    try:
        # Try to create an Ollama instance
        llm = OllamaLLM(model="mistral")
        # This will fail if Ollama is not running
        return True
    except Exception as e:
        return False


def get_available_models():
    """Get list of available Ollama models."""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except Exception:
        return []


def chat_with_documents(query: str, collection_name: Optional[str] = None, model_name: Optional[str] = None):
    """Chat with documents using RAG."""
    try:
        # Get vector store
        if collection_name:
            vectorstore = st.session_state.ingester.get_vector_store(collection_name)
        else:
            vectorstore = st.session_state.ingester.get_vector_store()
        
        # Get retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Use the selected model
        if not model_name:
            return "Please select a model from the sidebar."
        
        llm = OllamaLLM(model=model_name)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get response using invoke instead of __call__
        with st.spinner("🤔 Thinking..."):
            result = qa_chain.invoke({"query": query})
        
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        # Format response with sources
        response = f"{answer}\n\n"
        if source_docs:
            response += "**Sources:**\n"
            for i, doc in enumerate(source_docs[:3], 1):
                source = doc.metadata.get('source', 'Unknown')
                response += f"{i}. {source}\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat_with_documents: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Local RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown("### Chat with your PDF and Markdown files using local AI")
    
    # Check Ollama status
    if not check_ollama_status():
        st.error("""
        ⚠️ **Ollama is not running!** 
        
        Please start Ollama and pull a model:
        ```bash
        # Install Ollama (if not already installed)
        brew install ollama
        
        # Start Ollama
        ollama serve
        
        # Pull a model (in another terminal)
        ollama pull mistral
        ```
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['pdf', 'md', 'txt'],
            help="Upload a PDF, Markdown, or text file to chat with",
            key="file_uploader"
        )
        
        if uploaded_file:
            if st.button("📥 Process Document", key="process_doc_btn"):
                with st.spinner("Processing document..."):
                    # Save file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Ingest document
                    success = st.session_state.ingester.ingest_document(file_path)
                    
                    if success:
                        st.success(f"✅ Document processed successfully!")
                        st.session_state.current_collection = get_collection_name(file_path)
                    else:
                        st.error("❌ Failed to process document")
        
        # Collection selection
        st.markdown("## 📚 Available Collections")
        collections = st.session_state.ingester.list_collections()
        
        if collections:
            selected_collection = st.selectbox(
                "Select collection to chat with:",
                ["All Collections"] + collections,
                index=0,
                key="collection_selectbox"
            )
            
            if selected_collection == "All Collections":
                st.session_state.current_collection = None
            else:
                st.session_state.current_collection = selected_collection
            
            # Collection management
            st.markdown("### 🗑️ Collection Management")
            if st.button("Clear All Collections", key="clear_collections_btn"):
                if st.session_state.ingester.clear_all_collections():
                    st.success("All collections cleared!")
                    st.rerun()
                else:
                    st.error("Failed to clear collections")
        else:
            st.info("No collections available. Upload a document to get started!")
        
        # Model selection
        st.markdown("## 🤖 Model Settings")
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "Select Model", 
                available_models,
                key="model_selectbox"
            )
            st.session_state.selected_model = selected_model
            st.info(f"Using model: **{selected_model}**")
        else:
            st.warning("No models found. Pull a model with `ollama pull mistral`")
    
    # Main chat interface
    st.markdown("## 💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = chat_with_documents(
                    prompt, 
                    st.session_state.current_collection,
                    st.session_state.selected_model
                )
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.session_state.messages and st.button("🗑️ Clear Chat", key="clear_chat_btn"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit, LangChain, Ollama, and ChromaDB
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 