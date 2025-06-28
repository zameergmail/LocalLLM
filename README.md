# 🧠 Local RAG Web App with Ollama, LangChain, and Streamlit

A beginner-friendly, fully local web application for chatting with your PDF and Markdown files using Retrieval-Augmented Generation (RAG). This app runs entirely on your **Mac Studio M1 Ultra** with **Ollama** and open-source Python tools — no cloud services or API keys required.

## 🚀 Features

- **🔒 100% Local**: Everything runs on your machine, no data leaves your system
- **📄 Multi-format Support**: Upload PDF, Markdown, and text files
- **🤖 Multiple Models**: Use any Ollama model (Mistral, LLaMA, etc.)
- **💬 Chat Interface**: Modern Streamlit chat interface with message history
- **🔍 Smart Retrieval**: Advanced document chunking and similarity search
- **📚 Collection Management**: Organize documents into collections
- **🎨 Beautiful UI**: Modern, responsive design with gradient styling

## 🏗️ Architecture Overview

The application follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Document       │    │   Vector Store  │
│   (main.py)     │◄──►│  Processing     │◄──►│   (ChromaDB)    │
│                 │    │  (ingest.py)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Pipeline  │    │   Embeddings    │    │   Local LLM     │
│  (LangChain)    │    │  (FastEmbed)    │    │   (Ollama)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🧰 Tech Stack

| Component | Technology | Purpose | Version |
|-----------|------------|---------|---------|
| **UI Framework** | Streamlit | Web interface and file upload | 1.46.1 |
| **LLM Backend** | Ollama | Local language model inference | 0.5.1 |
| **RAG Framework** | LangChain | Document processing and retrieval | 0.3.26 |
| **Embeddings** | FastEmbed | Local, ONNX-based embeddings | 0.7.1 |
| **Vector Database** | ChromaDB | Local vector storage | 1.0.13 |
| **Document Parsing** | PyMuPDF | PDF and text processing | 1.26.1 |

## 📁 Project Structure

```
Local-LLM/
├── main.py              # Main Streamlit application
├── ingest.py            # Document ingestion and processing
├── utils.py             # Utility functions and helpers
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── setup.sh            # Automated setup script
├── test_app.py         # Setup verification script
├── docs/               # Uploaded documents (auto-created)
├── chroma/             # Vector database storage (auto-created)
└── example_documents/  # Sample documents for testing
    └── sample_article.md
```

## 🔄 How It Works: Technical Deep Dive

### 1. Document Processing Pipeline

#### 1.1 Document Loading (`utils.py`)
```python
def load_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.md') or file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()
```

**Process:**
- **PDF Files**: Uses PyMuPDF to extract text from PDF pages
- **Markdown/Text**: Direct text loading with UTF-8 encoding
- **Output**: List of LangChain Document objects with metadata

#### 1.2 Document Chunking (`utils.py`)
```python
def split_documents(documents: List[Document], 
                   chunk_size: int = 1024, 
                   chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)
```

**Process:**
- **Chunk Size**: 1024 characters (configurable)
- **Overlap**: 200 characters to maintain context
- **Separators**: Prioritizes paragraph breaks, then sentences, then words
- **Output**: Smaller, overlapping document chunks

### 2. Embedding Generation

#### 2.1 FastEmbed Wrapper (`utils.py`)
```python
class FastEmbedWrapper:
    def __init__(self):
        self.embedding_model = TextEmbedding()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.embedding_model.embed(texts))
        return [embedding.tolist() for embedding in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        embedding = list(self.embedding_model.embed([text]))[0]
        return embedding.tolist()
```

**Process:**
- **Model**: Uses FastEmbed's ONNX-based embedding model
- **Dimensions**: 384-dimensional vectors
- **Compatibility**: Wrapper makes FastEmbed compatible with LangChain
- **Performance**: Optimized for local inference

### 3. Vector Storage

#### 3.1 ChromaDB Configuration (`utils.py`)
```python
def create_chroma_client(persist_directory: str = "chroma"):
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
    )
```

**Features:**
- **Persistence**: Data stored locally in `chroma/` directory
- **Collections**: Each document gets its own collection
- **Metadata**: Stores source file information
- **Privacy**: No telemetry or external connections

#### 3.2 Document Ingestion (`ingest.py`)
```python
def ingest_document(self, file_path: str, chunk_size: int = 1024, chunk_overlap: int = 200):
    # Load and chunk document
    documents = load_document(file_path)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    
    # Create collection
    collection_name = get_collection_name(file_path)
    collection = self.chroma_client.get_or_create_collection(name=collection_name)
    
    # Store with embeddings
    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[f"{collection_name}_{i}" for i in range(len(chunks))]
    )
```

### 4. RAG Pipeline

#### 4.1 Retrieval-Augmented Generation (`main.py`)
```python
def chat_with_documents(query: str, collection_name: Optional[str] = None, model_name: Optional[str] = None):
    # Get vector store
    vectorstore = st.session_state.ingester.get_vector_store(collection_name)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Initialize LLM
    llm = OllamaLLM(model=model_name)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Generate response
    result = qa_chain.invoke({"query": query})
```

**Process:**
1. **Query Embedding**: Convert user question to vector
2. **Similarity Search**: Find 4 most relevant document chunks
3. **Context Assembly**: Combine retrieved chunks with query
4. **LLM Generation**: Generate answer using local Ollama model
5. **Source Attribution**: Return answer with source references

### 5. User Interface

#### 5.1 Streamlit Session Management (`main.py`)
```python
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ingester' not in st.session_state:
        st.session_state.ingester = DocumentIngester()
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
```

**Features:**
- **Chat History**: Persistent message history during session
- **Document Management**: Centralized ingester instance
- **Collection Selection**: Track current document collection
- **Model Selection**: Remember chosen LLM model

## 🔧 Configuration

### Environment Variables
```bash
# .env file
TOKENIZERS_PARALLELISM=false  # Prevents tokenizer warnings
CHUNK_SIZE=1024              # Document chunk size
CHUNK_OVERLAP=200            # Chunk overlap
PERSIST_DIRECTORY=chroma     # Vector store location
```

### Chunking Parameters
```python
# In utils.py
def split_documents(documents, chunk_size=1024, chunk_overlap=200):
    # Larger chunk_size = more context, slower processing
    # Higher chunk_overlap = better continuity, more storage
```

### Retrieval Settings
```python
# In main.py
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Number of chunks to retrieve
)
```

## 📊 Data Flow

### Document Upload Flow
```
1. User uploads file → Streamlit file_uploader
2. File saved to docs/ → save_uploaded_file()
3. Document loaded → load_document()
4. Text chunked → split_documents()
5. Chunks embedded → FastEmbedWrapper.embed_documents()
6. Stored in ChromaDB → collection.add()
7. Collection created → get_collection_name()
```

### Query Processing Flow
```
1. User asks question → Streamlit chat_input
2. Query embedded → FastEmbedWrapper.embed_query()
3. Similarity search → vectorstore.as_retriever()
4. Context retrieved → Top 4 relevant chunks
5. LLM generates answer → OllamaLLM.invoke()
6. Response formatted → Add source citations
7. Displayed to user → Streamlit chat_message
```

## 🧠 Model Integration

### Ollama Models
```python
# Available models (from ollama list)
- mistral:latest      # 7B parameters, fast inference
- deepseek-r1:70b     # 70B parameters, high quality
- llama2:7b           # 7B parameters, balanced
- codellama:7b        # 7B parameters, code-focused
```

### Model Selection Logic
```python
def get_available_models():
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    # Parse output and return model names
    return [line.split()[0] for line in lines if line.strip()]
```

## 🔍 Search and Retrieval

### Similarity Search
- **Algorithm**: Cosine similarity between query and document embeddings
- **Top-k**: Retrieves 4 most similar chunks
- **Scoring**: Normalized similarity scores (0-1)

### Context Assembly
```python
# Chain type: "stuff" (simple concatenation)
qa_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",  # Alternative: "map_reduce", "refine"
    retriever=retriever,
    return_source_documents=True
)
```

## 🛡️ Privacy and Security

### Data Privacy
- **Local Storage**: All data stored on your machine
- **No Telemetry**: ChromaDB telemetry disabled
- **No External Calls**: All processing local
- **File Isolation**: Each document in separate collection

### Security Features
- **Input Validation**: File type checking
- **Error Handling**: Graceful failure handling
- **Session Isolation**: Streamlit session state
- **Resource Limits**: Configurable chunk sizes

## 📈 Performance Optimization

### Memory Management
- **Chunking**: Reduces memory usage for large documents
- **Batch Processing**: Efficient embedding generation
- **Model Selection**: Choose appropriate model size
- **Collection Management**: Delete unused collections

### Speed Optimization
- **ONNX Models**: FastEmbed uses optimized ONNX runtime
- **Local Inference**: No network latency
- **Caching**: ChromaDB caches embeddings
- **Parallel Processing**: FastEmbed supports batching

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Connection Issues
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve

# Test connection
ollama list
```

#### 2. Memory Issues
```python
# Reduce chunk size for large documents
chunk_size = 512  # Instead of 1024

# Use smaller models
model_name = "mistral"  # Instead of "deepseek-r1:70b"
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt
pip install -r requirements.txt

# Check virtual environment
source venv/bin/activate
```

#### 4. ChromaDB Issues
```bash
# Clear vector store
rm -rf chroma/

# Restart application
streamlit run main.py
```

## 🔄 Development Workflow

### Adding New Features
1. **Document Types**: Add new loaders in `utils.py`
2. **Embedding Models**: Extend `FastEmbedWrapper` class
3. **UI Components**: Add Streamlit widgets in `main.py`
4. **Vector Stores**: Implement new store in `ingest.py`

### Testing
```bash
# Run test script
python test_app.py

# Test specific components
python -c "from utils import load_document; print(load_document('test.pdf'))"
```

## 📚 API Reference

### Core Classes

#### DocumentIngester
```python
class DocumentIngester:
    def ingest_document(self, file_path: str) -> bool
    def get_vector_store(self, collection_name: str)
    def list_collections(self) -> List[str]
    def delete_collection(self, collection_name: str) -> bool
```

#### FastEmbedWrapper
```python
class FastEmbedWrapper:
    def embed_documents(self, texts: List[str]) -> List[List[float]]
    def embed_query(self, text: str) -> List[float]
```

### Key Functions

#### Document Processing
```python
load_document(file_path: str) -> List[Document]
split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]
save_uploaded_file(uploaded_file, docs_dir: str) -> str
```

#### Vector Operations
```python
create_chroma_client(persist_directory: str) -> chromadb.PersistentClient
get_embedding_model() -> FastEmbedWrapper
get_collection_name(file_path: str) -> str
```

## 🎯 Use Cases

### Academic Research
- **Literature Review**: Upload research papers and ask questions
- **Note Taking**: Chat with lecture notes and study materials
- **Citation Finding**: Locate specific information in documents

### Business Applications
- **Document Analysis**: Extract insights from reports and manuals
- **Knowledge Management**: Create searchable knowledge bases
- **Compliance**: Review and query policy documents

### Personal Use
- **Book Summaries**: Chat with e-books and articles
- **Recipe Search**: Find specific cooking instructions
- **Learning**: Interactive study with educational materials

## 🔮 Future Enhancements

### Planned Features
- **Multi-modal Support**: Images and audio processing
- **Advanced Chunking**: Semantic chunking strategies
- **Hybrid Search**: Combine keyword and semantic search
- **Export Functionality**: Save conversations and insights

### Performance Improvements
- **GPU Acceleration**: Leverage Apple Metal for embeddings
- **Streaming Responses**: Real-time answer generation
- **Batch Processing**: Process multiple documents simultaneously
- **Caching Layer**: Intelligent response caching

---

**Happy chatting with your documents! 🧠✨** 
