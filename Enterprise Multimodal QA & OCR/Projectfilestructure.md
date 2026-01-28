# 📁 3-File Architecture Overview

## 🎯 Design Philosophy

This RAG system uses a **clean 3-file modular design**:

1. **config.py** - All settings and parameters
2. **data_ingestion.py** - Complete data pipeline
3. **app.py** - Streamlit UI

This separation provides:
- ✅ Clear organization
- ✅ Easy customization
- ✅ Simple maintenance
- ✅ Minimal dependencies

---

## 📄 File Breakdown

### 1. config.py (Configuration & Settings)

**Purpose**: Central configuration for all parameters

**Contains**:
- **LLM Parameters**:
  - Temperature presets (Precise, Balanced, Creative)
  - Top-P, Top-K, Max tokens
  - Frequency & presence penalties
- **Model Selection**:
  - Available models (Grok, Gemini, Llama, DeepSeek)
  - Default models for OCR and RAG
- **RAG Configuration**:
  - Retrieval settings (top-k, similarity threshold)
  - Chunking strategies & parameters
  - Embedding providers & models
- **Prompts**:
  - OCR extraction prompt
  - RAG system prompt
  - Query template
- **System Settings**:
  - API URLs, retry logic, timeouts
  - Paths, directories
  - UI configuration

**Key Functions**:
```python
get_generation_params()  # Get LLM parameters
validate_config()        # Validate settings
```

**Lines of Code**: ~250

---

### 2. data_ingestion.py (Data Pipeline)

**Purpose**: Complete data processing pipeline

**Contains**:

**Classes**:
1. **Document**: Document data structure
2. **TextChunker**: Split text into chunks
   - Semantic chunking (by paragraphs)
   - Fixed-size chunking (with overlap)
   - Sentence-based chunking
3. **EmbeddingManager**: Generate embeddings
   - HuggingFace (local, free)
   - OpenAI (cloud, paid)
4. **VectorStore**: ChromaDB operations
   - Add documents
   - Query similar documents
   - Manage collections
5. **RAGPipeline**: Retrieval + Generation
   - Retrieve context
   - Format context
   - Generate answer
6. **DataIngestionPipeline**: End-to-end pipeline
   - Initialize all components
   - Process documents
   - Query documents

**Functions**:
```python
image_to_base64()           # Convert images
extract_text_from_image()   # OCR via API
format_timestamp()          # Utilities
truncate_text()            # Utilities
```

**Flow**:
```
Image → OCR → Document → Chunks → Embeddings → Vector Store
Query → Retrieve → Format → Generate → Answer
```

**Lines of Code**: ~500

---

### 3. app.py (Streamlit UI)

**Purpose**: User interface and interaction

**Contains**:

**UI Components**:
1. **Sidebar**:
   - API key status
   - Model selection (OCR & RAG)
   - Chunking configuration
   - Embedding provider
   - Generation parameters (with presets)
   - Advanced parameters (collapsible)
   - Initialize system button
   - System status

2. **4 Tabs**:
   - **Upload**: Upload and process images
   - **Query**: Ask questions, view answers
   - **Library**: Manage documents, view stats
   - **Analytics**: Usage statistics, history

**Session State**:
- `pipeline`: DataIngestionPipeline instance
- `documents_processed`: List of processed docs
- `query_history`: Q&A history
- `system_initialized`: Initialization status

**User Flows**:
1. Initialize → Upload → Process → Query
2. View Library → Check Analytics
3. Clear Documents → Reset

**Lines of Code**: ~350

---

## 🔄 Data Flow

### Initialization Flow
```
app.py (UI)
  ↓ user clicks "Initialize"
config.py (settings)
  ↓ provides parameters
data_ingestion.py
  ↓ initializes
TextChunker + EmbeddingManager + VectorStore + RAGPipeline
  ↓ ready
app.py (UI enabled)
```

### Document Processing Flow
```
app.py (upload image)
  ↓
data_ingestion.py::DataIngestionPipeline.process_document()
  ↓
image_to_base64() → OCR API → extract text
  ↓
TextChunker.chunk() → split into chunks
  ↓
EmbeddingManager.embed_documents() → generate embeddings
  ↓
VectorStore.add_documents() → store in ChromaDB
  ↓
app.py (update UI)
```

### Query Flow
```
app.py (user question)
  ↓
data_ingestion.py::DataIngestionPipeline.query_documents()
  ↓
RAGPipeline.query()
  ↓
VectorStore.query() → retrieve similar chunks
  ↓
RAGPipeline.generate_response() → LLM with context
  ↓
app.py (display answer + sources)
```

---

## 🎨 Design Decisions

### Why 3 Files?

**config.py separate because**:
- Easy to adjust parameters
- No code changes needed
- Single source of truth
- Can be shared across projects

**data_ingestion.py separate because**:
- Complex pipeline logic
- Can be used independently
- Can be tested separately
- Can be imported by other apps

**app.py separate because**:
- UI can change without affecting logic
- Easy to customize interface
- Clear separation of concerns
- Can switch to different UI framework

### Why Not More Files?

Could split into more files, but:
- ❌ More complexity
- ❌ More imports
- ❌ Harder to navigate
- ❌ Overkill for this size

3 files is the sweet spot:
- ✅ Clear organization
- ✅ Easy to find things
- ✅ Simple structure
- ✅ Minimal overhead

---

## 🔧 Customization Points

### Change LLM Parameters
**File**: `config.py`
**Lines**: 20-50
**What**: Temperature, top_p, max_tokens, etc.

### Change Models
**File**: `config.py`
**Lines**: 60-80
**What**: Add/remove/modify models

### Change Chunking
**File**: `config.py`
**Lines**: 90-110
**What**: Strategies, sizes, overlaps

### Change Prompts
**File**: `config.py`
**Lines**: 140-170
**What**: OCR prompt, RAG system prompt, query template

### Add Chunking Strategy
**File**: `data_ingestion.py`
**Class**: `TextChunker`
**Method**: Add new method like `chunk_by_custom()`

### Add Embedding Provider
**File**: `data_ingestion.py`
**Class**: `EmbeddingManager`
**Method**: Modify `__init__()` and `embed_documents()`

### Change UI Layout
**File**: `app.py`
**What**: Modify tabs, sidebar, or add new sections

---

## 📊 File Sizes

- **config.py**: ~250 lines (~10 KB)
- **data_ingestion.py**: ~500 lines (~20 KB)
- **app.py**: ~350 lines (~15 KB)
- **Total Code**: ~1100 lines (~45 KB)

Plus:
- **README.md**: ~500 lines
- **QUICKSTART.md**: ~150 lines
- **requirements.txt**: ~15 lines
- **.env.example**: ~5 lines

---

## 🎯 When to Add More Files?

Add more files if:
- Individual files exceed 500 lines
- You need multiple UIs (web + CLI + API)
- You have many embedding providers
- You add complex pre/post-processing
- You need extensive testing infrastructure

For this project: **3 files is perfect!**

---

## 🚀 Getting Started

1. Read **QUICKSTART.md** (5 minutes)
2. Run the app
3. Experiment with settings in sidebar
4. Modify `config.py` for custom parameters
5. Extend `data_ingestion.py` for new features
6. Customize `app.py` for different UI

---

## 📚 Dependencies

**Direct Dependencies** (in requirements.txt):
- streamlit (UI framework)
- chromadb (vector database)
- sentence-transformers (embeddings)
- Pillow (image processing)
- requests (API calls)
- python-dotenv (config)
- numpy (data processing)
- torch (ML backend)
- transformers (ML models)

**Indirect Dependencies**:
- Everything required by the above

**Total Install Size**: ~2-3 GB (mostly PyTorch)

---

## 💡 Pro Tips

### For Development
- Modify `config.py` first for quick changes
- Test pipeline in `data_ingestion.py` independently
- Use `app.py` only for UI changes

### For Production
- Lock dependency versions in requirements.txt
- Add error logging to all 3 files
- Add unit tests for data_ingestion.py
- Add input validation in app.py

### For Extensions
- Add new features to data_ingestion.py
- Add new settings to config.py
- Add new UI elements to app.py
- Keep separation clean!

---

**Simple. Clean. Modular. Effective. 🎯**