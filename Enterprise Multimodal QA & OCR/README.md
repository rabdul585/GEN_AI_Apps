# 🔍 RAG-Powered OCR & Document Q&A System

A **3-file modular** RAG (Retrieval-Augmented Generation) system for OCR processing and intelligent document Q&A.

## Demo link - https://youtu.be/ikHKeFaCyns

## 📁 Project Structure

```
3_file_clean_structure/
├── config.py              # Configuration & LLM Parameters
├── data_ingestion.py      # Data Loading, Chunking, Embedding, Storage
├── app.py                 # Streamlit UI (Main File)
├── requirements.txt       # Dependencies
├── .env.example           # API key template
└── README.md              # This file
```

## ✨ Features

### 1. **config.py** - Configuration & LLM Parameters
- ✅ Temperature settings (Precise, Balanced, Creative)
- ✅ Top-P (nucleus sampling) control
- ✅ Max tokens configuration
- ✅ Frequency & presence penalties
- ✅ Model selection (Grok, Gemini, Llama, DeepSeek)
- ✅ Chunking strategies
- ✅ Embedding providers
- ✅ RAG prompts

### 2. **data_ingestion.py** - Complete Data Pipeline
- ✅ **Document Management**: Load, store, manage documents
- ✅ **Text Chunking**: Semantic, fixed-size, sentence-based strategies
- ✅ **Embedding Generation**: HuggingFace (local) or OpenAI (cloud)
- ✅ **Vector Storage**: ChromaDB for persistent storage
- ✅ **OCR Processing**: Image to text extraction via OpenRouter
- ✅ **RAG Retrieval**: Semantic search and context retrieval
- ✅ **Answer Generation**: LLM-based Q&A with context

### 3. **app.py** - Streamlit UI
- ✅ **Upload & Process**: Multi-image OCR processing
- ✅ **Ask Questions**: Intelligent Q&A with RAG
- ✅ **Document Library**: View and manage documents
- ✅ **Analytics**: Usage statistics and insights

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- OpenRouter API key (free from [openrouter.ai](https://openrouter.ai))

### 2. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create `.env` file:
```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key:
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Get your free API key at: https://openrouter.ai/keys

### 4. Run

```bash
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## 📖 Usage Guide

### Step 1: Initialize System

1. Open sidebar
2. Configure settings:
   - **Models**: Select OCR & RAG models (Grok 4.1 Fast recommended)
   - **Chunking**: Choose strategy (Semantic recommended)
   - **Embeddings**: Select provider (HuggingFace for free local)
   - **Generation**: Choose temperature preset
3. Click **"Initialize System"**

### Step 2: Upload Documents

1. Go to **"Upload"** tab
2. Upload images (PNG, JPG, etc.)
3. Click **"Process Documents"**
4. Wait for OCR + embedding generation

### Step 3: Ask Questions

1. Go to **"Query"** tab
2. Type your question
3. Optionally filter by source file
4. Click **"Ask"**
5. View answer with sources

### Step 4: Manage Documents

1. Go to **"Library"** tab
2. View processed documents
3. Check metadata and statistics
4. Clear database if needed

---

## ⚙️ Configuration Options

### LLM Parameters (in `config.py`)

#### Temperature Presets
```python
TEMPERATURE_PRESETS = {
    "Precise (0.1)": {
        "temp": 0.1,    # Most accurate, deterministic
        "top_p": 0.1,
        "desc": "Most accurate, deterministic"
    },
    "Balanced (0.5)": {
        "temp": 0.5,    # Balanced (default)
        "top_p": 0.9,
        "desc": "Balanced accuracy & fluency"
    },
    "Creative (0.9)": {
        "temp": 0.9,    # More creative
        "top_p": 0.95,
        "desc": "More creative & varied"
    }
}
```

#### Custom Parameters
```python
DEFAULT_TEMPERATURE = 0.5        # 0.0-2.0 (higher = more random)
DEFAULT_TOP_P = 0.9              # 0.0-1.0 (nucleus sampling)
DEFAULT_TOP_K = 40               # Top-k sampling
DEFAULT_MAX_TOKENS = 2000        # Maximum response length
FREQUENCY_PENALTY = 0.0          # -2.0 to 2.0 (reduce repetition)
PRESENCE_PENALTY = 0.0           # -2.0 to 2.0 (encourage new topics)
```

### Retrieval Parameters
```python
DEFAULT_RETRIEVAL_K = 5          # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3       # Minimum similarity (0-1)
```

### Chunking Parameters
```python
DEFAULT_CHUNK_SIZE = 1000        # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 200      # Overlap between chunks
MIN_CHUNK_SIZE = 100             # Minimum chunk size
MAX_CHUNK_SIZE = 4000            # Maximum chunk size
```

---

## 🎯 How It Works

### RAG Pipeline Flow

```
1. UPLOAD IMAGE
   ↓
2. OCR EXTRACTION (via OpenRouter)
   ↓
3. TEXT CHUNKING (semantic/fixed/sentence)
   ↓
4. EMBEDDING GENERATION (HuggingFace/OpenAI)
   ↓
5. VECTOR STORAGE (ChromaDB)
   ↓
6. USER QUERY
   ↓
7. SEMANTIC SEARCH (retrieve top-K similar chunks)
   ↓
8. CONTEXT FORMATTING
   ↓
9. LLM GENERATION (with retrieved context)
   ↓
10. ANSWER + SOURCES
```

### Temperature Explained

**Temperature** controls randomness in LLM responses:
- **0.0-0.3**: Deterministic, focused, factual (best for Q&A)
- **0.4-0.7**: Balanced creativity and accuracy
- **0.8-1.0**: Creative, diverse, less predictable
- **1.0+**: Very random (rarely used)

### Top-P Explained

**Top-P** (nucleus sampling) filters token probabilities:
- **0.1**: Very focused (only top 10% likely tokens)
- **0.5**: Balanced
- **0.9**: More diverse (default)
- **0.95**: Very diverse

### Temperature vs Top-P

Use together for best results:
- **Factual Q&A**: temp=0.1, top_p=0.1
- **General use**: temp=0.5, top_p=0.9
- **Creative writing**: temp=0.9, top_p=0.95

---

## 🔧 Customization

### Change Models

Edit `config.py`:
```python
AVAILABLE_MODELS = {
    "Grok 4.1 Fast": "x-ai/grok-4.1-fast",           # Fast & accurate
    "Gemini 2.0 Flash": "google/gemini-2.0-flash-exp:free",  # Large context
    "Your Model": "provider/model-name"               # Add your own
}
```

### Change Prompts

Edit in `config.py`:
```python
OCR_PROMPT = """Your custom OCR instructions"""

RAG_SYSTEM_PROMPT = """Your custom RAG system instructions"""

RAG_QUERY_TEMPLATE = """Your custom query template with {context} and {question}"""
```

### Change Chunking

In UI or edit `config.py`:
```python
DEFAULT_CHUNKING_STRATEGY = "Semantic"  # or "Fixed Size", "Sentence-based"
DEFAULT_CHUNK_SIZE = 1000               # Adjust as needed
```

---

## 🐛 Troubleshooting

### Issue: Rate Limit (429 Error)
**Solutions:**
1. Switch to different model
2. Wait 2-5 minutes
3. Process fewer documents at once
4. Check OpenRouter usage limits

### Issue: Slow Embeddings
**Solutions:**
1. First run downloads model (one-time ~100MB)
2. Use smaller EMBEDDING_BATCH_SIZE
3. Switch to OpenAI embeddings (faster, not free)
4. Install PyTorch with GPU support

### Issue: No Results from Query
**Solutions:**
1. Check documents were processed (Library tab)
2. Increase retrieval K (try 10-15)
3. Lower SIMILARITY_THRESHOLD
4. Rephrase question
5. Ensure question relates to documents

### Issue: Memory Errors
**Solutions:**
1. Reduce chunk_size
2. Reduce EMBEDDING_BATCH_SIZE
3. Process fewer documents at once
4. Clear old documents from library

### Issue: Poor Answer Quality
**Solutions:**
1. Adjust temperature (lower = more factual)
2. Increase retrieval K (more context)
3. Try different chunking strategy
4. Use better quality images for OCR
5. Switch to better model (Grok or Gemini)

---

## 📊 Performance Tips

### For Best Results
- **OCR**: Use high-resolution images (300+ DPI)
- **Chunking**: Semantic strategy for most documents
- **Chunk Size**: 1000-1500 chars for general documents
- **Retrieval K**: 5-10 for comprehensive answers
- **Temperature**: 0.1-0.3 for factual Q&A, 0.5-0.7 for general use

### For Speed
- Use HuggingFace embeddings (local, no API calls)
- Use Grok 4.1 Fast model (fastest)
- Process documents in batches of 5-10
- Enable GPU acceleration if available

### For Privacy
- Use HuggingFace embeddings (fully local)
- All documents stored locally in `data/chroma_db/`
- Only API calls send data to OpenRouter

---

## 📝 Example Use Cases

### 1. Student Study Assistant
- Upload textbook pages
- Ask: "Explain photosynthesis", "What are Newton's laws?"
- Get instant answers with sources

### 2. Invoice Processing
- Upload invoice images
- Ask: "What is the total amount?", "Who is the vendor?"
- Extract specific information

### 3. Research Papers
- Upload PDF screenshots
- Ask: "What are the main findings?", "Compare methodologies"
- Analyze and compare papers

### 4. Contract Review
- Upload contract pages
- Ask: "What are the payment terms?", "Find termination clauses"
- Extract key legal information

### 5. Technical Documentation
- Upload manual pages
- Ask: "How to configure X?", "What are requirements?"
- Quick reference lookup

---

## 🔒 Security & Privacy

- ✅ API keys stored in `.env` (not in code)
- ✅ Documents stored locally in `data/` directory
- ✅ ChromaDB runs locally (no cloud upload)
- ✅ HuggingFace embeddings are fully local
- ⚠️ OpenRouter API calls send data to their servers

---

## 📚 Technology Stack

- **Streamlit**: Web UI framework
- **ChromaDB**: Vector database
- **Sentence Transformers**: Local embeddings
- **OpenRouter**: LLM API (Grok, Gemini, etc.)
- **PIL**: Image processing
- **Python 3.8+**: Core language

---

## 📄 File Descriptions

### 1. **config.py** (Configuration)
Contains all settings:
- LLM parameters (temperature, top_p, tokens)
- Model definitions
- Chunking strategies
- Embedding providers
- Prompts and templates
- Helper functions

### 2. **data_ingestion.py** (Data Pipeline)
Complete data operations:
- `Document`: Document data class
- `TextChunker`: Text chunking with 3 strategies
- `EmbeddingManager`: Generate embeddings
- `VectorStore`: ChromaDB operations
- `RAGPipeline`: Retrieval and generation
- `DataIngestionPipeline`: End-to-end pipeline
- OCR functions
- Utility functions

### 3. **app.py** (UI)
Streamlit interface:
- Sidebar configuration
- 4 tabs (Upload, Query, Library, Analytics)
- Session state management
- User interactions
- Progress indicators
- Error handling

---

## 🎓 Learn More

### Understanding RAG
RAG (Retrieval-Augmented Generation) combines:
1. **Retrieval**: Find relevant information from documents
2. **Generation**: Use LLM to generate answer with context

Benefits:
- More accurate than pure LLM
- Grounded in your documents
- Can cite sources
- Works with proprietary data

### Understanding Embeddings
Embeddings convert text to vectors:
- Similar text → Similar vectors
- Enables semantic search
- Used for retrieval in RAG

### Understanding ChromaDB
Vector database for embeddings:
- Stores embeddings persistently
- Fast similarity search (HNSW index)
- Metadata filtering
- Local storage

---

## 🚀 Next Steps

1. ✅ Initialize system
2. ✅ Upload test document
3. ✅ Process it
4. ✅ Ask questions
5. ✅ Experiment with parameters
6. ✅ Add more documents
7. ✅ Build your knowledge base!

---

## 💡 Tips

- Start with default settings
- Adjust temperature based on use case
- Use semantic chunking for most documents
- Increase retrieval K for complex questions
- Lower temperature for factual accuracy
- Use HuggingFace for privacy (local embeddings)

---

## 📞 Support

For issues:
1. Check this README
2. Review error messages
3. Check `config.py` settings
4. Verify API key is set
5. Check OpenRouter API status

---

**Happy RAG-ing! 🚀**
