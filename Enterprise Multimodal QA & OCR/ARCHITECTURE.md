# 🏗️ RAG System Architecture

## Complete RAG Pipeline Architecture

This document explains the complete architecture of the RAG-powered OCR system.

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                      (Streamlit - app.py)                       │
└────────────────┬───────────────────────────────────────┬────────┘
                 │                                       │
                 ↓                                       ↓
┌────────────────────────────────┐    ┌─────────────────────────────┐
│     DOCUMENT PROCESSING        │    │     RAG QUERY SYSTEM        │
│                                │    │                             │
│  1. Image Upload               │    │  1. Question Input          │
│  2. OCR Extraction             │    │  2. Query Embedding         │
│  3. Text Chunking              │    │  3. Vector Search           │
│  4. Embedding Generation       │    │  4. Context Retrieval       │
│  5. Vector Storage             │    │  5. Answer Generation       │
└────────────────┬───────────────┘    └─────────────┬───────────────┘
                 │                                   │
                 ↓                                   ↓
┌────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                              │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │   ChromaDB   │  │  Processed   │  │   Config & Meta     │ │
│  │   Vectors    │  │  Documents   │  │   Settings          │ │
│  └──────────────┘  └──────────────┘  └─────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Document Processing Flow

### Step 1: Image Upload
```
User uploads image(s)
       ↓
utils/image_processing.py
       ↓
- Convert to PIL Image
- Optimize (resize, convert RGB)
- Convert to Base64
```

### Step 2: OCR Extraction
```
Base64 image
       ↓
OpenRouter API call
       ↓
Model: Grok 4.1 Fast / Gemini 2.0
       ↓
Extracted text
```

### Step 3: Document Creation
```
Extracted text
       ↓
rag/data_loader.py → Document object
       ↓
Metadata added:
  - source_file
  - timestamp
  - ocr_model
  - doc_id
```

### Step 4: Text Chunking
```
Document text
       ↓
rag/chunking.py → TextChunker
       ↓
Strategy selection:
  - Semantic (paragraphs)
  - Fixed Size (characters)
  - Sentence (sentences)
  - Recursive (hierarchical)
       ↓
List of TextChunk objects
```

### Step 5: Embedding Generation
```
Text chunks
       ↓
rag/embeddings.py → EmbeddingManager
       ↓
Provider selection:
  - HuggingFace (local)
  - OpenAI (cloud)
  - OpenRouter (cloud)
       ↓
Vector embeddings (float arrays)
```

### Step 6: Vector Storage
```
Embeddings + Metadata
       ↓
rag/vector_store.py → ChromaDB
       ↓
Stored in: data/chroma_db/
       ↓
Indexed for fast retrieval
```

---

## 💬 RAG Query Flow

### Step 1: Question Input
```
User types question
       ↓
"What is photosynthesis?"
       ↓
Passed to RAG retriever
```

### Step 2: Query Embedding
```
Question text
       ↓
rag/embeddings.py → embed_query()
       ↓
Query vector (same dimension as document embeddings)
```

### Step 3: Vector Search
```
Query vector
       ↓
rag/vector_store.py → similarity_search()
       ↓
ChromaDB cosine similarity
       ↓
Top-K most similar chunks
  (e.g., K=5)
```

### Step 4: Context Formatting
```
Retrieved chunks
       ↓
rag/retrieval.py → format_context()
       ↓
Formatted string:
  [Source: file1.png]
  Text chunk 1...
  
  ---
  
  [Source: file2.png]
  Text chunk 2...
```

### Step 5: Answer Generation
```
Context + Question
       ↓
Prompt template:
  CONTEXT: <formatted chunks>
  QUESTION: <user question>
  ANSWER:
       ↓
OpenRouter API call
       ↓
Model: Grok 4.1 Fast / Gemini 2.0
       ↓
Generated answer + sources
```

---

## 📦 Module Breakdown

### config.py
**Purpose:** Central configuration management

**Key Settings:**
- Model IDs (OCR, RAG)
- Chunking parameters (size, overlap, strategies)
- Embedding providers (HuggingFace, OpenAI, OpenRouter)
- Vector store settings (ChromaDB path, distance metric)
- API configurations (retry, timeout, temperature)

**Usage:**
```python
from config import DEFAULT_CHUNK_SIZE, OCR_MODELS
chunk_size = DEFAULT_CHUNK_SIZE
model = OCR_MODELS["grok-4.1-fast"]["id"]
```

---

### rag/data_loader.py
**Purpose:** Document lifecycle management

**Key Classes:**
- `Document`: Represents a document with text and metadata
- `DataLoader`: Manages document creation, storage, and retrieval

**Key Methods:**
```python
# Create document from OCR text
doc = loader.create_document(
    text="...",
    source_file="invoice.pdf",
    metadata={...}
)

# Save to disk
loader.save_document(doc)

# Load all documents
documents = loader.load_all_documents()

# Filter documents
invoices = loader.filter_documents(document_type="invoice")
```

---

### rag/chunking.py
**Purpose:** Split documents into retrieval-optimized chunks

**Key Classes:**
- `TextChunk`: Represents a single chunk with metadata
- `TextChunker`: Base class for chunking strategies
- `FixedSizeChunker`: Fixed character count
- `SemanticChunker`: Paragraph-based (recommended)
- `SentenceChunker`: Sentence-based grouping
- `RecursiveChunker`: Hierarchical splitting

**Chunking Example:**
```python
# Input: Long document text (5000 chars)
text = """
Photosynthesis is the process...
[4980 more characters]
"""

# Semantic chunking with overlap
chunks = chunk_document(
    text=text,
    strategy="semantic",
    chunk_size=1000,
    chunk_overlap=200
)

# Output: [TextChunk(0-1000), TextChunk(800-1800), ...]
```

**Why Chunking?**
- LLMs have context limits
- Smaller chunks = more precise retrieval
- Overlap maintains context between chunks
- Better matching for specific queries

---

### rag/embeddings.py
**Purpose:** Convert text to vector embeddings

**Key Classes:**
- `EmbeddingProvider`: Base class
- `HuggingFaceEmbeddings`: Local embeddings (free)
- `OpenAIEmbeddings`: Cloud embeddings (paid)
- `OpenRouterEmbeddings`: Cloud via OpenRouter (paid)
- `EmbeddingManager`: High-level interface

**How It Works:**
```python
# Initialize
embedder = EmbeddingManager(provider="huggingface")

# Single embedding
query_vec = embedder.embed_query("What is AI?")
# Output: [0.123, -0.456, 0.789, ..., 0.234]  (384 dims)

# Batch embeddings
chunk_vecs = embedder.embed_documents([
    "AI is artificial intelligence.",
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks."
])
# Output: [[...], [...], [...]]  (3 x 384 dims)
```

**Embedding Space:**
```
Similar meanings = nearby vectors

"AI" ───> [0.1, 0.5, 0.3]
                ↓ (close in vector space)
"Artificial Intelligence" ───> [0.12, 0.48, 0.31]

"Cat" ───> [0.8, -0.2, 0.1]  (far from AI concepts)
```

---

### rag/vector_store.py
**Purpose:** Persistent vector storage and retrieval

**Key Class:**
- `ChromaVectorStore`: ChromaDB operations

**Architecture:**
```
ChromaVectorStore
     ↓
ChromaDB Collection
     ↓
Vector Index (HNSW)
     ↓
Disk Storage (data/chroma_db/)
```

**Key Operations:**
```python
# Initialize
vector_store = ChromaVectorStore(
    collection_name="ocr_documents",
    embedding_manager=embedder
)

# Add documents
ids = vector_store.add_documents(
    texts=["doc1", "doc2"],
    metadatas=[{...}, {...}]
)

# Query (cosine similarity)
results = vector_store.query(
    query_text="What is AI?",
    n_results=5
)

# Results structure:
# {
#   "ids": ["id1", "id2", ...],
#   "documents": ["text1", "text2", ...],
#   "metadatas": [{...}, {...}, ...],
#   "distances": [0.23, 0.45, ...]  # Lower = more similar
# }
```

**ChromaDB Advantages:**
- Persistent storage
- Fast retrieval (HNSW index)
- Metadata filtering
- Local-first (privacy)
- No cloud dependencies

---

### rag/retrieval.py
**Purpose:** Complete RAG query orchestration

**Key Classes:**
- `RAGRetriever`: Main RAG pipeline
- `ConversationalRAG`: RAG with chat history

**Query Pipeline:**
```python
retriever = RAGRetriever(
    vector_store=vector_store,
    model_id="x-ai/grok-4.1-fast",
    temperature=0.5,
    top_k=5
)

# Complete RAG query
result = retriever.query(
    question="What is photosynthesis?",
    k=5,
    include_sources=True
)

# Result structure:
# {
#   "answer": "Photosynthesis is the process by which...",
#   "sources": [
#       {"file": "biology.png", "page": 3},
#       {"file": "science.png", "page": 12}
#   ],
#   "num_sources": 5
# }
```

**RAG Prompt Template:**
```
SYSTEM: You are an intelligent document assistant...

USER:
Based on the following context from processed documents, answer the question.

CONTEXT:
[Source: biology.png]
Photosynthesis is the process by which plants convert light energy...

[Source: science.png]
The chloroplasts in plant cells contain chlorophyll...

QUESTION:
What is photosynthesis?

ANSWER (based only on context above):
```

---

## 🔍 Similarity Search Deep Dive

### Vector Similarity
```
Query: "What is AI?"
Query Vector: q = [0.1, 0.5, 0.3, ...]

Document Chunks:
Chunk 1: "AI is artificial intelligence"
  Vector: d1 = [0.12, 0.48, 0.31, ...]
  Similarity: cosine(q, d1) = 0.98 ✓ High

Chunk 2: "Machine learning algorithms"
  Vector: d2 = [0.15, 0.52, 0.28, ...]
  Similarity: cosine(q, d2) = 0.91 ✓ Good

Chunk 3: "The cat sat on the mat"
  Vector: d3 = [0.8, -0.2, 0.1, ...]
  Similarity: cosine(q, d3) = 0.15 ✗ Low
```

### Cosine Similarity Formula
```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

Result: -1 to 1
  1.0  = identical
  0.5  = somewhat similar
  0.0  = orthogonal (unrelated)
  -1.0 = opposite
```

### Top-K Retrieval
```
K = 5 (retrieve 5 most similar)

All chunks sorted by similarity:
1. Chunk 42: 0.98 ✓
2. Chunk 15: 0.94 ✓
3. Chunk 33: 0.91 ✓
4. Chunk 8:  0.87 ✓
5. Chunk 19: 0.85 ✓
--- cut-off ---
6. Chunk 5:  0.76 ✗
7. Chunk 27: 0.71 ✗
...

Top 5 chunks sent to LLM as context
```

---

## 🎯 RAG vs Traditional Approaches

### Without RAG (Traditional)
```
User: "What is photosynthesis?"
     ↓
LLM (with general knowledge)
     ↓
Answer based on training data
     ↓
✗ May be outdated
✗ No specific document context
✗ Can't reference your documents
✗ Might hallucinate details
```

### With RAG (This System)
```
User: "What is photosynthesis?"
     ↓
Vector search in YOUR documents
     ↓
Retrieve relevant chunks from YOUR textbook
     ↓
LLM + Retrieved context
     ↓
Answer based on YOUR specific documents
     ↓
✓ Up-to-date (your content)
✓ Specific to your documents
✓ Cites actual sources
✓ Grounded in retrieved facts
```

---

## 🧪 Example: Complete Flow

### Setup
```python
# 1. Initialize components
embedder = EmbeddingManager(provider="huggingface")
vector_store = ChromaVectorStore(embedding_manager=embedder)
retriever = RAGRetriever(vector_store=vector_store)
```

### Document Processing
```python
# 2. Process document
text = extract_text_from_image("biology_page3.png")
# "Photosynthesis is the process by which green plants..."

# 3. Chunk
chunks = chunk_document(text, strategy="semantic", chunk_size=1000)
# [Chunk1(0-1000), Chunk2(800-1800), Chunk3(1600-2600)]

# 4. Embed
embeddings = embedder.embed_documents([c.text for c in chunks])
# [[0.1, 0.5, ...], [0.2, 0.4, ...], [0.15, 0.48, ...]]

# 5. Store
vector_store.add_documents(
    texts=[c.text for c in chunks],
    metadatas=[c.metadata for c in chunks],
    embeddings=embeddings
)
```

### Query Processing
```python
# 6. User query
question = "What is photosynthesis?"

# 7. Embed query
query_vec = embedder.embed_query(question)
# [0.12, 0.49, 0.31, ...]

# 8. Search
results = vector_store.query(query_vec, n_results=3)
# Top 3 most similar chunks from biology_page3.png

# 9. Format context
context = format_context(results["documents"], results["metadatas"])
# "[Source: biology_page3.png]
#  Photosynthesis is the process..."

# 10. Generate answer
answer = retriever.generate_response(question, context)
# "Based on the document, photosynthesis is the process by which
#  green plants convert light energy into chemical energy..."

# 11. Return with sources
return {
    "answer": answer,
    "sources": ["biology_page3.png"],
    "chunks_used": 3
}
```

---

## 📈 Performance Considerations

### Embedding Generation
- **HuggingFace (Local)**: ~0.5-2s per document (CPU)
- **OpenAI (Cloud)**: ~0.2-0.5s per document
- **Batch Processing**: Much faster than individual

### Vector Search
- **ChromaDB Query**: ~10-50ms for 1000 documents
- **Scales well**: Sub-linear with HNSW index
- **Metadata filtering**: Slight overhead but efficient

### LLM Generation
- **Grok 4.1 Fast**: ~2-5s response
- **Gemini 2.0 Flash**: ~1-3s response
- **Includes retry logic**: 3 attempts with backoff

### Bottlenecks
1. **OCR extraction**: Slowest step (5-15s per image)
2. **Embedding generation**: Moderate (batching helps)
3. **Vector search**: Fast (negligible)
4. **LLM generation**: Moderate (model dependent)

### Optimization Tips
- Process documents in batches
- Use local embeddings (avoid API calls)
- Cache embeddings (don't regenerate)
- Use faster LLM models
- Optimize chunk size (balance precision vs speed)

---

## 🔒 Security & Privacy

### Data Flow
```
Local:
  - Images uploaded
  - OCR text stored locally
  - Embeddings stored locally (ChromaDB)
  - All in: data/ directory

Cloud (API Calls):
  - OCR: Image sent to OpenRouter → Text returned
  - Embeddings (if using cloud): Text sent → Vectors returned
  - RAG Query: Context + question sent → Answer returned
```

### Privacy Levels
- **Maximum Privacy**: HuggingFace embeddings + local OCR (future feature)
- **Balanced**: HuggingFace embeddings + cloud OCR (current default)
- **Cloud-based**: All cloud providers (fastest but less private)

### Recommendations
- Use `.env` for API keys (never commit)
- Add `data/` to `.gitignore`
- Use HuggingFace embeddings for sensitive documents
- Review OpenRouter privacy policy for compliance

---

## 🚀 Future Enhancements

### Possible Extensions
1. **Multi-modal RAG**: Handle images directly in retrieval
2. **Hybrid Search**: Combine vector + keyword search
3. **Re-ranking**: Post-retrieval relevance re-scoring
4. **Query Expansion**: Augment queries with synonyms
5. **Conversational Memory**: Context across multiple questions
6. **Document Summarization**: Auto-generate summaries
7. **Entity Extraction**: Extract and index key entities
8. **Multi-language**: Support non-English documents

### Architecture Extensions
```
Current: Image → OCR → Chunk → Embed → Store → Query

Enhanced: Image → [OCR, Layout Detection, Entity Extraction]
              ↓
          Structured Document
              ↓
          [Chunking, Summarization, Metadata]
              ↓
          Multi-Index Storage (Vector + Graph + Keywords)
              ↓
          Hybrid Retrieval + Re-ranking
              ↓
          Enhanced Answer Generation
```

---

## 📚 References

- **RAG Paper**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/
- **OpenRouter API**: https://openrouter.ai/docs

---

**This architecture provides a complete, production-ready RAG system optimized for OCR-based document Q&A!** 🎉
