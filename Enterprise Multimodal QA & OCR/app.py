"""
RAG-Powered OCR & Document Q&A Application
Streamlit UI - Main Application File
"""

import streamlit as st
from config import *
from data_ingestion import DataIngestionPipeline, format_timestamp, truncate_text

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT
)

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DataIngestionPipeline()
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

init_session_state()

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

st.sidebar.title("⚙️ Configuration")

# API Key Status
st.sidebar.header("🔑 API Key")
if OPENROUTER_API_KEY:
    st.sidebar.success("✅ API Key Loaded")
    st.sidebar.code(f"{OPENROUTER_API_KEY[:8]}...{OPENROUTER_API_KEY[-4:]}")
else:
    st.sidebar.error("❌ API Key Missing")
    st.sidebar.warning("Add OPENROUTER_API_KEY to .env")
    st.stop()

# Model Selection
st.sidebar.header("🤖 Models")

ocr_model_name = st.sidebar.selectbox(
    "OCR Model:",
    list(AVAILABLE_MODELS.keys()),
    index=0
)
ocr_model_id = AVAILABLE_MODELS[ocr_model_name]

rag_model_name = st.sidebar.selectbox(
    "RAG Model:",
    list(AVAILABLE_MODELS.keys()),
    index=0
)
rag_model_id = AVAILABLE_MODELS[rag_model_name]

# Chunking
st.sidebar.header("📝 Chunking")

chunking_strategy = st.sidebar.selectbox(
    "Strategy:",
    CHUNKING_STRATEGIES,
    index=0
)

chunk_size = st.sidebar.slider(
    "Chunk Size (chars):",
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    DEFAULT_CHUNK_SIZE,
    step=100
)

chunk_overlap = st.sidebar.slider(
    "Overlap (chars):",
    0,
    chunk_size // 2,
    DEFAULT_CHUNK_OVERLAP,
    step=50
)

# Embeddings
st.sidebar.header("🔢 Embeddings")

embedding_provider = st.sidebar.selectbox(
    "Provider:",
    list(EMBEDDING_PROVIDERS.keys()),
    index=0
)

# Generation Parameters
st.sidebar.header("🎯 Generation Parameters")

# Temperature Preset
temp_preset = st.sidebar.selectbox(
    "Preset:",
    list(TEMPERATURE_PRESETS.keys()),
    index=1
)

# Show preset description
st.sidebar.info(f"📊 {TEMPERATURE_PRESETS[temp_preset]['desc']}")

# Advanced Parameters
with st.sidebar.expander("⚙️ Advanced"):
    use_custom = st.checkbox("Use Custom Parameters")
    
    if use_custom:
        temperature = st.slider(
            "Temperature:",
            0.0, 2.0,
            TEMPERATURE_PRESETS[temp_preset]['temp'],
            step=0.1
        )
        top_p = st.slider(
            "Top-P:",
            0.0, 1.0,
            TEMPERATURE_PRESETS[temp_preset]['top_p'],
            step=0.05
        )
    else:
        temperature = TEMPERATURE_PRESETS[temp_preset]['temp']
        top_p = TEMPERATURE_PRESETS[temp_preset]['top_p']
    
    max_tokens = st.slider(
        "Max Tokens:",
        MIN_TOKENS,
        MAX_TOKENS_LIMIT,
        DEFAULT_MAX_TOKENS,
        step=100
    )
    
    retrieval_k = st.slider(
        "Top-K Retrieval:",
        MIN_RETRIEVAL_K,
        MAX_RETRIEVAL_K,
        DEFAULT_RETRIEVAL_K
    )

# Initialize System
st.sidebar.header("🚀 System")

if st.sidebar.button("Initialize System", type="primary", disabled=st.session_state.system_initialized):
    with st.spinner("Initializing..."):
        try:
            st.session_state.pipeline.initialize(
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_provider=EMBEDDING_PROVIDERS[embedding_provider]["provider"],
                rag_model=rag_model_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                retrieval_k=retrieval_k
            )
            st.session_state.system_initialized = True
            st.sidebar.success("✅ Initialized!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

# System Status
if st.session_state.system_initialized:
    st.sidebar.success("✅ Ready")
    if st.session_state.pipeline.vector_store:
        count = st.session_state.pipeline.vector_store.count()
        st.sidebar.metric("Documents", count)
else:
    st.sidebar.info("⏸️ Not Initialized")

# ============================================================================
# MAIN APP
# ============================================================================

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown("**Upload → Process → Ask Questions**")

# Current Config
with st.expander("ℹ️ Current Configuration"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**OCR:** {ocr_model_name}")
        st.markdown(f"**RAG:** {rag_model_name}")
    with col2:
        st.markdown(f"**Chunking:** {chunking_strategy}")
        st.markdown(f"**Size:** {chunk_size} chars")
    with col3:
        st.markdown(f"**Temperature:** {temperature}")
        st.markdown(f"**Top-P:** {top_p}")
    with col4:
        st.markdown(f"**Max Tokens:** {max_tokens}")
        st.markdown(f"**Retrieval K:** {retrieval_k}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "💬 Query", "📚 Library", "📊 Analytics"])

# ============================================================================
# TAB 1: UPLOAD & PROCESS
# ============================================================================

with tab1:
    st.header("📤 Upload & Process Documents")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Initialize system first (sidebar)")
    else:
        uploaded_files = st.file_uploader(
            "Upload Documents (Images, PDFs, Word):",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Supported: PNG, JPG, GIF, WebP, PDF, DOCX"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Preview - only show images
            for f in uploaded_files[:3]:
                ext = Path(f.name).suffix.lower()
                with st.expander(f"📄 {f.name}"):
                    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                        st.image(f, use_column_width=True)
                    else:
                        st.info(f"📄 {ext.upper()} document - {f.size/1024:.1f} KB")
            
            # Process Button
            if st.button("🔄 Process Documents", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    progress.progress((idx + 1) / len(uploaded_files))
                    
                    file.seek(0)
                    
                    try:
                        result = st.session_state.pipeline.process_document(
                            file, ocr_model_id, chunking_strategy
                        )
                        
                        if result["success"]:
                            st.session_state.documents_processed.append({
                                "filename": file.name,
                                "doc_id": result["doc_id"],
                                "chunks": result["chunks"],
                                "text_length": result["text_length"],
                                "timestamp": format_timestamp()
                            })
                            st.success(f"✅ {file.name}")
                        else:
                            st.error(f"❌ {file.name}: {result.get('error')}")
                    except Exception as e:
                        st.error(f"❌ {file.name}: {str(e)}")
                
                status.text("✅ Complete!")
                st.balloons()

# ============================================================================
# TAB 2: QUERY
# ============================================================================

with tab2:
    st.header("💬 Ask Questions")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Initialize system first")
    elif not st.session_state.documents_processed:
        st.info("ℹ️ Process documents first")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question = st.text_area(
                "Question:",
                height=100,
                placeholder="What information is in the documents?"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                ask = st.button("🔍 Ask", type="primary")
            with col_b:
                if st.button("🗑️ Clear History"):
                    st.session_state.query_history = []
                    st.rerun()
        
        with col2:
            st.subheader("Options")
            show_sources = st.checkbox("Show sources", value=True)
            show_context = st.checkbox("Show context", value=False)
            
            filter_source = st.selectbox(
                "Filter by:",
                ["All"] + [d["filename"] for d in st.session_state.documents_processed]
            )
        
        # Ask
        if ask and question:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.pipeline.query_documents(
                        question=question,
                        k=retrieval_k,
                        filter_source=filter_source if filter_source != "All" else None
                    )
                    
                    # Answer
                    st.markdown("### 📝 Answer")
                    st.markdown(result["answer"])
                    
                    # Sources
                    if show_sources and result.get("sources"):
                        st.markdown("### 📚 Sources")
                        for s in result["sources"]:
                            st.markdown(f"- **{s['file']}**")
                    
                    # Context
                    if show_context and result.get("context"):
                        with st.expander(f"📄 Context ({len(result['context'])} chunks)"):
                            for i, ctx in enumerate(result["context"], 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.text(truncate_text(ctx, 300))
                                st.markdown("---")
                    
                    # Save to history
                    st.session_state.query_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "timestamp": format_timestamp()
                    })
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # History
        if st.session_state.query_history:
            st.markdown("---")
            st.subheader("📜 History")
            
            for i, item in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"Q{len(st.session_state.query_history)-i}: {truncate_text(item['question'], 80)}"):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer']}")
                    st.caption(f"🕒 {item['timestamp']}")

# ============================================================================
# TAB 3: LIBRARY
# ============================================================================

with tab3:
    st.header("📚 Document Library")
    
    if st.session_state.system_initialized and st.session_state.pipeline.vector_store:
        count = st.session_state.pipeline.vector_store.count()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", count)
        with col2:
            st.metric("Processed", len(st.session_state.documents_processed))
        
        # Processed docs
        if st.session_state.documents_processed:
            st.subheader("Processed Documents")
            
            for doc in st.session_state.documents_processed:
                with st.expander(f"📄 {doc['filename']}"):
                    st.markdown(f"**Doc ID:** `{doc['doc_id']}`")
                    st.markdown(f"**Chunks:** {doc['chunks']}")
                    st.markdown(f"**Length:** {doc['text_length']:,} chars")
                    st.markdown(f"**Time:** {doc['timestamp']}")
        
        # Operations
        st.subheader("🔧 Operations")
        
        if st.button("🗑️ Clear All"):
            st.session_state.pipeline.vector_store.clear()
            st.session_state.documents_processed = []
            st.success("✅ Cleared")
            st.rerun()
    else:
        st.info("Initialize system to view library")

# ============================================================================
# TAB 4: ANALYTICS
# ============================================================================

with tab4:
    st.header("📊 Analytics")
    
    if st.session_state.documents_processed:
        total_chunks = sum(d["chunks"] for d in st.session_state.documents_processed)
        total_chars = sum(d["text_length"] for d in st.session_state.documents_processed)
        avg_chunks = total_chunks / len(st.session_state.documents_processed)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", len(st.session_state.documents_processed))
        with col2:
            st.metric("Chunks", total_chunks)
        with col3:
            st.metric("Avg Chunks", f"{avg_chunks:.1f}")
        with col4:
            st.metric("Total Chars", f"{total_chars:,}")
        
        # Recent Activity
        st.subheader("📅 Recent Activity")
        recent = list(reversed(st.session_state.documents_processed[-5:]))
        for doc in recent:
            st.markdown(f"**{doc['filename']}** - {doc['timestamp']}")
            st.progress(min(doc['chunks'] / 20, 1.0))
        
        # Queries
        if st.session_state.query_history:
            st.subheader("💬 Query Stats")
            st.metric("Total Queries", len(st.session_state.query_history))
            
            st.markdown("**Recent:**")
            for q in st.session_state.query_history[-5:]:
                st.markdown(f"- {truncate_text(q['question'], 60)} ({q['timestamp']})")
    else:
        st.info("Process documents to see analytics")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        <p><strong>{APP_TITLE}</strong></p>
        <p>🔍 Extract → 📝 Chunk → 🔢 Embed → 💾 Store → 💬 Query</p>
        <p><small>Temp: {temperature} | Top-P: {top_p} | Max Tokens: {max_tokens}</small></p>
    </div>
    """,
    unsafe_allow_html=True
)