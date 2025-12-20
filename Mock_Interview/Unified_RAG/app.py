import streamlit as st
import os
# Fix for HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Abdul bot - PDF Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .source-doc {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #ff9800;
    }
    
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        transition: all 0.3s;
    }
    
    .hero-box {
        background: linear-gradient(135deg, #eef2ff 0%, #f3e8ff 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e7ff;
    }
    
    .pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background-color: #e0e7ff;
        font-size: 0.8rem;
        font-weight: 500;
        color: #3730a3;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0

def process_pdf(uploaded_file):
    """Process uploaded PDF and create vector store"""
    try:
        with st.spinner("📄 Processing PDF..."):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            st.info(f"✅ Loaded {len(documents)} pages from PDF")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            st.info(f"✅ Split into {len(chunks)} text chunks")
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Create vector store
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            st.session_state.vectorstore = vectorstore
            st.session_state.chunk_count = len(chunks)
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_processed = True
            
            # Create conversation chain
            setup_conversation_chain()
            
            st.success("🎉 PDF processed successfully! You can now ask questions.")
            
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None

def setup_conversation_chain():
    """Setup conversational retrieval chain using modern LCEL"""
    try:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Answer question prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        st.session_state.conversation_chain = rag_chain
        
    except Exception as e:
        st.error(f"❌ Error setting up conversation: {str(e)}")
        st.info("💡 Make sure your OPENAI_API_KEY is set in the .env file")

def get_response(question):
    """Get response from the RAG system"""
    try:
        if st.session_state.conversation_chain is None:
            st.error("❌ Please upload and process a PDF first!")
            return None
        
        # Convert chat history to LangChain message format
        chat_history = []
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                chat_history.append(HumanMessage(content=msg['content']))
            else:
                chat_history.append(AIMessage(content=msg['content']))
        
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.conversation_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })
            
            # Return response in expected format
            return {
                "answer": response["answer"],
                "source_documents": response.get("context", [])
            }
            
    except Exception as e:
        st.error(f"❌ Error getting response: {str(e)}")
        return None

def display_chat_message(role, content, sources=None):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            st.markdown("**📚 Sources:**")
            for i, source in enumerate(sources, 1):
                page = source.metadata.get('page', 'Unknown')
                snippet = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                
                with st.expander(f"📄 Source {i} (Page {page + 1})"):
                    st.write(snippet)
                    st.caption(f"Page {page + 1} of document")

# ----------------- PAGE RENDER FUNCTIONS ----------------- #

def render_home_page():
    st.markdown('<h1 class="main-header">🤖 Abdul's RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered Q&A over your PDFs, with sources.")
    
    st.markdown("""
    <div class="hero-box">
        <div class="pill">Document AI • RAG • Chatbot</div>
        <h2>Chat with your PDFs like a human expert is sitting next to you.</h2>
        <p>
            Upload any text-based PDF, and Abdul bot will build a mini 
            knowledge base on top of it. Ask natural language questions and get
            concise answers backed by citations from the original document.
        </p>
        <ul>
            <li>✅ Supports multi-page PDFs</li>
            <li>✅ Shows source snippets for transparency</li>
            <li>✅ Remembers your chat history in the current session</li>
        </ul>
        <p><b>Get started:</b> upload a PDF from the sidebar and then open the <b>💬 Chat</b> page.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🧠 RAG Powered")
        st.write("Uses Retrieval-Augmented Generation so answers are grounded in your document, not hallucinated.")
    with col2:
        st.markdown("### 📚 Source Aware")
        st.write("Every answer comes with traceable source chunks, so you can verify the information.")
    with col3:
        st.markdown("### ⚡ Fast Setup")
        st.write("Just upload a PDF and start chatting. No extra configuration needed.")


def render_chat_page():
    st.markdown('<h1 class="main-header">💬 Chat with your PDF</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.pdf_processed:
        st.subheader("Ask Questions About Your Document")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(
                message['role'],
                message['content'],
                message.get('sources')
            )
        
        # Chat input
        with st.form(key='chat_form', clear_on_submit=True):
            user_question = st.text_input(
                "Your question:",
                placeholder="e.g., What is the main topic of this document?",
                key='user_input'
            )
            submit_button = st.form_submit_button("Send 📤")
        
        if submit_button and user_question.strip():
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Get response
            response = get_response(user_question)
            
            if response:
                # Add bot response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'sources': response.get('source_documents', [])
                })
            
            st.rerun()
        
    else:
        # Welcome screen specific to Chat page
        st.info("👋 First, upload and process a PDF from the sidebar to start chatting.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📄 Step 1: Upload
            Use the sidebar to upload your PDF document. 
            Supports any text-based PDF file.
            """)
        
        with col2:
            st.markdown("""
            ### ⚙️ Step 2: Process
            Click 'Process PDF' in the sidebar to analyze the document. 
            This creates a searchable knowledge base.
            """)
        
        with col3:
            st.markdown("""
            ### 💬 Step 3: Chat
            Ask questions about your document and get 
            intelligent answers with source citations.
            """)

        st.markdown("---")
        st.subheader("💡 Example Questions You Can Ask:")
        examples = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the conclusions or recommendations?",
            "Explain the methodology described in the document",
            "What data or statistics are mentioned?",
            "Who are the authors and what are their affiliations?"
        ]
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                st.markdown(f"• {example}")


def render_how_it_works_page():
    st.markdown('<h1 class="main-header">⚙️ How It Works</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### End-to-end Pipeline")
    
    st.markdown("""
1. **📄 Upload PDF**  
   You upload a text-based PDF using the sidebar. The app reads all pages and converts them into text documents.

2. **✂️ Chunking the Document**  
   Long documents are split into manageable chunks (e.g., 1000 characters with overlap).  
   This helps the retriever find precise sections relevant to your question.

3. **🔍 Embeddings & Vector Store (FAISS)**  
   Each chunk is converted into a numeric vector using **HuggingFace sentence-transformers**.  
   These vectors are stored in a FAISS index, enabling fast similarity search.

4. **🧠 History-Aware Retrieval (RAG)**  
   - Your user question + chat history are used to build a standalone query.  
   - The system retrieves the top relevant chunks from the FAISS index.  
   - These chunks are passed as context to the language model.

5. **💬 Answer Generation with OpenAI**  
   The **ChatOpenAI** model (gpt-4o-mini) receives:
   - The user’s question  
   - The retrieved context  
   It then generates a concise answer. If it doesn’t know, it’s prompted to say so.

6. **📚 Source Transparency**  
   The exact chunks used to answer your question are displayed as “Sources” with page numbers,  
   so you can verify and read the original text.
    """)

    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
- **Frontend & UI**: Streamlit  
- **RAG Orchestration**: LangChain  
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)  
        """)
    with col2:
        st.markdown("""
- **Vector Store**: FAISS  
- **LLM**: OpenAI Chat API (gpt-4o-mini)  
- **PDF Loader**: LangChain `PyPDFLoader`  
        """)


def render_about_page():
    st.markdown('<h1 class="main-header">ℹ️ About</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
### What is Abdul bot?

Abdul bot is a **Retrieval-Augmented Generation (RAG)** powered assistant 
that helps you query and understand long PDF documents. Instead of manually scanning 
pages, you can simply ask questions in natural language and get concise, referenced answers.

### Why this project?

- To explore **practical GenAI** use cases beyond simple chat
- To build a **transparent** assistant that always shows where answers come from
- To experiment with **RAG patterns** (history-aware retriever, chunking, embeddings, etc.)

### How to use it?

1. Go to the **sidebar**, upload a PDF and click **"🚀 Process PDF"**  
2. Navigate to the **💬 Chat** page  
3. Ask any question related to the PDF’s content  
4. Expand the **📚 Sources** to see the exact snippets used to answer

### Notes

- Your document is processed in-memory (local session).  
- Chat history is preserved only while your current session is active.  
- You can clear chat or remove the PDF from the sidebar controls.
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, LangChain, HuggingFace, FAISS, and OpenAI.")


# ----------------- SIDEBAR & ROUTING ----------------- #

with st.sidebar:
    st.markdown("## 🔍 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "💬 Chat", "⚙️ How it works", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    st.header("📁 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process PDF"):
            process_pdf(uploaded_file)
    
    st.markdown("---")
    
    # Display statistics
    if st.session_state.pdf_processed:
        st.markdown(f"""
        <div class="stats-box">
            <h3>📊 Document Stats</h3>
            <p><strong>File:</strong> {st.session_state.pdf_name}</p>
            <p><strong>Chunks:</strong> {st.session_state.chunk_count}</p>
            <p><strong>Status:</strong> ✅ Ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("❌ Remove PDF"):
            st.session_state.vectorstore = None
            st.session_state.conversation_chain = None
            st.session_state.pdf_processed = False
            st.session_state.pdf_name = None
            st.session_state.chunk_count = 0
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### 📖 Quick Guide
    1. Upload a PDF
    2. Click **Process PDF**
    3. Open **💬 Chat** page
    4. Ask questions about the content
    
    ### ⚙️ Configuration
    - Model: gpt-4o-mini (OpenAI)
    - Embeddings: HuggingFace
    - Vector Store: FAISS
    - Chunk Size: 1000 chars
    """)

# ----------------- ROUTE TO SELECTED PAGE ----------------- #

if page.startswith("🏠"):
    render_home_page()
elif page.startswith("💬"):
    render_chat_page()
elif page.startswith("⚙️"):
    render_how_it_works_page()
else:
    render_about_page()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Streamlit, LangChain, HuggingFace & OpenAI</p>
    <p>🔒 Your data is processed locally within this session</p>
</div>
""", unsafe_allow_html=True)
