import os
import sys
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.tools import Tool

import streamlit.components.v1 as components
from pyvis.network import Network


# ---------------------------------------------------------
# 1. Load .env and API keys
# ---------------------------------------------------------
load_dotenv()


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Optional: Tavily API for web search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Neo4j KG environment values
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ---------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------
USER_GUIDE_PATH = os.path.join("data", "user_guide.txt")
DEFAULT_COMPANY_DOMAIN = os.getenv("COMPANY_DOMAIN", "example.com")


# ---------------------------------------------------------
# 3. Load User Guide
# ---------------------------------------------------------
def load_user_guide_docs() -> List[Document]:
    """Load documents from user guide file"""
    if not os.path.exists(USER_GUIDE_PATH):
        st.error(f"User guide not found at: {USER_GUIDE_PATH}")
        st.info("Please create a file at data/user_guide.txt with your documentation.")
        st.stop()

    try:
        if USER_GUIDE_PATH.lower().endswith(".pdf"):
            loader = PyPDFLoader(USER_GUIDE_PATH)
        else:
            loader = TextLoader(USER_GUIDE_PATH, encoding="utf-8")
        
        docs = loader.load()
        if not docs:
            st.warning("User guide file is empty.")
        return docs
    except Exception as e:
        st.error(f"Error loading user guide: {e}")
        st.stop()


def build_vectorstore(documents: List[Document]) -> FAISS:
    """Build FAISS vector store from documents"""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        splits = splitter.split_documents(documents)
        
        if not splits:
            st.warning("No text chunks created from documents.")
            
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error building vector store: {e}")
        st.stop()


# ---------------------------------------------------------
# 4. Tavily Website Search (Optional)
# ---------------------------------------------------------
def make_company_search_tool(company_domain: str) -> Optional[Tool]:
    """Create Tavily web search tool if API key is available"""
    if not TAVILY_API_KEY:
        return None

    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=TAVILY_API_KEY)

        def company_search(query: str) -> str:
            try:
                result = tavily.search(
                    query=query,
                    include_domains=[company_domain],
                    max_results=3,
                )
                
                results = result.get("results", [])
                if not results:
                    return "No relevant results found."

                collected = []
                for r in results:
                    text = r.get("content", "")
                    url = r.get("url", "")
                    if text:
                        collected.append(f"{text}\n(Source: {url})")

                return "\n\n".join(collected)
            except Exception as e:
                return f"[Search error] {e}"

        return Tool(
            name="company_web_search",
            func=company_search,
            description=f"Search company website: {company_domain}",
        )
    except ImportError:
        return None


# ---------------------------------------------------------
# 5. Neo4j Knowledge Graph Visualization
# ---------------------------------------------------------
def visualize_knowledge_graph(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    output_file: str = "knowledge_graph.html",
    max_nodes: int = 100
):
    """
    Generate interactive knowledge graph visualization from Neo4j
    Uses elementId() instead of deprecated id() function
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("neo4j package not installed. Run: pip install neo4j")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    try:
        with driver.session() as session:
            # Fetch nodes using elementId()
            nodes_query = f"""
            MATCH (n)
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as properties
            LIMIT {max_nodes}
            """
            
            nodes_result = session.run(nodes_query)
            nodes = []
            node_ids = []
            
            for record in nodes_result:
                node_id = record["id"]
                labels = record["labels"]
                props = record["properties"]
                
                # Get node label and title
                label = labels[0] if labels else "Node"
                title = props.get("name", props.get("title", str(node_id)[:10]))
                
                nodes.append({
                    "id": node_id,
                    "label": label,
                    "title": title,
                    "properties": props
                })
                node_ids.append(node_id)
            
            # Fetch relationships using elementId()
            edges = []
            if node_ids:
                id_list = ", ".join([f"'{nid}'" for nid in node_ids])
                
                edges_query = f"""
                MATCH (a)-[r]->(b)
                WHERE elementId(a) IN [{id_list}] AND elementId(b) IN [{id_list}]
                RETURN elementId(a) as source, elementId(b) as target, 
                       type(r) as type, properties(r) as properties
                LIMIT 200
                """
                
                edges_result = session.run(edges_query)
                
                for record in edges_result:
                    edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "properties": record["properties"]
                    })
        
        # Create visualization using Pyvis
        net = Network(
            height="600px", 
            width="100%", 
            bgcolor="#1e1e1e", 
            font_color="white",
            directed=True
        )
        
        # Physics settings for better layout
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=100,
            spring_strength=0.001,
            damping=0.09
        )
        
        # Add nodes with colors based on label
        color_map = {
            "Person": "#FF6B6B",
            "Organization": "#4ECDC4",
            "Location": "#45B7D1",
            "Document": "#FFA07A",
            "Concept": "#98D8C8",
        }
        
        for node in nodes:
            color = color_map.get(node["label"], "#97C2FC")
            net.add_node(
                node["id"],
                label=node["title"],
                title=f"{node['label']}: {node['title']}",
                color=color,
                size=25
            )
        
        # Add edges
        for edge in edges:
            net.add_edge(
                edge["source"],
                edge["target"],
                title=edge["type"],
                label=edge["type"],
                arrows="to"
            )
        
        # Save to HTML
        net.save_graph(output_file)
        
        return output_file
        
    except Exception as e:
        raise Exception(f"Neo4j query error: {str(e)}")
    finally:
        driver.close()


# ---------------------------------------------------------
# 6. Build RAG Chain
# ---------------------------------------------------------
def build_rag_chain(vectorstore: FAISS, temperature: float, company_domain: str):
    """Build the RAG chain with retrieval and LLM"""
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
    )

    company_search_tool = make_company_search_tool(company_domain)

    system_prompt = (
        "You are a helpful support chatbot.\n\n"
        "Information sources:\n"
        "1. User Guide documentation (primary source)\n"
        "2. Company website (fallback if needed)\n\n"
        "Guidelines:\n"
        "- Always prioritize information from the User Guide\n"
        "- Use clear, concise language\n"
        "- If information is not available, be honest about it\n"
        "- Provide helpful suggestions when possible\n"
    )

    decision_instructions = (
        "\nIf the User Guide context does not contain enough information "
        "to answer the question, reply with exactly:\n"
        "NEED_WEB_SEARCH\n\n"
        "Otherwise, provide a helpful answer based on the User Guide.\n"
    )

    def format_docs(docs: List[Document]) -> str:
        """Format retrieved documents into a string"""
        return "\n\n".join(d.page_content for d in docs) if docs else ""

    retrieval_chain = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )

    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + decision_instructions),
        ("human", "Question: {question}\n\nUser Guide Context:\n{context}"),
    ])

    def answer_with_fallback(inputs: dict) -> dict:
        """Process query with fallback to web search if needed"""
        question = inputs["question"]
        context = inputs["context"]

        # First attempt: Use User Guide
        decision = llm.invoke(
            decision_prompt.format_messages(
                question=question,
                context=context
            )
        ).content.strip()

        # Check if web search is needed
        if decision == "NEED_WEB_SEARCH":
            if company_search_tool is None:
                return {
                    "question": question,
                    "answer": (
                        "I couldn't find enough information in the User Guide to answer your question. "
                        "Web search is not configured. Please contact support for assistance."
                    ),
                    "source": "User Guide",
                }

            # Perform web search
            web_result = company_search_tool.run(question)

            if web_result.startswith("[Search error]"):
                return {
                    "question": question,
                    "answer": (
                        "I couldn't find sufficient information in the User Guide, "
                        "and the web search encountered an error. Please contact support."
                    ),
                    "source": "User Guide",
                }

            # Generate answer using both sources
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                (
                    "human",
                    "Question: {question}\n\n"
                    "User Guide Context:\n{context}\n\n"
                    "Additional Web Information:\n{web_result}\n\n"
                    "Please provide a comprehensive answer using both sources.\n"
                ),
            ])

            answer = llm.invoke(
                final_prompt.format_messages(
                    question=question,
                    context=context,
                    web_result=web_result
                )
            ).content

            return {
                "question": question,
                "answer": answer,
                "source": "User Guide + Web",
            }

        # Return User Guide response
        return {
            "question": question,
            "answer": decision,
            "source": "User Guide",
        }

    return retrieval_chain | answer_with_fallback


# ---------------------------------------------------------
# 7. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="RAG + Knowledge Graph Chatbot", 
    page_icon="💬", 
    layout="wide"
)

st.title("💬 Product Support Chatbot")
st.markdown("*Powered by RAG + Knowledge Graph*")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("RAG Settings")
    company_domain = st.text_input(
        "Company Domain:",
        value=DEFAULT_COMPANY_DOMAIN,
        help="Domain to search for additional information"
    )

    temperature = st.slider(
        "Response Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower = more focused, Higher = more creative"
    )

    st.markdown("---")
    
    # Status indicators
    st.subheader("📊 System Status")
    
    if os.path.exists(USER_GUIDE_PATH):
        st.success(f"✅ User Guide: Ready")
    else:
        st.error(f"❌ User Guide: Not found")
        st.caption(f"Expected at: {USER_GUIDE_PATH}")

    if TAVILY_API_KEY:
        st.success("✅ Web Search: Enabled")
    else:
        st.info("ℹ️ Web Search: Disabled")
    
    st.markdown("---")
    
    st.subheader("🔗 Knowledge Graph")
    
    if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:
        st.success("✅ Neo4j: Connected")
    else:
        st.warning("⚠️ Neo4j: Not configured")
        st.caption("Add credentials to .env to enable")


# ---------------------------------------------------------
# 8. Initialize Session State and Load Data
# ---------------------------------------------------------
if "vectorstore" not in st.session_state:
    with st.spinner("🔄 Loading User Guide and building vector store..."):
        try:
            docs = load_user_guide_docs()
            st.session_state.vectorstore = build_vectorstore(docs)
            st.success("✅ User Guide loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load User Guide: {e}")
            st.stop()

# Build RAG chain
if "rag_chain" not in st.session_state or st.session_state.get("last_temp") != temperature:
    st.session_state.rag_chain = build_rag_chain(
        st.session_state.vectorstore,
        temperature,
        company_domain
    )
    st.session_state.last_temp = temperature

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# 9. Main Layout - Two Columns
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

# Left Column: Chat Interface
with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question:",
            placeholder="Example: How do I reset my password?",
            key="user_input"
        )
        submit_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
    
    if submit_button and user_question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.rag_chain.invoke(user_question)
                st.session_state.chat_history.append(result)
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.exception(e)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### 📜 Conversation History")
        
        # Show last 5 conversations (most recent first)
        for idx, turn in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.chat_history)-idx}:** {turn['question']}")
                st.markdown(f"**A:** {turn['answer']}")
                st.caption(f"📚 Source: {turn['source']}")
                st.divider()
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("👋 Ask a question to get started!")


# Right Column: Knowledge Graph
with col2:
    st.markdown("### 📊 Knowledge Graph Visualization")
    
    if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
        st.info(
            "🔗 **Neo4j Configuration Required**\n\n"
            "To enable Knowledge Graph visualization, add these to your `.env` file:\n"
            "```\n"
            "NEO4J_URI=bolt://localhost:7687\n"
            "NEO4J_USERNAME=neo4j\n"
            "NEO4J_PASSWORD=your-password\n"
            "```"
        )
    else:
        max_nodes = st.slider(
            "Maximum nodes to display:",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
            help="Limit the number of nodes for better performance"
        )

        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            generate_btn = st.button("🔄 Generate Graph", type="primary", use_container_width=True)
        
        with col_btn2:
            if "kg_graph_file" in st.session_state:
                clear_btn = st.button("🗑️ Clear Graph", use_container_width=True)
                if clear_btn:
                    del st.session_state["kg_graph_file"]
                    st.rerun()

        if generate_btn:
            with st.spinner("🎨 Generating knowledge graph..."):
                try:
                    output_file = "knowledge_graph.html"
                    visualize_knowledge_graph(
                        neo4j_uri=NEO4J_URI,
                        neo4j_user=NEO4J_USERNAME,
                        neo4j_password=NEO4J_PASSWORD,
                        output_file=output_file,
                        max_nodes=max_nodes,
                    )
                    st.session_state["kg_graph_file"] = output_file
                    st.success("✅ Graph generated successfully!")
                except Exception as e:
                    st.error(f"❌ Graph generation failed: {e}")
                    st.exception(e)

        # Display graph if it exists
        if "kg_graph_file" in st.session_state:
            try:
                with open(st.session_state["kg_graph_file"], "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                st.markdown("---")
                components.html(html_content, height=600, scrolling=True)
                
                st.caption("💡 Tip: Click and drag to explore, scroll to zoom")
            except Exception as e:
                st.error(f"Unable to display graph: {e}")
        else:
            st.info("👆 Click 'Generate Graph' to visualize the knowledge graph")


# ---------------------------------------------------------
# 10. Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit • LangChain • OpenAI • Neo4j"
    "</div>",
    unsafe_allow_html=True
)