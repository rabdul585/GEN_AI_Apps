
import os
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
from tavily import TavilyClient

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

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    st.warning("TAVILY_API_KEY not found in .env. Web search fallback will not work.")
else:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ---------------------------------------------------------
# 2. Paths & constants
# ---------------------------------------------------------
DEFAULT_COMPANY_DOMAIN = os.getenv("COMPANY_DOMAIN", "example.com")
USER_GUIDE_PATH = os.path.join("data", "user_guide.txt")

# ---------------------------------------------------------
# 3. Load User Guide
# ---------------------------------------------------------
def load_user_guide_docs() -> List[Document]:
    if not os.path.exists(USER_GUIDE_PATH):
        st.error(f"User guide not found at: {USER_GUIDE_PATH}")
        st.stop()

    if USER_GUIDE_PATH.lower().endswith(".pdf"):
        loader = PyPDFLoader(USER_GUIDE_PATH)
    else:
        loader = TextLoader(USER_GUIDE_PATH, encoding="utf-8")

    return loader.load()


def build_vectorstore(documents: List[Document]) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# ---------------------------------------------------------
# 4. Tavily Website Search
# ---------------------------------------------------------
def make_company_search_tool(company_domain: str) -> Optional[Tool]:
    if not TAVILY_API_KEY:
        return None

    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    def company_search(query: str) -> str:
        try:
            result = tavily.search(
                query=query,
                include_domains=[company_domain],
                max_results=3,
            )
        except Exception as e:
            return f"[Tavily error] {e}"

        results = result.get("results", [])
        if not results:
            return "No relevant website results."

        snippets = []
        for r in results:
            content = r.get("content", "")
            url = r.get("url", "")
            if content:
                snippets.append(f"{content}\n(Source: {url})")

        return "\n\n".join(snippets)

    return Tool(
        name="company_web_search",
        func=company_search,
        description=f"Search {company_domain} using Tavily.",
    )

# ---------------------------------------------------------
# 5. Build RAG Chain (Source = User Guide OR Website)
# ---------------------------------------------------------
def build_rag_chain(vectorstore: FAISS, temperature: float, company_domain: str):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
    )

    company_search_tool = make_company_search_tool(company_domain)

    system_prompt = """
You are a production-grade support chatbot.

Preferred sources:
1. User Guide (primary)
2. Company Website (fallback)

Rules:
- Use User Guide first.
- Only use website if User Guide is insufficient.
- Never show internal document verbatim.
- If both fail, ask user to contact support.
"""

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs) if docs else ""

    retrieval_chain = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )

    decision_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
                + """
If the User Guide context is insufficient, reply ONLY with:
NEED_WEB_SEARCH

Otherwise answer using only the User Guide.
"""
            ),
            ("human", "Question: {question}\n\nUser Guide Context:\n{context}")
        ]
    )

    def answer_with_fallback(inputs: dict) -> dict:
        question = inputs["question"]
        context = inputs["context"]

        decision = llm.invoke(
            decision_prompt.format_messages(
                question=question,
                context=context
            )
        ).content.strip()

        # ---- Website fallback ----
        if decision == "NEED_WEB_SEARCH":
            if company_search_tool is None:
                return {
                    "answer": "User Guide has no information and website search is unavailable.",
                    "source": "Website",
                }

            web_result = company_search_tool.run(question)

            if web_result.startswith("[Tavily error]"):
                return {
                    "answer": "User Guide has no info and website search failed. Contact support.",
                    "source": "User Guide",
                }

            final_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        """Question: {question}

User Guide Context:
{context}

Website Result:
{web_result}

Provide the final answer.
"""
                    )
                ]
            )

            answer = llm.invoke(
                final_prompt.format_messages(
                    question=question,
                    context=context,
                    web_result=web_result
                )
            ).content

            return {"answer": answer, "source": "Website"}

        # ---- User Guide answer ----
        return {"answer": decision, "source": "User Guide"}

    return retrieval_chain | answer_with_fallback

# ---------------------------------------------------------
# 6. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Product RAG Chatbot", page_icon="💬")

st.title("💬 Product Support RAG Chatbot")
st.write("Using **preloaded User Guide** from `data/user_guide.txt`.")

with st.sidebar:
    st.header("Settings")

    company_domain = st.text_input(
        "Company Domain",
        value=DEFAULT_COMPANY_DOMAIN,
        help="Example: yourcompany.com"
    )

    temperature = st.slider(
        "Model Temperature",
        0.0, 1.0, 0.2, 0.05
    )

    st.success(f"Using: `{USER_GUIDE_PATH}`")

# ---------------------------------------------------------
# 7. Vectorstore Load (once) + RAG chain + chat history
# ---------------------------------------------------------
if "vectorstore" not in st.session_state:
    with st.spinner("Loading user guide..."):
        docs = load_user_guide_docs()
        st.session_state.vectorstore = build_vectorstore(docs)

# Build / refresh RAG chain (depends on temp + domain)
st.session_state.rag_chain = build_rag_chain(
    st.session_state.vectorstore,
    temperature,
    company_domain
)

# Initialise chat history
if "chat_history" not in st.session_state:
    # each item: {"question": str, "answer": str, "source": "User Guide" | "Website"}
    st.session_state.chat_history = []

# ---------------------------------------------------------
# 8. Chat input + display history
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Ask a question")

user_question = st.text_input(
    "Your question:",
    placeholder="e.g., How do I reset my password?",
    key="user_question",
)

if st.button("Ask") and user_question.strip():
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.rag_chain.invoke(user_question)
            answer = result.get("answer", "")
            source = result.get("source", "Unknown")

            # Save to history
            st.session_state.chat_history.append(
                {
                    "question": user_question,
                    "answer": answer,
                    "source": source,
                }
            )
        except Exception as e:
            st.error(f"Error while generating answer: {e}")

# Show chat history (most recent on top)
if st.session_state.chat_history:
    st.markdown("### Conversation History")

    # iterate in reverse so latest question appears first
    for i, turn in enumerate(reversed(st.session_state.chat_history), start=1):
        st.markdown(f"**Q{i}.** {turn['question']}")
        st.markdown(f"**A{i}.** {turn['answer']}")
        st.markdown(
            f"<div style='color:gray; font-size:0.85rem;'>Source: "
            f"<b>{turn['source']}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")