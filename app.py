import streamlit as st
import time
import json
from typing import Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# ══════════════════════════════════════════════
#  ⚙️  CONFIG
# ══════════════════════════════════════════════
VECTOR_DB_PATH = r"D:\HACKATHONS and PROJECTS\Project ChatBot(SQL)\VectorDB"
COLLECTION_NAME = "handwritten_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K = 5

st.set_page_config(page_title="SQL AI Agent", page_icon="🧠", layout="wide")

# CSS Styling
st.markdown("""
<style>
:root {
    --bg:#0a0a0f; --bg2:#111118; --surface:#16161f; --border:#2a2a3a;
    --accent:#7c6af7; --accent2:#4fd1c5; --text:#e8e8f0;
}
html,body,.stApp { background:var(--bg)!important; color:var(--text)!important; }
[data-testid="stSidebar"] { background:var(--bg2)!important; border-right:1px solid var(--border)!important; }
.user-msg { background:#1a1a2e; border-radius:12px; padding:12px 16px; margin:4px 0; }
.ai-msg { background:#111118; border-radius:12px; padding:12px 16px; margin:4px 0; }
.stChatInput { background:var(--surface)!important; border:1px solid var(--border)!important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

@st.cache_resource
def get_llm():
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1024
    )

# ---------------------------------------------------------------------------
# Vector Database Setup
# ---------------------------------------------------------------------------

@st.cache_resource
def get_vector_store(collection_name):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectordb

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

@st.cache_resource
def get_available_collections(db_path):
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        st.error(f"Error reading collections: {e}")
        return []

# ---------------------------------------------------------------------------
# Single Tool - Forced Search
# ---------------------------------------------------------------------------

def create_sql_search_tool(vectordb, top_k):
    @tool
    def search_sql_database(query: str) -> str:
        """
        Search the SQL documentation for relevant information.
        Use this tool for every question.
        """
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k}
        )
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "I couldn't find any relevant information in the SQL documentation about this topic."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"[Source {i}]:\n{doc.page_content}\n")
        
        return "\n".join(results)
    
    return search_sql_database

# ---------------------------------------------------------------------------
# Simple QA Chain - Alternative to Agent (More Reliable)
# ---------------------------------------------------------------------------

def create_simple_qa(collection_name, top_k):
    """Simpler approach: Direct Q&A without agent complexity"""
    vectordb = get_vector_store(collection_name)
    llm = get_llm()
    
    def answer_question(question: str) -> str:
        # First, search the database
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k}
        )
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return "I couldn't find any information about this in the SQL documentation. Please ask me about SQL concepts like SELECT, JOIN, GROUP BY, etc."
        
        # Combine the retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt for the LLM
        prompt = f"""You are a SQL expert assistant. Answer the question based ONLY on the following context from the SQL documentation.

Context from SQL documentation:
{context}

Question: {question}

Answer (based only on the context above, do not use your own knowledge):"""
        
        # Get answer from LLM
        response = llm.invoke(prompt)
        return response
    
    return answer_question, vectordb

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 SQL AI Agent")
    st.markdown(f"Powered by **{OLLAMA_MODEL}** · Ollama")
    
    available_collections = get_available_collections(VECTOR_DB_PATH)
    
    if available_collections:
        selected_collection = st.selectbox(
            "📚 Select Collection",
            options=available_collections,
            help="Choose which vector database collection to query"
        )
    else:
        selected_collection = "handwritten_notes"
        st.warning("⚠️ No collections found!")
    
    top_k_value = st.slider("🔎 Number of chunks", 1, 10, TOP_K)
    
    st.divider()
    st.info("🔍 **Mode**: RAG (Retrieval-Augmented Generation)\n\nAnswers are based ONLY on the SQL documentation")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        if selected_collection:
            collection = client.get_collection(selected_collection)
            chunk_count = collection.count()
        else:
            chunk_count = "?"
    except:
        chunk_count = "?"
    
    st.markdown("### ℹ️ Database Info")
    st.markdown(f"""
    - 📁 DB: `{VECTOR_DB_PATH.split(chr(92))[-1]}`
    - 🗂️ Collection: `{selected_collection}`
    - 📄 Chunks: `{chunk_count}`
    """)

st.markdown("## 📚 SQL Documentation Assistant")
st.caption(f"**{selected_collection}** | Model: **{OLLAMA_MODEL}** | Mode: **RAG (No Hallucination)**")
st.warning("⚠️ **Note:** This assistant ONLY answers SQL-related questions based on the documentation. Non-SQL questions will be rejected.", icon="🔍")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.qa_chain = None
    st.session_state.current_config = None

current_config = f"{selected_collection}_{top_k_value}"
if st.session_state.current_config != current_config:
    st.session_state.qa_chain = None
    st.session_state.current_config = current_config

if st.session_state.qa_chain is None:
    with st.spinner("🔧 Loading SQL Documentation..."):
        try:
            st.session_state.qa_chain, st.session_state.vector_store = create_simple_qa(selected_collection, top_k_value)
        except Exception as e:
            st.error(f"Failed to load: {e}")
            st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask about SQL (e.g., 'what is a JOIN', 'how to use GROUP BY')..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)

    # ============================
    # 🧠 CONTEXT BUILDING (NEW FIX)
    # ============================
    
    recent_context = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]]
    )

    combined_query = f"""
Previous conversation:
{recent_context}

Current question:
{query}

If this is a follow-up question (like "that", "it", "this", "why", "how"), resolve it using context.
"""

    # ============================
    # 🔍 SQL CHECK (FIXED LOGIC)
    # ============================

    sql_keywords = [
        'sql', 'database', 'query', 'select', 'join', 'group by',
        'order by', 'where', 'insert', 'update', 'delete', 'table',
        'index', 'transaction', 'window function', 'cte',
        'subquery', 'primary key', 'foreign key'
    ]

    is_sql_related = any(
        k in (query.lower() + recent_context.lower())
        for k in sql_keywords
    )

    # ============================
    # 🧾 RESPONSE AREA
    # ============================

    if not is_sql_related:
        with st.chat_message("assistant"):
            response = (
                "I can only help with SQL and database topics. "
                "Try asking about SELECT, JOIN, GROUP BY, etc."
            )
            st.markdown(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔍 Searching SQL documentation...")

            t0 = time.time()

            try:
                # ============================
                # 🔥 FIX: USE CONTEXT-AWARE QUERY
                # ============================
                answer = st.session_state.qa_chain(combined_query)

                elapsed = time.time() - t0

                placeholder.markdown(answer)
                st.caption(f"⏱️ {elapsed:.1f}s | 📚 from {selected_collection}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                placeholder.error(f"Error: {str(e)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {str(e)}"}
                )