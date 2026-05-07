"""
PDF Q&A Bot — Agentic AI Project
Author: Malliswari Kasireddy
Stack: LangChain · FAISS · OpenAI/Claude · Streamlit
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

load_dotenv()

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Q&A Bot")
st.markdown(
    "Upload any PDF and have a conversation with it using AI. "
    "Powered by **LangChain + FAISS + OpenAI**."
)
st.divider()


# ─────────────────────────────────────────────
#  Session State Init
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ─────────────────────────────────────────────
#  Helper: Build RAG Chain
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(file_bytes: bytes, filename: str):
    """
    1. Load PDF → 2. Chunk text → 3. Embed into FAISS
    4. Wrap in a ConversationalRetrievalChain with memory
    """
    # Save uploaded bytes to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # 1. Load
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # 2. Chunk  (500 chars, 50 overlap keeps context across splits)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # 3. Embed + store in FAISS (runs locally — no extra cost)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Conversational chain with sliding-window memory
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )
    return chain


# ─────────────────────────────────────────────
#  Sidebar — Upload + Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing PDF... building vector store..."):
            st.session_state.chain = build_rag_chain(
                uploaded_file.read(), uploaded_file.name
            )
            st.session_state.pdf_processed = True
            st.session_state.chat_history = []

        st.success(f"✅ Ready: **{uploaded_file.name}**")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown(
        "1. PDF is split into chunks\n"
        "2. Each chunk is embedded via OpenAI\n"
        "3. Your question retrieves the top 4 relevant chunks\n"
        "4. GPT answers using only those chunks (RAG)"
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────
#  Main — Chat Interface
# ─────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user input
    if user_question := st.chat_input("Ask anything about your PDF..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        # Get AI answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain({"question": user_question})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

            st.markdown(answer)

            # Show source page references
            if source_docs:
                with st.expander("📎 Source pages used"):
                    seen = set()
                    for doc in source_docs:
                        page = doc.metadata.get("page", "?")
                        if page not in seen:
                            seen.add(page)
                            st.markdown(f"- **Page {page + 1}**")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
