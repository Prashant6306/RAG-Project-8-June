import sys
import os

# Add parent directory (where 'src' lives) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import get_vectorstore_retriever, build_and_save_vectorstore, ingest_uploaded_documents
from src.llm_chain import build_rag_chain

import streamlit as st
import time
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(page_title="Ancient Greece Chatbot", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📜 Ancient Greece Chatbot")

# Sidebar Developer Tools
st.sidebar.header("🧪 Developer Options")
rag_mode = st.sidebar.toggle("Enable RAG Mode", value=True)
show_chunks = st.sidebar.toggle("Show Retrieved Chunks", value=False)
show_latency = st.sidebar.toggle("Show Latency Timer", value=True)
show_sources = st.sidebar.toggle("Show Source Citations", value=True)

st.sidebar.header("🛠️ Admin Tools")
# if st.sidebar.button("🔄 Rebuild Vector DB"):
#     build_and_save_vectorstore()
#     st.sidebar.success("✅ Milvus collection rebuilt.")

uploaded_files = st.sidebar.file_uploader("📂 Upload New Documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)
if uploaded_files:
    ingest_uploaded_documents(uploaded_files)
    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) embedded and indexed.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load retriever and RAG chain
retriever = get_vectorstore_retriever()
rag_chain = build_rag_chain(retriever)

# Display welcome message
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant"):
        st.markdown("Hello! Ask me anything about Ancient Greece 🏛️")

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"🕒 *{msg['timestamp']}* — {msg['question']}")
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
        if show_sources and "retrieved_docs" in msg:
            with st.expander("📚 Source Citations"):
                for i, doc in enumerate(msg["retrieved_docs"], 1):
                    st.markdown(f"**Source {i}** — `{doc.metadata.get('source', 'unknown')}`\n\n{doc.page_content[:300]}...")

# Chat input box
query = st.chat_input("Ask a question...")
if query:
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "question": query,
        "answer": "...",
        "timestamp": now
    })

    with st.chat_message("user"):
        st.markdown(f"🕒 *{now}* — {query}")

    start_time = time.time()

    if rag_mode:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = rag_chain.invoke(query)
        st.session_state.chat_history[-1]["retrieved_docs"] = docs  # ✅ Save retrieved docs for citation
    else:
        from langchain_ollama import ChatOllama
        from src.config import OLLAMA_MODEL_NAME, OLLAMA_URL
        llm_direct = ChatOllama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_URL)
        response = llm_direct.invoke(query).content
        docs = []

    latency = time.time() - start_time
    st.session_state.chat_history[-1]["answer"] = response

    if show_chunks:
        st.session_state.chat_history[-1]["retrieved_chunks"] = [doc.page_content[:300] for doc in docs]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for word in response.split():
            full_text += word + " "
            placeholder.markdown(full_text + "▌")
            #time.sleep(0.02)
        placeholder.markdown(full_text)

        if show_latency:
            st.caption(f"⚡ Response Time: {latency:.2f} sec")

# Export chat history as .txt and .csv
chat_export_txt = ""
chat_export_csv = []

for msg in st.session_state.chat_history:
    chat_export_txt += f"[{msg['timestamp']}] You: {msg['question']}\n"
    chat_export_txt += f"[{msg['timestamp']}] Bot: {msg['answer']}\n\n"
    chat_export_csv.append({"timestamp": msg['timestamp'], "question": msg['question'], "answer": msg['answer']})

st.download_button("💾 Download Chat (.txt)", chat_export_txt, file_name="chat_history.txt")

if chat_export_csv:
    df = pd.DataFrame(chat_export_csv)
    st.download_button("📊 Download Q&A Log (.csv)", df.to_csv(index=False), file_name="chat_log.csv", mime="text/csv")
