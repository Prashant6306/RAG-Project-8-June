import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_milvus import Milvus
from pymilvus import connections
from langdetect import detect

from src.utils import preprocess_text
from src.config import MILVUS_CONFIG, EMBEDDING_MODEL_NAME


def load_and_preprocess_documents(data_path="/home/prashant/RAG1/ancient_greece_data"):
    if not os.path.exists(data_path):  
        print(f"⚠️ Folder '{data_path}' not found. Please upload files from UI.")
        return []

    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    for doc in docs:
        doc.page_content = preprocess_text(doc.page_content)
        # ✅ Ensure source metadata is present
        doc.metadata["source"] = doc.metadata.get("source", "unknown.txt").split("/")[-1]

    return docs


from torch import device as torch_device
import torch

def get_embedding_model():
    # Force CPU only if CUDA is not available
    device = torch_device("cuda" if torch.cuda.is_available() else "cpu")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},  # Use resolved device
        encode_kwargs={"normalize_embeddings": True}
    )

import asyncio

def build_and_save_vectorstore():
    docs = load_and_preprocess_documents()
    if not docs:
        print("⚠️ No documents found.")
        return

    embedding = get_embedding_model()
    connections.connect(host=MILVUS_CONFIG["host"], port=MILVUS_CONFIG["port"])

    async def async_build():
        return Milvus.from_documents(
            documents=docs,
            embedding=embedding,
            connection_args={"host": MILVUS_CONFIG["host"], "port": MILVUS_CONFIG["port"]},
            collection_name=MILVUS_CONFIG["collection_name"],
            drop_old=True
        )

    asyncio.run(async_build())
    print("✅ Vector store created successfully.")


def get_vectorstore_retriever():
    embedding = get_embedding_model()
    connections.connect(host=MILVUS_CONFIG["host"], port=MILVUS_CONFIG["port"])

    async def _get_vectorstore_async():
        return Milvus(
            embedding_function=embedding,
            connection_args={"host": MILVUS_CONFIG["host"], "port": MILVUS_CONFIG["port"]},
            collection_name=MILVUS_CONFIG["collection_name"]
        )

    vectorstore = asyncio.run(_get_vectorstore_async())

    return vectorstore.as_retriever(search_kwargs={"k": 3})


import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def ingest_uploaded_documents(uploaded_files, data_path="./data/uploaded"):
    os.makedirs(data_path, exist_ok=True)
    all_docs = []

    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Select appropriate loader
        if file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        elif file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            continue  # Skip unsupported

        docs = loader.load()
        for doc in docs:
            doc.page_content = preprocess_text(doc.page_content)
            doc.metadata["source"] = file.name
        all_docs.extend(docs)

    # Embed and insert into Milvus
    embedding = get_embedding_model()
    connections.connect(host=MILVUS_CONFIG["host"], port=MILVUS_CONFIG["port"])

    async def async_ingest():
        return Milvus.from_documents(
            documents=all_docs,
            embedding=embedding,
            connection_args={"host": MILVUS_CONFIG["host"], "port": MILVUS_CONFIG["port"]},
            collection_name=MILVUS_CONFIG["collection_name"],
            drop_old=False
        )

    asyncio.run(async_ingest())

    print(f"✅ Uploaded {len(uploaded_files)} files indexed to Milvus.")
