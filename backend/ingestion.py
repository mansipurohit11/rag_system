import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load documents from the "data/" directory
def load_documents(data_dir: str):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs

# 2. Chunk documents into smaller pieces
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

# 3. Embed + Store chunks in a FAISS vectorstore
def build_vectorstore(chunks, save_path: str = "faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"Saved vectorstore to {save_path}/")
    return vectorstore

# Run it
if __name__ == "__main__":
    docs   = load_documents("data/")
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks)