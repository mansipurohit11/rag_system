from langchain_community.vectorstores import FAISS
from openai import embeddings
from sentence_transformers import CrossEncoder

vectorstore = FAISS.load_local("faiss_index", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Reranker — narrows top-10 down to top-3 with better accuracy
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query: str, top_k: int = 3):
    candidates = retriever.invoke(query)
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [doc for _, doc in ranked[:top_k]]