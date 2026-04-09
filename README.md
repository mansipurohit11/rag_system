# 🧠 RAG System — Chat With Your Documents

> because ctrl+F is not going to cut it anymore

Tired of reading 200-page PDFs like a normal person? same.

Upload any document, ask it questions in plain English, get answers back with actual sources. No hallucinations (we literally measure those). Built this to actually understand RAG from scratch, not just copy paste from docs and pretend I know what's happening.

---

## what's inside

```
backend/
├── ingestion.py    # load → chunk → embed → store
├── retrieval.py    # query → retrieve → rerank
└── main.py         # FastAPI

frontend/
└── app.py          # Streamlit UI

evaluation/
└── eval.py         # RAGAS (because vibes aren't a metric)
```

---

## stack

| thing | why |
|---|---|
| LangChain | orchestrates everything |
| FAISS / Qdrant | vector storage |
| BGE + OpenAI embeddings | compared both (BGE is free and honestly fine) |
| cross-encoder reranker | top-10 → top-3, actually makes a difference |
| GPT-4o-mini | fast, cheap, does the job |
| FastAPI + Streamlit | backend + UI |
| RAGAS | faithfulness / relevancy / recall scores |

---

## run it

```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
pip install -r requirements.txt

# add OPENAI_API_KEY to .env
# drop PDFs in /data

python backend/ingestion.py
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

that's it. go to `localhost:8501` and talk to your documents.

---

## what I learned

chunking is an art. reranking helps way more than expected. RAGAS scores will humble you. and apparently I can build a full RAG pipeline from scratch which is cool I guess.

---

*built with too many terminal tabs open and a concerning interest in vector databases.*
