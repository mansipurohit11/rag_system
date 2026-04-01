# backend/main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # save, load, chunk, embed, store
    ...
    return {"status": "indexed", "chunks": len(chunks)}

@app.post("/query")
async def query(req: QueryRequest):
    docs = retrieve_and_rerank(req.question)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Answer based on context only:\n{context}\n\nQuestion: {req.question}"
    answer = llm.invoke(prompt)
    return {
        "answer": answer.content,
        "sources": [d.metadata for d in docs]
    }