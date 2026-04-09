# frontend/app.py
import streamlit as st
import requests

st.title("Chat with your documents")

uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded:
    r = requests.post("http://localhost:8000/upload", files={"file": uploaded})
    st.success(f"Indexed! ({r.json()['chunks']} chunks)")

question = st.text_input("Ask a question")
if question:
    r = requests.post("http://localhost:8000/query", json={"question": question})
    data = r.json()
    st.write(data["answer"])
    with st.expander("Sources"):
        for s in data["sources"]:
            st.write(s)