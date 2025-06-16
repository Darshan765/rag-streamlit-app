import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="RAG Chat App", page_icon="🤖")

st.title("🔍 Simple RAG (Retrieval-Augmented Generation) App")

# OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])

query = st.text_input("Ask a question about your document:")

if uploaded_file and query and openai_api_key:
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. Load and process the file
    loader = TextLoader("temp.txt")
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. Generate Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Create retriever and LLM
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")

    # 5. RAG Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # 6. Answer the query
    result = rag_chain.run(query)
    st.subheader("🔎 Answer:")
    st.write(result)
