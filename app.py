import streamlit as st
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

# Load Environment Variables
load_dotenv()

# Check Gemini API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="RAG PDF Chatbot")
st.title("📚 RAG PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

if uploaded_file is not None:

    # Save PDF
    pdf_path = uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully!")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split Text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS Vector Database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    # Question Input
    question = st.text_input(
        "Ask a Question About the PDF"
    )

    if question:

        # Retrieve Documents
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt = f"""
You are a helpful AI assistant.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        try:

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

        except Exception as e:

            st.error(f"Error: {str(e)}")