import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os
from langchain.schema import Document #Schema created in backend
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

#api config
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
gemini_model=genai.GenerativeModel('gemini-2.0-flash')

# Cache the HF embeddings to avoid slow reload of embeddings
@st.cache_resource(show_spinner='Loading Embeding model..')
def embeddings():
    return (HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"))
embedding_model = embeddings()

# userinterface
st.header("RAG Assistant: orange[HF embedings + Gemini LLM]")
st.subheader("Your personal assistant for document analysis and Q&A")
uploaded_file = st.file_uploader(label="Upload a PDF file", type="pdf")
if 