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
genai.configure(api_key=os.getenv("GOOGLE-API-KEY"))
gemini_model=genai.GenerativeModel('gemini-2.0-flash')

# Cache the HF embeddings to avoid slow reload of embeddings
@st.cache_resource(show_spinner='Loading Embeding model..')
def embeddings():
    return (HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"))
embedding_model = embeddings()

# userinterface
st.header("RAG Assistant: HF embedings + Gemini LLM")
st.subheader("Your personal assistant for document analysis and Q&A")
uploaded_file = st.file_uploader(label="Upload a PDF file", type="pdf")
if uploaded_file:
    raw_text = ""
    pdf = PdfReader(uploaded_file)
    for i,page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    if raw_text.strip():
        document = Document(page_content=raw_text)
        # using CharacterTextSplitter to split the document into smaller chunks and pass it to model
        splitter= CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks = splitter.split_documents([document])
        
        # store the chunks into FAISS vectorstore
        chunk_pieces = [chunk.page_content for chunk in chunks]
        vectordb = FAISS.from_texts(chunk_pieces, embedding_model) # convert them into vector embeddings
        retriever = vectordb.as_retriever() # retrieve the vector embeddings
        
        st.success("Embeddings are generated.Ask your questions!")
        user_input = st.text_input(label="Ask a question :")
        if user_input:
            with st.chat_message('user'):
                st.write(user_input)
                
            with st.spinner("Generating response..."):
                relevant_docs= retriever.get_relevant_documents(user_input)
                context = '\n\n'.join([doc.page_content for doc in relevant_docs])
                prompt = f'''You are an expert assistant. Use the context below to answer the query.
                if unsure or information not avialable in the doc.Look into other sources
                context:{context}
                query:{user_input} Answer:'''
                response = gemini_model.generate_content(prompt)
                st.markdown('Answer: ')
                st.write(response.text)
    else:
        st.warning("No text could be extracted from PDF.Please upload a readable PDF file.")
                
            

