import streamlit as st
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API keys from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Set up the Streamlit interface
st.title("Staylist LLAMA 3 RAG Trial")

# Initialize the Groq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Convert DataFrame to text
        text = df.to_string(index=False)
        
        # Create a Document with the text
        from langchain_core.documents import Document
        document = Document(page_content=text)
        st.session_state.docs = [document]
        
        # Debug: Check if document is loaded correctly
        st.write(f"Document content: {document.page_content[:500]}")  # Display first 500 characters
        
        # Chunk Creation
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Debug: Check if documents are split correctly
        st.write(f"Number of chunks created: {len(st.session_state.final_documents)}")
        if len(st.session_state.final_documents) == 0:
            st.write("No chunks created. Please check the text splitter settings.")
            return
        
        # Vector OpenAI embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Define the path to the specific Excel file
excel_file_path = "Data - Inventory.xlsx"
st.markdown('Prompt Style:')
st.markdown('Body Shape:<Body Shape>, Occasion: <Ocassion>, Category: Dress. Recommend Top 3 Clothes Ranking them by most appropriate based on the suitability to the Body Shape and Occasion. Output Format= Title, Image Link, Meta attribute(Fabric, Length, Pattern, Neck, Occasion, Print, Shape, Sleeve Length, Sleeve Styling) for each Recommendations.')

# Text input for the user's question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to trigger document embedding
if st.button("Enter"):
    vector_embedding(excel_file_path)
    st.write("Vector Store DB Is Ready")

import time

# Processing the user's question
if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Display relevant document chunks in an expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Vector store is not ready. Please upload the document and click 'Enter' to initialize.")
