import PyPDF2
import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

@st.cache_data(show_spinner=False)
def process_textbook(uploaded_file):
    try:
        # 1. READ THE PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        # 2. BUILD THE FAISS DATABASE (Local)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local("faiss_index")
        
        # 3. GENERATE JSON DATA VIA GROQ (Cloud)
        toc_text = full_text[:6000] # Give enough text to find TOC
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # We tell Groq to be very brief to stay under character limits
        prompt = f"""
        Extract data from the Table of Contents of this book.
        
        INSTRUCTIONS:
        1. Extract the FIRST 8 main chapters.
        2. Format titles with Roman Numerals (I, II, III...).
        3. Each chapter summary must be EXACTLY 1 dense paragraph of 3 sentences.
        4. Include a 4-step 'roadmap'.
        5. Include 5 'exam_questions'.
        
        You must output ONLY valid JSON. 
        
        JSON STRUCTURE:
        {{
            "chapters": {{ "I. Chapter Name": ["Summary paragraph"] }},
            "roadmap": ["Step 1", "Step 2", "Step 3", "Step 4"],
            "exam_questions": ["Q1", "Q2", "Q3", "Q4", "Q5"]
        }}
        
        Text: {toc_text}
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            # 🚨 THIS FORCES GROQ TO OUTPUT PERFECT JSON 🚨
            response_format={"type": "json_object"} 
        )
        
        raw_response = response.choices[0].message.content
        return json.loads(raw_response)

    except Exception as e:
        print(f"Groq Processing Error: {e}")
        return {"Error Processing Book": [f"Details: {str(e)}", "The book structure was too complex. Try a smaller section."]}