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
        # 1. READ THE ENTIRE PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        # 2. BUILD THE LOCAL VECTOR DATABASE (FAISS)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local("faiss_index")
        
        # 3. GENERATE CHAPTER TILES VIA GROQ API
        toc_text = full_text[:6000]
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # THE "EXPANDED SUMMARY" PROMPT
        prompt = f"""
        You are an expert educational summarizer. Extract ALL the main chapter titles from the textbook's Table of Contents below.
        
        CRITICAL INSTRUCTIONS:
        1. Find the dedicated "Table of Contents" or "Contents" page.
        2. Extract EVERY main chapter title exactly as written (e.g., I. Title, II. Title).
        3. STRICT STOP: Once you reach the end of the Table of Contents list, STOP. DO NOT extract sub-headings from the body pages.
        4. Ensure there are NO duplicate chapters and the numbering is strictly sequential.
        5. Write a HIGHLY DETAILED, comprehensive educational summary (exactly 2 large paragraphs) for EVERY chapter you extract. Each paragraph MUST be at least 3 to 4 sentences long, explaining the core concepts, historical context, and key takeaways.
        6. Output ONLY a valid JSON object. No markdown formatting (like ```json), no introductory text.
        
        Example JSON format:
        {{
            "I. Exact Chapter Name": [
                "First highly detailed paragraph explaining the core themes, historical context, and background of this chapter (at least 3-4 sentences long)...",
                "Second highly detailed paragraph explaining the key events, impacts, and major takeaways students will learn (at least 3-4 sentences long)..."
            ]
        }}
        
        Textbook Front Matter:
        {toc_text}
        """

        # Call Groq
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2, # Bumped up slightly to give it creative freedom to write longer sentences
        )
        
        # Clean up the response to ensure perfect JSON parsing
        raw_response = response.choices[0].message.content
        cleaned_json = raw_response.strip().replace('```json', '').replace('```', '')
        dynamic_lessons = json.loads(cleaned_json)
        
        return dynamic_lessons

    except Exception as e:
        print(f"Ingestion Error: {e}")
        return {"Error Processing Book": [f"Details: {str(e)}", "Check terminal for more info."]}