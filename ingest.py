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
        
        # 3. GENERATE CHAPTERS, ROADMAP, AND QUESTIONS
        toc_text = full_text[:6000]
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""
        Extract the main chapter titles and create summaries for this textbook.
        
        STRICT RULES:
        1. Only extract main titles from the Table of Contents.
        2. Write ONE dense paragraph (4 sentences) per chapter.
        3. Create a 4-step 'roadmap' and 5 'exam_questions'.
        4. Output ONLY valid JSON. No conversational text. No markdown.
        
        JSON STRUCTURE:
        {{
            "chapters": {{ "Title": ["Summary paragraph"] }},
            "roadmap": ["Step 1", "Step 2", "Step 3", "Step 4"],
            "exam_questions": ["Q1", "Q2", "Q3", "Q4", "Q5"]
        }}
        
        Textbook Content:
        {toc_text}
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=4000
        )
        
        raw_content = response.choices[0].message.content
        
        # 🚨 THE NUCLEAR CLEANER: Rips out anything that isn't the JSON block
        try:
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}') + 1
            json_str = raw_content[start_idx:end_idx]
            dynamic_lessons = json.loads(json_str)
        except Exception:
            # Fallback if cleaning fails
            cleaned_json = raw_content.strip().replace('```json', '').replace('```', '')
            dynamic_lessons = json.loads(cleaned_json)
        
        return dynamic_lessons

    except Exception as e:
        print(f"Ingestion Error: {e}")
        return {"Error Processing Book": [f"Details: {str(e)}", "Please try a different section or shorter name."]}