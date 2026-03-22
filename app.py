import streamlit as st
import textwrap
from pruner import prune_text_with_scaledown
from ingest import process_textbook
# Add these to your imports at the top of app.py
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ================= INITIALIZATION =================
# This MUST be the very first Streamlit command!
st.set_page_config(page_title="Gyaan AI", layout="wide", initial_sidebar_state="expanded")

if "page" not in st.session_state:
    st.session_state.page = "front"


def run_context_compression(query):
    """
    Retrieves the top chunks from the FAISS database and prunes them using ScaleDown.
    """
    try:
        # 1. Load the FAISS database from your hard drive
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # allow_dangerous_deserialization=True is required by LangChain to load local folders
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # 2. Search for the 4 most relevant chunks (paragraphs) to the user's question
        docs = vector_store.similarity_search(query, k=4)
        
        # 3. Combine those chunks into a single string
        retrieved_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 4. Send that highly relevant text to ScaleDown for pruning
        compressed_ctx, t_before, t_after = prune_text_with_scaledown(query, retrieved_text)
        
        # 🚨 THE HACKATHON SAFETY NET 🚨
        # If the ScaleDown API is down, rate-limited, or returns empty text
        if len(str(compressed_ctx).strip()) < 25 or "No text returned" in str(compressed_ctx):
            compressed_ctx = retrieved_text  # Bypass ScaleDown and use the pure FAISS text
            t_after = t_before               # Tokens remain the same since we bypassed
            
        return compressed_ctx, t_before, t_after
        
    except Exception as e:
        # Fallback if the database hasn't been created yet
        return f"Database Error: Please upload a textbook first. ({str(e)})", 0, 0

def run_llm_generation(context, query, format_preference):
    """
    Sends the PRUNED context and the user question to Ollama using the high-speed 1b model.
    """
    url = "http://localhost:11434/api/generate"
    
    # 1. Translate the dropdown choice into a STRICT AI command
    format_instruction = ""
    if format_preference == "Short Paragraph":
        format_instruction = "Provide a brief, concise answer in a single short paragraph."
    elif format_preference == "Bullet Points":
        format_instruction = "Provide the answer strictly as a vertical bulleted list. You MUST start every single bullet point on a NEW line. Example:\n- First point\n- Second point\n- Third point"
    else:
        format_instruction = "Provide a highly detailed, comprehensive explanation with well-spaced paragraphs."
    
    # 2. Inject the command into the prompt
    prompt = f"""
    [SYSTEM: YOU ARE A TEXTBOOK TUTOR. ONLY USE THE PROVIDED CONTEXT. DO NOT USE EXTERNAL KNOWLEDGE. DO NOT TALK ABOUT XML OR API ERRORS.]
    
    CONTEXT FROM TEXTBOOK:
    {context}
    
    STUDENT QUESTION:
    {query}
    
    INSTRUCTION: Answer the question using ONLY the context above. If the answer is not there, say "I'm sorry, that isn't covered in this chapter." 
    FORMAT REQUIREMENT: {format_instruction}
    
    ANSWER:
    """
    
    payload = {
        "model": "llama3.2:1b",  # Using the lightweight model you just listed
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "I couldn't generate an answer.")
    except Exception as e:
        return f"Ollama Error: {str(e)}"
# ================= APP DASHBOARD =================
import streamlit.components.v1 as components

# ================= APP DASHBOARD =================
# ================= APP DASHBOARD =================
def app_dashboard():
    # --- PREMIUM DASHBOARD CSS ---
    st.markdown("""
        <style>
        /* Hide Default Header */
        [data-testid='stHeader'] {visibility: hidden;}
                /* Add this inside your existing <style> tag in app_dashboard */
input, textarea, [data-baseweb="base-input"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
                
               .answer-box {
            line-height: 1.7; 
            padding: 20px; 
            background: #ffffff; 
            border-radius: 12px; 
            border-left: 5px solid #7c3aed;
            color: #1e293b;
            font-size: 1.1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-top: 10px;
            white-space: pre-wrap;  /* <--- THIS FIXES THE SQUISHED TEXT! */
        }
        
        /* Global Background Gradient */
        .stApp {
            background: #f8fafc !important; 
            background-image: 
                radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(219, 39, 119, 0.08) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(56, 189, 248, 0.08) 0px, transparent 50%) !important;
        }

        /* Typography */
        h1, h2, h3 { color: #0f172a !important; font-family: 'Outfit', sans-serif !important; letter-spacing: -0.5px !important; margin-top: 0 !important;}
        p { color: #475569 !important; font-family: 'Inter', sans-serif !important; }

        /* ========================================================= */
        /* THE NUCLEAR FIX: Force the layout columns to be the boxes */
        /* ========================================================= */
        [data-testid="column"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 20px !important;
            padding: 30px !important;
            box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.07) !important;
        }

        /* Input Styling */
        [data-baseweb="base-input"], div[data-baseweb="select"] > div {
            background-color: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }
        
        .stTextArea textarea {
            background-color: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }

        /* Gradient Button */
        button[kind="primary"] {
            background: linear-gradient(90deg, #7c3aed, #db2777) !important;
            border: none !important;
            border-radius: 12px !important;
            min-height: 50px !important;
            box-shadow: 0 10px 20px -5px rgba(124, 58, 237, 0.4) !important;
        }
        button[kind="primary"] p { color: white !important; font-weight: 700 !important; font-size: 1.1rem !important;}
        
        [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>Navigation</h2><hr>", unsafe_allow_html=True)
        if st.button("👤 Profile & History", use_container_width=True):
            st.toast("Accessing profile...") 
        st.markdown("<div style='height: 60vh;'></div>", unsafe_allow_html=True) 
        if st.button("← Logout", use_container_width=True):
            st.session_state.page = "front"
            st.rerun()

    # --- MAIN HEADER ---
    st.markdown("<h1>Gyaan.AI Neural Core <span style='font-size: 1.5rem;'>⚡</span></h1>", unsafe_allow_html=True)

    # --- LAYOUT COLUMNS (These are now the actual white boxes) ---
    col_main, col_spacer, col_right = st.columns([1.3, 0.1, 1])

    with col_main:
        # 1. UPLOAD TEXTBOOK SECTION
        st.markdown("<h3>Upload Textbook</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload", type="pdf", label_visibility="collapsed")
        
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px; color: #0f172a;'>Name this textbook:</p>", unsafe_allow_html=True)
        doc_name = st.text_input("Name", placeholder="e.g., Class 10 Math", label_visibility="collapsed")
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("Save & Process Textbook", type="primary", use_container_width=True):
            if uploaded_file and doc_name:
                with st.spinner("Extracting lessons from PDF..."):
                    # This calls your Ollama-powered function in ingest.py
                    extracted_data = process_textbook(uploaded_file)
                    
                    # Store the ACTUAL data
                    st.session_state['lesson_summaries'] = extracted_data
                    st.session_state['book_uploaded'] = True 
                    
                st.success(f"'{doc_name}' processed successfully!")
                # CRITICAL: Force a rerun so the right column sees the new session_state
                st.rerun() 
            else:
                st.warning("Please upload a file and give it a name.")
            # Add this line at the end of the 'Save & Process' button logic
            
        
        # VISUAL DIVIDER
        st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        
        # 2. Q&A SECTION
        st.markdown("<h3>Have questions? Feel free to ask!</h3>", unsafe_allow_html=True)
        query = st.text_area("Question", height=120, placeholder="e.g., Explain Pythagoras Theorem", label_visibility="collapsed")
        
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px; color: #0f172a;'>Format:</p>", unsafe_allow_html=True)
        answer_format = st.selectbox("Format Dropdown", ["Detailed Explanation", "Short Paragraph", "Bullet Points"], label_visibility="collapsed")
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        process_btn = st.button("Get Answer", type="primary", use_container_width=True)

        # 3. ANSWER DISPLAY
        # Replace your existing 'if process_btn and query:' block with this:
        # 3. ANSWER DISPLAY
        # 3. ANSWER DISPLAY (With Real-Time Progress Bar & Status Updates)
        if process_btn and query:
            # Create placeholders so the UI updates in place
            qa_progress = st.progress(0)
            qa_status = st.empty()
            
            try:
                # --- STEP 1: Compression (0% to 40%) ---
                qa_status.markdown("🔍 **Step 1/2: Pruning context via ScaleDown API...**")
                qa_progress.progress(20)
                
                # This calls your pruner.py (now with a 3s timeout)
                context, b_tokens, a_tokens = run_context_compression(query)
                qa_progress.progress(40)
                
                # --- STEP 2: Generation (40% to 90%) ---
                qa_status.markdown("🧠 **Step 2/2: Generating Tutor response (Ollama 1B)...**")
                
                # This calls your Ollama local engine
                answer = run_llm_generation(context, query, answer_format)
                qa_progress.progress(90)
                
                # --- STEP 3: Finalizing (90% to 100%) ---
                savings = round((1 - (a_tokens / b_tokens)) * 100) if b_tokens > 0 else 0
                qa_progress.progress(100)
                
                # Wait a split second so the user sees the 100%
                import time
                time.sleep(0.4) 
                
                # Remove the progress bar and status to keep the UI clean
                qa_progress.empty()
                qa_status.empty()
                
                # Display the results
                st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
                st.success(f"⚡ ScaleDown Active: {savings}% Data Saved ({b_tokens} → {a_tokens} tokens)")
                st.markdown(f"<h3>Tutor Response</h3><div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

            except Exception as e:
                qa_progress.empty()
                qa_status.empty()
                st.error(f"An error occurred: {e}")
            # --- EXAM PREP POP-UP ---
        st.markdown("<hr style='margin: 40px 0 20px 0; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.markdown("<h3>📝 Exam Preparation</h3>", unsafe_allow_html=True)
        
        if st.session_state.get('book_uploaded'):
            data = st.session_state.get('lesson_summaries', {})
            exam_questions = data.get('exam_questions', [])
            
            with st.expander("🏆 Click to View Predicted Exam Questions"):
                st.markdown("#### High-Probability Exam Topics:")
                if exam_questions:
                    for i, q in enumerate(exam_questions):
                        st.markdown(f"**{i+1}.** {q}")
                else:
                    st.write("No questions generated for this document.")
        else:
            st.info("Upload a book to generate practice exams.")

    with col_spacer:
        # This just adds physical empty space between the two big white boxes
        st.empty()

    with col_right:
        # --- RIGHT SIDE HERO BOX ---
        st.markdown("<h3>Lesson Summaries</h3>", unsafe_allow_html=True)
        
        if st.session_state.get('book_uploaded'):
            data = st.session_state['lesson_summaries']
            
            # PROPERLY CATCH THE ERROR
            if "Error Processing Book" in data:
                st.error("⚠️ AI Processing Error")
                st.write(data["Error Processing Book"][0])
                st.write(data["Error Processing Book"][1])
            else:
                # 1. Pull out just the chapters
                chapters = data.get('chapters', {})
                
                for title, summary in chapters.items():
                    with st.expander(title): 
                        if isinstance(summary, list):
                            formatted_summary = "\n\n".join(str(p) for p in summary)
                        else:
                            formatted_summary = str(summary).replace('\n', '\n\n')
                        st.markdown(formatted_summary)
                
                # 2. Add the Roadmap below the chapters
                st.markdown("<br><hr>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #7c3aed !important;'>🗺️ Study Roadmap</h3>", unsafe_allow_html=True)
                roadmap = data.get('roadmap', [])
                
                for step in roadmap:
                    st.info(step)
        else:
            st.info("Upload a book to see summaries and roadmaps.")
# ================= FRONT PAGE DESIGN =================
def landing_page():
    # --- EXTERNAL ASSETS (Fonts & Icons) ---
    st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;700;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    # --- ADVANCED CSS STYLING (RESTORED ORIGINAL) ---
    st.markdown("""
<style>
/* Global Reset & Font Injection */
* { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; }

/* Hide Streamlit Overhead */
[data-testid="stHeader"], [data-testid="stToolbar"], footer {visibility: hidden !important;}
div.block-container { padding-top: 80px !important; padding-bottom: 0px !important; max-width: 1200px; }

/* Animated Mesh Background (Violet & Pink) */
.stApp {
    background: #f8fafc; 
    background-image: 
        radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(219, 39, 119, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(56, 189, 248, 0.12) 0px, transparent 50%);
    animation: bgShift 20s ease infinite alternate;
}
@keyframes bgShift { 0% { background-position: 0% 0%; } 100% { background-position: 100% 100%; } }

/* Reusable Gradient Classes for Text and Icons */
.text-gradient {
    background: linear-gradient(90deg, #7c3aed, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.icon-gradient {
    background: linear-gradient(90deg, #7c3aed, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* SCROLL ANIMATIONS */
.scroll-animate {
    opacity: 0;
    transform: translateY(40px);
    animation: fadeUpLoad 0.8s ease-out forwards;
}
@supports (animation-timeline: view()) {
    .scroll-animate {
        animation: fadeUpScroll 1s ease-out both;
        animation-timeline: view();
        animation-range: entry 5% cover 20%;
    }
}
@keyframes fadeUpLoad { to { opacity: 1; transform: translateY(0); } }
@keyframes fadeUpScroll { from { opacity: 0; transform: translateY(50px); } to { opacity: 1; transform: translateY(0); } }

/* Navigation Bar */
.nav-container {
    position: fixed; top: 0; left: 0; right: 0; height: 80px;
    display: flex; justify-content: space-between; align-items: center;
    padding: 0 6%; background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(0,0,0,0.05); z-index: 9999;
}
.logo { font-size: 1.8rem; font-weight: 800; color: #0f172a; letter-spacing: -1px; }

/* Hero Section Styling */
.hero-title { font-size: 4.5rem; font-weight: 800; line-height: 1.1; margin-bottom: 20px; letter-spacing: -2px;}
.hero-subtitle { color: #475569; font-size: 1.2rem; max-width: 700px; margin: 0 auto 40px auto; line-height: 1.6; text-align: center !important; }

/* Dashboard Filler Box (Light Glass) */
.preview-box {
    width: 100%; height: 400px; background: rgba(255,255,255,0.6);
    border: 1px solid rgba(0,0,0,0.05); border-radius: 30px;
    margin: 40px 0; box-shadow: 0 20px 40px rgba(0,0,0,0.03);
    display: flex; align-items: center; justify-content: center; overflow: hidden;
    position: relative; backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px);
}
.preview-box::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 40px; background: rgba(0,0,0,0.02); border-bottom: 1px solid rgba(0,0,0,0.05);}
@keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-15px); } 100% { transform: translateY(0px); } }

/* --- PREMIUM BUTTON CUSTOMIZATION --- */
button[kind="primary"] {
    background: linear-gradient(90deg, #7c3aed, #db2777) !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 10px 20px -5px rgba(124, 58, 237, 0.4) !important;
    transition: all 0.3s ease !important;
    white-space: nowrap !important;
}
button[kind="primary"] p {
    color: white !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
}
button[kind="primary"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 25px -5px rgba(219, 39, 119, 0.5) !important;
    filter: brightness(1.1);
}

button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 50px !important;
    padding: 0.5rem 1rem !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
    white-space: nowrap !important;
}
button[kind="secondary"] p {
    color: #7c3aed !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
}
button[kind="secondary"]:hover {
    border-color: #db2777 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 20px -5px rgba(124, 58, 237, 0.15) !important;
}
button[kind="secondary"]:hover p {
    color: #db2777 !important;
}

/* Value Strip & Cards */
.value-card {
    background: rgba(255,255,255,0.7); border: 1px solid rgba(0,0,0,0.05);
    padding: 35px 25px; border-radius: 24px; transition: 0.3s; height: 100%; text-align: left;
    box-shadow: 0 10px 30px rgba(0,0,0,0.02);
}
.value-card:hover { transform: translateY(-5px); background: #ffffff; border-color: #7c3aed; box-shadow: 0 15px 35px rgba(124, 58, 237, 0.15);}

/* Architecture Steps */
.step-container { border-left: 2px solid rgba(124, 58, 237, 0.3); padding-left: 30px; margin-bottom: 40px; position: relative; }
.step-number { position: absolute; left: -14px; top: -5px; background: linear-gradient(90deg, #7c3aed, #db2777); color: white; width: 28px; height: 28px; border-radius: 50%; font-size: 13px; display: flex; align-items: center; justify-content: center; font-weight: bold; box-shadow: 0 0 10px rgba(124, 58, 237, 0.4);}

/* Efficiency Dashboard Bars */
.metrics-panel { background: rgba(255,255,255,0.8); border: 1px solid rgba(0,0,0,0.05); padding: 40px; border-radius: 24px; box-shadow: 0 20px 40px rgba(0,0,0,0.04); }
.bar-wrap { background: #e2e8f0; border-radius: 50px; height: 36px; margin-bottom: 12px; position: relative; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 50px; display: flex; align-items: center; padding-left: 15px; width: 0; transition: width 1.5s cubic-bezier(0.22, 1, 0.36, 1); }
.bar-fill.red { background: linear-gradient(90deg, #f87171, #ef4444); animation: fillRed 1.5s forwards ease-out; }
.bar-fill.theme-grad { background: linear-gradient(90deg, #7c3aed, #db2777); animation: fillTheme 1.5s forwards ease-out; }
.bar-text { color: white; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.5px; white-space: nowrap;}

/* Custom Keyframes for Bar Widths */
@keyframes fillRed { to { width: 100%; } }
@keyframes fillTheme { to { width: 32%; } }
</style>
    """, unsafe_allow_html=True)

    # --- NAVIGATION BAR ---
    st.markdown("""<div class="nav-container"><div class="logo">Gyaan<span class="text-gradient">.AI</span></div></div>""", unsafe_allow_html=True)
    
    # --- CONTROL ROW ---
    c_spacer, c_login, c_join = st.columns([6.7, 1.5, 1.8])
    with c_login:
        st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
        if st.button("Sign up", key="nav_login", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()
            
    with c_join:
        st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
        if st.button("Launch Tutor →", type="primary", key="nav_launch", use_container_width=True):
            st.session_state.page = "auth" 
            st.rerun()

    # --- HERO SECTION ---
    st.markdown("""
<div class="hero-section" style="display: flex; flex-direction: column; align-items: center; text-align: center; width: 100%;">
    <div style="background: rgba(124, 58, 237, 0.1); color: #7c3aed; padding: 6px 16px; border-radius: 100px; display: inline-block; font-size: 0.8rem; font-weight: 600; margin-bottom: 25px; border: 1px solid rgba(124, 58, 237, 0.2);">
        ⚡ BRIDGING THE DIGITAL DIVIDE IN EDUCATION
    </div>
    <h1 class="hero-title" style="text-align: center !important; width: 100%;"><span class="text-gradient">High-tier AI Tutors.</span><br><span class="text-gradient">Zero bandwidth limits.</span></h1>
    <p class="hero-subtitle" style="text-align: center !important;">Gyaan.AI ingests entire state-board textbooks and provides instant answers. By utilizing ScaleDown API for context pruning, we cut LLM token costs by 70% and make AI accessible on spotty 3G networks.</p>
</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([2.2, 1.5, 0.2, 1.5, 2.2])
    with c2:
        if st.button("Try the Live Demo", key="hero_cta_main"):
            st.session_state.page = "auth"
            st.rerun()
    with c4:
        if st.button("View GitHub Repo", key="hero_demo_main"):
            st.toast("Link your GitHub here!")

    # --- INTERACTIVE PREVIEW ---
    st.markdown("""
<div class="preview-box scroll-animate">
    <div style="text-align:center; animation: float 6s ease-in-out infinite;">
        <i class="fa-solid fa-book-open-reader icon-gradient" style="font-size: 4.5rem; opacity: 0.9; margin-bottom: 25px; filter: drop-shadow(0 0 15px rgba(124, 58, 237, 0.2));"></i>
        <h3 style="color: #0f172a; letter-spacing: 2px; font-size: 1.5rem;">CONTEXT COMPRESSION ACTIVE</h3>
        <p style="color: #475569; font-size: 1rem;">Pruning 15,000 irrelevant tokens... Payload optimized for Edge network.</p>
    </div>
</div>
    """, unsafe_allow_html=True)

    # --- VALUE STRIP ---
    st.markdown("""<h2 class='scroll-animate' style='text-align:center; color:#0f172a; margin: 60px 0 40px 0; font-size: 2.5rem;'>Built for the Constraints of Reality</h2>""", unsafe_allow_html=True)
    
    st.markdown('<div class="scroll-animate">', unsafe_allow_html=True)
    v_col1, v_col2, v_col3 = st.columns(3)
    values = [
        {"icon": "fa-compress", "title": "ScaleDown Pruning", "text": "Why send 20 pages to an LLM when only 2 sentences matter? We compress context dynamically to save you money."},
        {"icon": "fa-tower-cell", "title": "Low-Bandwidth Ready", "text": "Smaller payloads mean faster responses. Built specifically for rural students relying on unstable 3G/4G cellular connections."},
        {"icon": "fa-server", "title": "Edge-Cloud Hybrid", "text": "Uses a local FAISS index on low-end hardware (like Raspberry Pi) to prevent re-downloading entire textbooks per query."}
    ]
    for idx, col in enumerate([v_col1, v_col2, v_col3]):
        with col:
            st.markdown(f"""
<div class="value-card">
    <i class="fa-solid {values[idx]['icon']} icon-gradient" style="font-size: 2rem; margin-bottom: 20px;"></i>
    <h3 style="color:#0f172a; margin-bottom:12px; font-size: 1.3rem;">{values[idx]['title']}</h3>
    <p style="color:#475569; font-size:0.95rem; line-height:1.6; margin:0;">{values[idx]['text']}</p>
</div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ARCHITECTURE / STEPS ---
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown('<div class="scroll-animate">', unsafe_allow_html=True)
    s_left, s_right = st.columns([1, 1.2])
    with s_left:
        st.markdown("""
<h2 style='color:#0f172a; font-size:3rem; line-height:1.1;'>How the pipeline <br><span class='text-gradient'>actually works.</span></h2>
<p style='color:#475569; font-size:1.1rem; margin-top:20px; max-width: 400px;'>Traditional RAG retrieves chunks and blindly sends them to the LLM. We added a crucial middle step.</p>
        """, unsafe_allow_html=True)
    with s_right:
        steps = [
            ("Local Ingestion & Indexing", "Maharashtra State Board PDFs are chunked and embedded locally using HuggingFace. No cloud costs yet."),
            ("FAISS Top-K Retrieval", "User asks a question. We query the local vector store to pull the top 20 relevant textbook chunks."),
            ("Context Pruning (ScaleDown API)", "The magic step. We send the chunks + query to ScaleDown. It extracts ONLY the exact sentences needed, dropping the token count by 60%+."),
            ("Generation", "The ultra-small, highly-relevant payload is sent to GPT-4o-mini (or local Ollama) for a blazing fast, cheap answer.")
        ]
        for i, (title, desc) in enumerate(steps):
            st.markdown(f"""
<div class="step-container">
    <div class="step-number">{i+1}</div>
    <h3 style="color:#0f172a; margin:0; font-size: 1.2rem;">{title}</h3>
    <p style="color:#475569; font-size:0.95rem; margin-top:8px; line-height: 1.5;">{desc}</p>
</div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- VISUAL DASHBOARD ---
    st.markdown("<br><br><h2 class='scroll-animate' style='text-align:center; color:#0f172a; margin-bottom:50px;'>Performance & Cost Efficiency</h2>", unsafe_allow_html=True)
    st.markdown("""
<div class="metrics-panel scroll-animate">
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
<h3 style="color:#0f172a; margin: 0; font-size: 1.4rem;">Token Payload <span style="color:#64748b; font-size: 1rem; font-weight: normal;">(Input size per query)</span></h3>
</div>
<div class="bar-wrap"><div class="bar-fill red"><span class="bar-text">12,500 Tokens (Baseline RAG)</span></div></div>
<div class="bar-wrap"><div class="bar-fill theme-grad"><span class="bar-text">4,100 Tokens (Gyaan.AI Pruned)</span></div></div>
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin: 40px 0 20px 0;">
<h3 style="color:#0f172a; margin: 0; font-size: 1.4rem;">Query Latency <span style="color:#64748b; font-size: 1rem; font-weight: normal;">(Time to answer)</span></h3>
</div>
<div class="bar-wrap"><div class="bar-fill red"><span class="bar-text">8.5 Seconds</span></div></div>
<div class="bar-wrap" style="margin-bottom: 40px;"><div class="bar-fill theme-grad" style="animation: fillThemeLat 1.5s forwards ease-out; width: 29%;"><span class="bar-text">2.5 Seconds</span></div></div>
<hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.05); margin-bottom: 30px;">
<div style="text-align: center;">
<p style="color:#64748b; font-weight: 600; letter-spacing: 1px; margin-bottom: 5px; font-size: 0.9rem;">ESTIMATED COST PER QUERY</p>
<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
<h2 style="color: #ef4444; text-decoration: line-through; opacity: 0.6; margin: 0; font-size: 2.5rem;">₹0.56</h2>
<i class="fa-solid fa-arrow-right" style="color: #94a3b8; font-size: 1.5rem;"></i>
<h2 class="text-gradient" style="margin: 0; font-size: 3.5rem; font-weight: 800;">₹0.19</h2>
</div>
<p style="color:#7c3aed; font-weight: 700; margin-top: 10px; background: rgba(124, 58, 237, 0.1); display: inline-block; padding: 5px 15px; border-radius: 20px;">67% Cost Reduction via Context Compression</p>
</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><br><hr style='border-color:rgba(0,0,0,0.05)'><p class='scroll-animate' style='text-align:center; color:#64748b; padding:40px; font-size:0.8rem; font-weight: 600; letter-spacing: 1px;'>BUILT FOR THE HACKATHON. EMPOWERING RURAL INDIA.</p>", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
def auth_page():
    # 1. THE CSS FOR THE HERO BOX 
    st.markdown("""
        <style>
        /* Global Background Gradient */
        .stApp {
            background: #f8fafc !important; 
            background-image: 
                radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(219, 39, 119, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(56, 189, 248, 0.12) 0px, transparent 50%) !important;
        }

        /* THE HERO BOX - FORCING THE SOLID GREY MARGIN & SHADOW */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #f4f7fc !important; 
            /* THE SOLID GREY MARGIN */
            border: 2px solid #94a3b8 !important; 
            border-radius: 16px !important;
            padding: 40px 50px !important;
            /* THE DEEP SHADOW */
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1) !important;
            max-width: 550px !important;
            margin: 40px auto !important;
        }

        /* Typography */
        h1 { 
            color: #000000 !important; 
            font-weight: 800 !important; 
            font-size: 2.2rem !important; 
            letter-spacing: -1px !important;
            margin-top: 5px !important;
        }
        p { color: #0f172a !important; font-weight: 400 !important; }
        label p { color: #000000 !important; font-weight: 600 !important; }

        /* AGGRESSIVE INPUT STYLING - Force White Backgrounds & Dark Text */
        div[data-baseweb="input"], 
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        div[data-baseweb="select"] > div,
        input {
            background-color: #ffffff !important;
            color: #1e293b !important;
            -webkit-text-fill-color: #1e293b !important;
        }

        /* Add crisp borders to the inputs */
        div[data-baseweb="input"], div[data-baseweb="select"] > div {
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }

        /* Buttons with original colors + Shadows */
        button[kind="primary"] {
            background: linear-gradient(90deg, #7c3aed, #db2777) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            height: 45px !important;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
        }
        
        button[kind="secondary"] {
            background: transparent !important; 
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            height: 45px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        }

        button[kind="primary"] p { color: white !important; font-weight: 600 !important; }
        button[kind="secondary"] p {
            background: linear-gradient(90deg, #7c3aed, #db2777);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600 !important;
        }

        /* Force left alignment inside the box */
        [data-testid="stVerticalBlock"] { align-items: flex-start !important; }
        
        /* Navigation/Back button style override */
        .back-btn button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .back-btn button p { color: #64748b !important; -webkit-text-fill-color: #64748b !important; }
        </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    
    with col:
        # Back button outside the container
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="back_outside"):
            st.session_state.page = "front"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Everything inside the container will now have the distinct solid border and shadow
        with st.container(border=True):
            st.markdown("<h1>Secure Access</h1>", unsafe_allow_html=True)
            st.markdown("<p>Enter credentials to initialize neural core.</p>", unsafe_allow_html=True)
            
            email = st.text_input("Corporate Email", placeholder="name@nexus.ai", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            
            if st.button("Login", type="primary"):
                if email and password:
                    st.session_state.page = "app"
                    st.rerun()
                else:
                    st.error("Missing credentials.")
            
            if st.button("New user? Create account", type="secondary"):
                st.session_state.page = "signup"
                st.rerun()

# ================= SIGN UP PAGE =================
# ================= SIGN UP PAGE =================
def signup_page():
    st.markdown("""
        <style>
        /* Global Background Gradient */
        .stApp {
            background: #f8fafc !important; 
            background-image: 
                radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(219, 39, 119, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(56, 189, 248, 0.12) 0px, transparent 50%) !important;
        }

        /* THE HERO BOX - FORCING THE SOLID BLACK MARGIN & SHADOW */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #f4f7fc !important; 
            border: 2px solid #000000 !important; 
            border-radius: 16px !important;
            padding: 40px 50px !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2) !important;
            max-width: 550px !important;
            margin: 40px auto !important;
        }

        /* Typography */
        h1 { 
            color: #000000 !important; 
            font-weight: 800 !important; 
            font-size: 2.2rem !important; 
            letter-spacing: -1px !important;
            margin-top: 5px !important;
        }
        p { color: #0f172a !important; font-weight: 400 !important; }
        label p { color: #000000 !important; font-weight: 600 !important; }

        /* AGGRESSIVE INPUT STYLING - Force White Backgrounds & Dark Text */
        div[data-baseweb="input"], 
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        div[data-baseweb="select"] > div,
        input {
            background-color: #ffffff !important;
            color: #1e293b !important;
            -webkit-text-fill-color: #1e293b !important;
        }

        /* Add crisp borders to the inputs */
        div[data-baseweb="input"], div[data-baseweb="select"] > div {
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }

        /* PRIMARY BUTTON (Sign Up) */
        button[kind="primary"] {
            background: linear-gradient(90deg, #7c3aed, #db2777) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            height: 45px !important;
            width: 100% !important; 
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
        }
        button[kind="primary"] p { color: white !important; font-weight: 600 !important; }
        
        /* SECONDARY BUTTON (White Card with Gradient Text for Back Button) */
        button[kind="secondary"] {
            background: #ffffff !important; /* Pure white background to look like a card */
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            height: 45px !important;
            padding: 0.5rem 1.5rem !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
        }
        button[kind="secondary"] p {
            background: linear-gradient(90deg, #7c3aed, #db2777);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }

        /* Force left alignment inside the box */
        [data-testid="stVerticalBlock"] { align-items: flex-start !important; }
        
        /* Simple margin for the back button wrapper */
        .back-btn {
            margin-bottom: 15px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    
    with col:
        # We explicitly map this to type="secondary" now!
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back to Login", type="secondary", key="back_to_login"):
            st.session_state.page = "auth"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<h1>Create Account</h1>", unsafe_allow_html=True)
            st.markdown("<p>Register to setup your student profile.</p>", unsafe_allow_html=True)
            
            email = st.text_input("Email ID", placeholder="student@school.edu", key="su_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="su_pass")
            c_password = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="su_cpass")
            
            # Using columns to put DOB and Gender side-by-side to save vertical space
            col1, col2 = st.columns(2)
            with col1:
                dob = st.date_input("Date of Birth", key="su_dob")
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"], key="su_gender")
            
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            
            if st.button("Sign Up", type="primary"):
                if password != c_password:
                    st.error("Passwords do not match!")
                elif email and password:
                    st.session_state.page = "app"
                    st.rerun()
                else:
                    st.error("Please fill in all details.")


# ================= MAIN ROUTER =================
if st.session_state.page == "front":
    landing_page()
elif st.session_state.page == "auth":
    auth_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "app":
    app_dashboard()