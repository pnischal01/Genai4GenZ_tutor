# Gyaan.AI – Low-Cost AI Tutor for Rural India

## Overview

Gyaan.AI is an intelligent tutoring system designed to make AI-powered education accessible and affordable for students in rural India.

Traditional AI tutors rely on large models with high token usage, leading to increased cost, latency, and bandwidth requirements. This project addresses these challenges by introducing context pruning, which removes irrelevant information before sending data to the language model, significantly reducing cost and improving response speed while maintaining accuracy.

---

## Problem Statement

AI tutors today face several limitations:

- High cost per query
- Dependence on high-speed internet
- Inefficiency when handling large textbooks

These factors make them inaccessible to students in low-resource environments.

---

## Solution

Gyaan.AI:

- Ingests entire state-board textbooks (PDFs)
- Retrieves only relevant content using vector search
- Prunes unnecessary context using the ScaleDown API
- Generates answers using a lightweight local LLM (Ollama)

This results in faster, cheaper, and bandwidth-efficient AI tutoring.

---

## Key Features

- PDF textbook ingestion and processing
- Local FAISS-based semantic search
- Context pruning using ScaleDown API
- Lightweight LLM inference using Ollama (LLaMA 3.2 1B)
- Token usage tracking and savings display
- Multiple answer formats (detailed, short paragraph, bullet points)
- Chapter-wise summaries generation
- Study roadmap generation
- Predicted exam questions generation
- Robust fallback handling for pruning failures

---

## System Architecture

1. Textbook Upload  
   PDF is converted into raw text

2. Chunking and Embedding  
   Text is split into chunks and embeddings are generated using HuggingFace

3. Vector Storage  
   Stored locally using FAISS

4. Query Processing  
   User query retrieves relevant chunks

5. Context Pruning  
   ScaleDown API removes irrelevant text and reduces token usage

6. Answer Generation  
   Pruned context is sent to Ollama to generate the final answer

---
## Workflow Pipeline

PDF Upload
→ Text Extraction
→ Chunking
→ Embeddings
→ FAISS Storage
→ User Query
→ Top-K Retrieval
→ Context Pruning (ScaleDown)
→ LLM Generation (Ollama)
→ Final Answer

---
## Tech Stack

- Frontend: Streamlit
- Backend: Python
- LLM: Ollama (LLaMA 3.2 1B)
- Vector Database: FAISS
- Embeddings: HuggingFace (MiniLM)
- APIs:
  - ScaleDown API (context compression)
  - Groq API (chapter summarization)

---

## Performance Improvements

| Metric           | Baseline RAG | Gyaan.AI |
| ---------------- | ------------ | -------- |
| Tokens per query | ~12,000      | ~4,000   |
| Latency          | ~8.5 sec     | ~2.5 sec |
| Cost per query   | ₹0.56        | ₹0.19    |

Approximately 67% cost reduction achieved through context pruning.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Genai4GenZ_tutor.git
cd Genai4GenZ_tutor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

```bash
Create a .env file in the root directory and add:

GROQ_API_KEY=your_groq_key
SCALEDOWN_API_KEY=your_scaledown_key
```

### 4. Run Ollama

```bash
Ensure Ollama is running locally:

ollama run llama3.2:1b
```

### 5. Run the Application

```bash
streamlit run app.py
```

### Usage

1. Upload a textbook PDF
2. Provide a name for the textbook
3. Ask a question based on the content
4. The system retrieves relevant content, prunes unnecessary context, and generates an answer
5. View token savings and generated response
6. Explore chapter summaries, roadmap, and predicted exam questions

### Key Innovation: Context Pruning

Traditional RAG systems send large chunks of text to the LLM, increasing cost and latency.

Gyaan.AI improves this by:

- Removing irrelevant content

- Reducing token usage

- Improving response time

### Future Improvements

- Mobile-friendly interface

- Personalized learning paths

- Multi-language support

- Offline-first capabilities

### Contributors

1. P. Nischal

2. Samruddhi Kadre
