# 🔬 Experiment Assistant (MVP v1.0)

> A bilingual RAG assistant for online experimentation topics (A/B testing, SRM, FDR...)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 What is this?

An AI assistant that answers questions about online experimentation using:
- Wikipedia articles (FR/EN) as knowledge base
- FAISS for vector search
- Llama-3.1-8B (via Groq) for answer generation
- Gradio for the UI

## 🚀 Quick Start
```bash
# Clone and setup
git clone <your-repo>
cd experiment-assistant-v2
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the app
python app.py
```

Open http://localhost:7860

## 📚 How it works

1. **Dataset**: 24 curated Wikipedia articles about experimentation
2. **Indexing**: FAISS creates embeddings for semantic search
3. **Retrieval**: Find 3 most relevant passages
4. **Generation**: Llama-3.1 generates answer with citations

## 🔧 Tech Stack

- **LLM**: Llama-3.1-8B-Instant (Groq - free tier)
- **Vector Store**: FAISS
- **Embeddings**: sentence-transformers (multilingual)
- **Framework**: LangChain
- **UI**: Gradio

## 📂 Project Structure
```
experiment-assistant-v2/
├── data/
│   └── build_dataset.py    # Wikipedia curation
├── src/
│   ├── indexer.py          # FAISS indexing
│   └── retriever.py        # Document retrieval
├── app.py                  # Gradio UI
└── requirements.txt
```

## ⚠️ Current Limitations (MVP)

- Limited to 24 Wikipedia articles
- FAISS-only retrieval (no BM25)
- No confidence scores
- No language filter in UI
- No tests yet

## 🗓️ Roadmap

- [ ] v1.1: Add unit tests
- [ ] v1.2: Hybrid retrieval (FAISS + BM25)
- [ ] v1.3: Display confidence scores
- [ ] v2.0: Production-ready (CI/CD, monitoring, advanced UI)

## 📄 License

MIT © 2025