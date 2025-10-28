# 🔬 Experiment Assistant - RAG for Online Experimentation

> Bilingual (FR/EN) RAG assistant for answering questions about online experimentation (A/B testing, SRM, FDR, statistical power...)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/lmhdii/experiment-assistant-dataset)

[🇫🇷 Version française](README.md)

---

## 🎯 Problem

Product and Data Science teams waste time searching for reliable definitions about online experimentation. Sources are scattered (Wikipedia, blogs, internal Slack) and pure LLMs can hallucinate.

**Solution**: A RAG (Retrieval-Augmented Generation) assistant that generates sourced and verifiable answers.

---

## 🏗️ Architecture

```
┌─────────────┐
│   Gradio    │ ← User interface
└──────┬──────┘
       │
┌──────▼────────────┐
│  LangChain        │ ← RAG orchestration
│  RetrievalQA      │
└──────┬────────────┘
       │
   ┌───▼────┐  ┌─────────┐
   │ FAISS  │  │  Groq   │
   │(dense) │  │Llama3.1 │
   └────────┘  └─────────┘
```

---

## 🔧 Tech Stack

- **LLM**: Llama-3.1-8B-Instant (Groq) - fast inference, free tier
- **Vector Store**: FAISS - efficient semantic search
- **Embeddings**: Sentence-Transformers - multilingual support
- **Framework**: LangChain - standard RAG orchestration
- **UI**: Gradio - easy HuggingFace Spaces deployment

**Total cost**: $0

---

## 📚 Dataset

**17 curated Wikipedia articles**:
- **10 EN**: A/B testing, False discovery rate, Power (statistics), Multi-armed bandit, Thompson sampling, Sequential analysis, Sample size determination, Equivalence test, Randomized controlled trial, Scientific control
- **7 FR**: Test A/B, Analyse séquentielle, Puissance statistique, Bandit manchot, Échantillonnage de Thompson, Essai randomisé contrôlé, Groupe de contrôle

📦 **Public dataset**: [lmhdii/experiment-assistant-dataset](https://huggingface.co/datasets/lmhdii/experiment-assistant-dataset)

---

## 🚀 Quick Start

```bash
git clone https://github.com/lmhdii/experiment-checklist-v2.git
cd experiment-checklist-v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add GROQ_API_KEY
python app.py  # Open http://localhost:7860
```

---

## 📊 Performance (MVP v1.0)

| Metric | Target | Current |
|--------|--------|---------|
| Response latency | < 2s | ~1.5s ✅ |
| Source citations | 100% | 100% ✅ |
| Glossary coverage | 90% | ~45% ⚠️ Limited (17 articles) |
| Infrastructure cost | $0 | $0 ✅ |

---

## 🗓️ Roadmap

- [x] **v1.0**: MVP with Gradio + FAISS + citations
- [ ] **v1.1**: Unit tests (pytest)
- [ ] **v1.2**: Hybrid retrieval (FAISS + BM25)
- [ ] **v1.3**: Confidence scores + JSON export
- [ ] **v2.0**: Production-ready (CI/CD, monitoring)

---

## 🎓 Academic Context

Project developed for the **Data Science** course at **DataBird** (2025).

**Subject Adaptation**: Instead of analyzing climate change tweets, I created an assistant aligned with my Product Manager role in experimentation. The project meets all technical requirements (HuggingFace, LLM, Langchain, Gradio) while being directly useful in my daily work.

**Development Approach**: As a PM, I used Claude (Anthropic) as a technical co-pilot for implementation, focusing on architecture, product decisions, and understanding RAG concepts (retrieval, embeddings, generation). This approach reflects the modern use of AI tools for rapid prototyping and developing technical expertise appropriate for the Product Manager role.

---

## 🤝 Contribution

Contributions are welcome! Open an issue or pull request.

**Note**: This project was initially developed as a PM prototype with AI assistance. Developer contributions to improve code quality, optimize performance, or add features are particularly appreciated.

---

## 📄 License

MIT © 2025 El Mehdi BELAHNECH

---

**Built with ❤️ by a PM tired of Googling "what's an SRM"**
