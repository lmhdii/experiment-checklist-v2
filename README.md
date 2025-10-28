# 🔬 Experiment Assistant - Assistant RAG pour l'Expérimentation

> Assistant bilingue (FR/EN) basé sur RAG pour répondre aux questions sur l'expérimentation en ligne (A/B testing, SRM, FDR, puissance statistique...)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/lmhdii/experiment-assistant-dataset)

[🇬🇧 English version](README_EN.md)

---

## 🎯 Problématique

Les équipes Product et Data Science perdent du temps à chercher des définitions fiables sur l'expérimentation en ligne. Les sources sont dispersées (Wikipedia, blogs, Slack interne) et les LLM purs peuvent halluciner.

**Solution** : Un assistant RAG (Retrieval-Augmented Generation) qui génère des réponses sourcées et vérifiables.

---

## 🔍 Pourquoi RAG ? Vérifiabilité avant Confiance Aveugle

> *"À défaut d'avoir une IA fiable, on veut une IA vérifiable"*  
> — JIMINI AI (legal tech française)

Ce principe s'applique à l'expérimentation :
- **Décisions data-driven** → besoin de sources traçables
- **Compliance** (RGPD, légal) → recommandations auditables  
- **Montée en compétence** → les juniors peuvent vérifier les sources

**RAG = Retrieval + Generation** garantit que chaque réponse est vérifiable via des citations Wikipedia.

---

## 🏗️ Architecture

```
┌─────────────┐
│   Gradio    │ ← Interface utilisateur
└──────┬──────┘
       │
┌──────▼────────────┐
│  LangChain        │ ← Orchestration RAG
│  RetrievalQA      │
└──────┬────────────┘
       │
   ┌───▼────┐  ┌─────────┐
   │ FAISS  │  │  Groq   │
   │(dense) │  │Llama3.1 │
   └────────┘  └─────────┘
```

**Flux** :
1. Question utilisateur → Recherche sémantique FAISS
2. Top-3 passages Wikipedia → Contexte pour le LLM
3. Llama-3.1 génère réponse + citations cliquables

---

## 🔧 Stack Technique

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| **LLM** | Llama-3.1-8B-Instant (Groq) | Inférence rapide, tier gratuit (30k tokens/h) |
| **Framework** | LangChain | Orchestration RAG standard |
| **Vector Store** | FAISS | Recherche sémantique efficace, pas de serveur |
| **Embeddings** | Sentence-Transformers | Support multilingue (FR/EN) |
| **UI** | Gradio | Déploiement HuggingFace Spaces simplifié |
| **Hébergement** | HuggingFace Space | Tier gratuit CPU |

**Coût total** : 0€

---

## 📚 Dataset

**17 articles Wikipedia** sélectionnés manuellement :
- **10 EN** : A/B testing, False discovery rate, Power (statistics), Multi-armed bandit, Thompson sampling, Sequential analysis, Sample size determination, Equivalence test, Randomized controlled trial, Scientific control
- **7 FR** : Test A/B, Analyse séquentielle, Puissance statistique, Bandit manchot, Échantillonnage de Thompson, Essai randomisé contrôlé, Groupe de contrôle

📦 **Dataset public** : [lmhdii/experiment-assistant-dataset](https://huggingface.co/datasets/lmhdii/experiment-assistant-dataset)

---

## 🚀 Installation Locale

```bash
# 1. Cloner le projet
git clone https://github.com/lmhdii/experiment-checklist-v2.git
cd experiment-checklist-v2

# 2. Créer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les clés API
cp .env.example .env
nano .env  # Ajouter GROQ_API_KEY (gratuit sur console.groq.com)

# 5. Lancer l'application
python app.py
```

Ouvrir http://localhost:7860

---

## 🧪 Exemples de Requêtes

| Question | Réponse attendue | Sources citées |
|----------|------------------|----------------|
| "Qu'est-ce qu'un SRM ?" | Définition du Sample Ratio Mismatch | A/B testing (EN), Test A/B (FR) |
| "Différence entre interleaving et A/B ?" | Comparaison des méthodes | Interleaving (EN), A/B testing (EN) |
| "Comment calculer la puissance statistique ?" | Formule + interprétation | Power (statistics) (EN), Puissance statistique (FR) |

---

## 📂 Structure du Projet

```
experiment-checklist-v2/
├── data/
│   ├── __init__.py
│   └── build_dataset.py       # Curation Wikipedia (17 articles)
├── src/
│   ├── __init__.py
│   ├── indexer.py             # Création index FAISS
│   └── retriever.py           # Recherche de documents
├── faiss_index/               # Index vectoriel (Git LFS)
│   ├── index.faiss
│   └── index.pkl
├── app.py                     # Interface Gradio
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 📊 Métriques (MVP v1.0)

| Métrique | Cible | Actuel | Statut |
|----------|-------|--------|--------|
| Latence réponse | < 2s | ~1.5s | ✅ |
| Citations systématiques | 100% | 100% | ✅ |
| Couverture glossaire | 90% | ~45% | ⚠️ Limité (17 articles) |
| Coût infrastructure | 0€ | 0€ | ✅ |

---

## ⚠️ Limitations Actuelles (MVP)

- Corpus limité à 17 articles Wikipedia
- Recherche vectorielle FAISS uniquement (pas de BM25)
- Pas de score de confiance affiché
- Pas de filtre de langue dans l'UI
- Évaluation qualitative uniquement (pas de métriques automatiques)

---

## 🗓️ Roadmap

| Version | Objectif | Livrable | Date cible |
|---------|----------|----------|------------|
| **v1.0** ✅ | MVP fonctionnel | App Gradio + FAISS + citations | Déc 2024 |
| **v1.1** | Tests unitaires | Pytest + couverture 80%+ | Jan 2025 |
| **v1.2** | Retrieval hybride | FAISS + BM25 + RRF | Fév 2025 |
| **v1.3** | UI avancée | Score confiance + export JSON | Mar 2025 |
| **v2.0** | Production-ready | CI/CD + monitoring + logs | Avr 2025 |

---

## 🎓 Contexte Académique

Projet réalisé dans le cadre du cours **Data Science** à **DataBird** (2025).

**Adaptation du sujet** : Au lieu d'analyser des tweets sur le changement climatique, j'ai créé un assistant aligné avec mon rôle de Product Manager en expérimentation. Le projet respecte tous les critères techniques (HuggingFace, LLM, Langchain, Gradio) tout en étant directement utilisable dans mon travail quotidien.

**Approche de développement** : En tant que PM, j'ai utilisé Claude (Anthropic) comme co-pilote technique pour l'implémentation, me concentrant sur l'architecture, les décisions produit, et la compréhension des concepts RAG (retrieval, embeddings, generation). Cette démarche reflète l'utilisation moderne d'outils AI pour le prototypage rapide et le développement d'une expertise technique adaptée au rôle de Product Manager.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Ouvrez une issue ou une pull request.

**Note** : Ce projet a été initialement développé comme prototype PM avec assistance AI. Les contributions de développeurs pour améliorer la qualité du code, optimiser les performances, ou ajouter des features sont particulièrement appréciées.

**Axes d'amélioration prioritaires** :
- Élargir le corpus (blogs techniques, docs internes)
- Ajouter des tests automatisés
- Implémenter la recherche hybride

---

## 📄 Licence

MIT © 2025 El Mehdi BELAHNECH

---

## 🔗 Liens Utiles

- 📦 [Dataset HuggingFace](https://huggingface.co/datasets/lmhdii/experiment-assistant-dataset)
- 🚀 [Démo en ligne](https://huggingface.co/spaces/lmhdii/experiment-assistant) *(à venir)*
- 💬 [Issues GitHub](https://github.com/lmhdii/experiment-checklist-v2/issues)

---

**Maintenu avec ❤️ par un PM qui en avait marre de chercher "c'est quoi un SRM" sur Google**