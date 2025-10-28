"""
Build curated Wikipedia dataset for experimentation topics.
Fetches 24 hand-selected articles (FR/EN) and pushes to HuggingFace Hub.
"""

import os
from typing import List, Dict, Optional
import wikipediaapi
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from dotenv import load_dotenv

load_dotenv()

# Configuration
HF_USER = os.getenv("HF_USER", "lmhdii")
DS_NAME = f"{HF_USER}/experiment-assistant-dataset"

# Curated Wikipedia pages (12 EN + 12 FR)
CANDIDATES_EN = {
    "A/B testing": ["A/B testing"],
    "Interleaving": ["Interleaving (information retrieval)"],
    "Sequential analysis": ["Sequential analysis"],
    "False discovery rate": ["False discovery rate"],
    "Sample size": ["Sample size determination"],
    "Power": ["Power (statistics)"],
    "Non-inferiority": ["Non-inferiority trial"],
    "Equivalence": ["Equivalence test"],
    "Bandit": ["Multi-armed bandit"],
    "Thompson": ["Thompson sampling"],
    "RCT": ["Randomized controlled trial"],
    "Control": ["Scientific control"],
}

CANDIDATES_FR = {
    "Test A/B": ["Test A/B"],
    "Analyse séquentielle": ["Analyse séquentielle"],
    "FDR": ["Taux de fausses découvertes"],
    "Taille échantillon": ["Taille d'échantillon"],
    "Puissance": ["Puissance statistique"],
    "Non-infériorité": ["Essai de non-infériorité"],
    "Équivalence": ["Test d'équivalence"],
    "Bandit": ["Bandit manchot"],
    "Thompson": ["Échantillonnage de Thompson"],
    "Essai randomisé": ["Essai randomisé contrôlé"],
    "Témoin": ["Groupe témoin"],
    "Contrôle": ["Groupe de contrôle"],
}

# Schema
FEATURES = Features({
    "id": Value("string"),
    "source_type": Value("string"),
    "title": Value("string"),
    "url": Value("string"),
    "language": Value("string"),
    "year": Value("string"),
    "topics": Sequence(Value("string")),
    "text": Value("string"),
})


def fetch_page(wiki: wikipediaapi.Wikipedia, lang: str, title: str) -> Optional[Dict]:
    """Fetch a single Wikipedia page."""
    page = wiki.page(title)
    
    if not page.exists():
        print(f"  ❌ '{title}' not found")
        return None
    
    if not page.text or len(page.text.strip()) < 100:
        print(f"  ⚠️  '{title}' too short")
        return None
    
    print(f"  ✅ {page.title} ({len(page.text)} chars)")
    
    return {
        "id": f"wiki::{lang}::{page.title}",
        "source_type": "wikipedia",
        "title": page.title,
        "url": page.fullurl,
        "language": lang,
        "year": "",
        "topics": [],
        "text": page.text,
    }


def collect_language(lang: str, candidates: Dict[str, List[str]]) -> List[Dict]:
    """Collect all pages for a given language."""
    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent="experiment-assistant/1.0 (educational project)"
    )
    
    results = []
    seen_titles = set()
    
    print(f"\n📚 Fetching {lang.upper()} articles...")
    
    for topic, titles in candidates.items():
        print(f"\n🔍 Topic: {topic}")
        for title in titles:
            if title in seen_titles:
                continue
            
            row = fetch_page(wiki, lang, title)
            if row and row["title"] not in seen_titles:
                results.append(row)
                seen_titles.add(row["title"])
                break  # Found one, move to next topic
    
    return results


def main():
    """Main execution."""
    print("=" * 60)
    print("🚀 Building Wikipedia Dataset for Experiment Assistant")
    print("=" * 60)
    
    # Collect EN
    en_articles = collect_language("en", CANDIDATES_EN)
    
    # Collect FR
    fr_articles = collect_language("fr", CANDIDATES_FR)
    
    # Create datasets
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"  - EN articles: {len(en_articles)}")
    print(f"  - FR articles: {len(fr_articles)}")
    print(f"  - Total: {len(en_articles) + len(fr_articles)}")
    
    wiki_en = Dataset.from_list(en_articles, features=FEATURES)
    wiki_fr = Dataset.from_list(fr_articles, features=FEATURES)
    
    dataset_dict = DatasetDict({
        "wiki_en": wiki_en,
        "wiki_fr": wiki_fr,
    })
    
    # Push to HuggingFace Hub
    print("\n" + "=" * 60)
    print(f"📤 Pushing to HuggingFace Hub: {DS_NAME}")
    
    try:
        dataset_dict.push_to_hub(DS_NAME, private=False)
        print(f"✅ Dataset published successfully!")
        print(f"🔗 https://huggingface.co/datasets/{DS_NAME}")
    except Exception as e:
        print(f"⚠️  Upload failed: {e}")
        print("💡 Tip: Make sure HF_TOKEN is set in .env")
        print("    Get token at: https://huggingface.co/settings/tokens")
    
    print("=" * 60)


if __name__ == "__main__":
    main()