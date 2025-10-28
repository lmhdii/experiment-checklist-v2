"""
Create FAISS index from Wikipedia dataset.
Chunks articles and creates vector embeddings for semantic search.
"""

import os
import re
from pathlib import Path
from typing import List
from tqdm import tqdm

from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATASET_NAME = os.getenv("DATASET_NAME", "lmhdii/experiment-assistant-dataset")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "faiss_index"))


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text or "").strip()
    
    if len(text) <= size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def load_documents() -> List[Document]:
    """Load Wikipedia dataset and convert to LangChain documents."""
    print(f"📥 Loading dataset: {DATASET_NAME}")
    
    try:
        dataset = load_dataset(DATASET_NAME)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("💡 Make sure the dataset was published successfully")
        return []
    
    documents = []
    
    for split_name in ["wiki_en", "wiki_fr"]:
        if split_name not in dataset:
            print(f"⚠️  Split '{split_name}' not found, skipping")
            continue
        
        split = dataset[split_name]
        print(f"\n📚 Processing {split_name}: {len(split)} articles")
        
        for row in tqdm(split, desc=f"Chunking {split_name}"):
            metadata = {
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "language": row["language"],
                "source_type": row["source_type"],
            }
            
            # Chunk the article text
            chunks = chunk_text(row["text"], size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata=metadata)
                documents.append(doc)
    
    return documents


def build_index(documents: List[Document]) -> FAISS:
    """Create FAISS index from documents."""
    print(f"\n🔧 Creating embeddings with: {EMBED_MODEL}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print(f"🔨 Building FAISS index from {len(documents)} chunks...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore


def save_index(vectorstore: FAISS, output_dir: Path):
    """Save FAISS index to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving index to: {output_dir}")
    vectorstore.save_local(str(output_dir))
    
    print(f"✅ Index saved successfully!")
    print(f"   - {output_dir}/index.faiss")
    print(f"   - {output_dir}/index.pkl")


def test_index(vectorstore: FAISS):
    """Test the index with a sample query."""
    print("\n🧪 Testing index with sample query...")
    
    test_queries = [
        "What is A/B testing?",
        "Qu'est-ce qu'un test A/B ?",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            title = doc.metadata.get("title", "Unknown")
            lang = doc.metadata.get("language", "??")
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"  {i}. [{lang}] {title}")
            print(f"     {snippet}...")


def main():
    """Main execution."""
    print("=" * 60)
    print("🚀 Building FAISS Index for Experiment Assistant")
    print("=" * 60)
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        print("❌ No documents loaded. Exiting.")
        return
    
    print(f"\n📊 Total chunks: {len(documents)}")
    
    # Build index
    vectorstore = build_index(documents)
    
    # Save index
    save_index(vectorstore, INDEX_DIR)
    
    # Test index
    test_index(vectorstore)
    
    print("\n" + "=" * 60)
    print("✅ Indexing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()