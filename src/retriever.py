"""
Simple FAISS retriever for document search.
Wraps FAISS vectorstore with a clean interface.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "faiss_index"))


class SimpleRetriever:
    """Simple FAISS-based retriever for semantic search."""
    
    def __init__(self, index_dir: Path = INDEX_DIR):
        """Initialize retriever by loading FAISS index."""
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_dir}\n"
                f"Run 'python src/indexer.py' first to create the index."
            )
        
        print(f"📂 Loading FAISS index from: {index_dir}")
        
        # Load embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Load FAISS index
        self.vectorstore = FAISS.load_local(
            str(index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Retriever ready with {EMBED_MODEL}")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects with metadata
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)


def main():
    """CLI for testing retriever."""
    print("=" * 60)
    print("🔍 Simple Retriever - Interactive Search")
    print("=" * 60)
    
    try:
        retriever = SimpleRetriever()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    
    print("\n💡 Enter your questions (CTRL+C to quit)")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if not query:
                continue
            
            # Search
            results = retriever.search(query, k=3)
            
            print(f"\n📊 Found {len(results)} results:\n")
            
            for i, doc in enumerate(results, 1):
                title = doc.metadata.get("title", "Unknown")
                lang = doc.metadata.get("language", "??")
                url = doc.metadata.get("url", "#")
                snippet = doc.page_content[:150].replace("\n", " ")
                
                print(f"{i}. [{lang.upper()}] {title}")
                print(f"   {url}")
                print(f"   {snippet}...\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()