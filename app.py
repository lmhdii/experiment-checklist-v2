"""
Experiment Assistant - Gradio UI
A bilingual RAG assistant for online experimentation topics.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import gradio as gr

# Load environment
load_dotenv()

# Check API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment.\n"
        "Get a free key at: https://console.groq.com/keys\n"
        "Add it to your .env file: GROQ_API_KEY=your_key_here"
    )

# Configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

print("🚀 Initializing Experiment Assistant...")

# Initialize LLM
print(f"🤖 Loading LLM: {LLM_MODEL}")
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    groq_api_key=GROQ_API_KEY,
)

# Prompt template (bilingual)
prompt_template = """Tu es un assistant expert en expérimentation en ligne (A/B testing, analyse statistique).
Réponds de manière concise et claire en utilisant uniquement le contexte ci-dessous.
Si le contexte ne permet pas de répondre, dis simplement « Je ne trouve pas cette information dans ma base de connaissances. »

Contexte :
{context}

Question : {question}

Réponse (3-5 phrases maximum) :"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Load FAISS index
print(f"📂 Loading FAISS index from: {INDEX_DIR}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
    chain_type_kwargs={"prompt": PROMPT},
)

print("✅ Assistant ready!")


def answer_question(question: str) -> str:
    """
    Answer a question using RAG.
    
    Args:
        question: User's question
        
    Returns:
        HTML formatted response with answer and sources
    """
    question = question.strip()
    
    if not question:
        return "<i>Veuillez entrer une question...</i>"
    
    try:
        # Generate answer
        answer = qa_chain.run(question)
        
        # Retrieve source documents
        docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)
        
        # Format response with sources
        html_parts = [
            "<div style='font-family: system-ui; max-width: 800px;'>",
            "<h3 style='color: #2563eb; margin-bottom: 16px;'>💬 Réponse</h3>",
            f"<div style='padding: 16px; background: #f8fafc; border-radius: 8px; margin-bottom: 24px;'>{answer}</div>",
            "<h3 style='color: #2563eb; margin-bottom: 16px;'>📚 Sources</h3>",
        ]
        
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Unknown")
            url = doc.metadata.get("url", "#")
            lang = doc.metadata.get("language", "??").upper()
            snippet = doc.page_content[:200].replace("\n", " ")
            
            source_html = f"""
            <div style='padding: 12px; margin-bottom: 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: white;'>
                <div style='font-weight: 600; margin-bottom: 4px;'>
                    <span style='background: #dbeafe; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;'>{lang}</span>
                    {title}
                </div>
                <a href='{url}' target='_blank' style='color: #3b82f6; text-decoration: none; font-size: 14px;'>{url}</a>
                <div style='color: #64748b; margin-top: 8px; font-size: 14px;'>{snippet}...</div>
            </div>
            """
            html_parts.append(source_html)
        
        html_parts.append("</div>")
        
        return "".join(html_parts)
    
    except Exception as e:
        return f"<div style='color: #dc2626; padding: 16px;'>❌ Erreur : {str(e)}</div>"


# Gradio Interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Experiment Assistant",
    css="""
    .gradio-container {max-width: 900px !important; margin: auto;}
    """
) as demo:
    
    gr.Markdown(
        """
        # 🔬 Experiment Assistant
        
        Assistant bilingue (FR/EN) pour l'expérimentation en ligne.  
        Posez vos questions sur l'A/B testing, le SRM, la FDR, la puissance statistique, etc.
        
        **Exemples** :
        - Qu'est-ce qu'un test A/B ?
        - What is a Sample Ratio Mismatch?
        - Comment calculer la puissance statistique ?
        """
    )
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Votre question",
            placeholder="Ex: Quelle est la différence entre A/B testing et interleaving ?",
            lines=2,
        )
    
    with gr.Row():
        submit_btn = gr.Button("🔍 Rechercher", variant="primary", size="lg")
        clear_btn = gr.ClearButton([question_input], value="🗑️ Effacer")
    
    output = gr.HTML(label="Réponse")
    
    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=output,
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=question_input,
        outputs=output,
    )
    
    gr.Markdown(
        """
        ---
        **💡 Comment ça marche ?**
        1. Votre question est convertie en vecteur
        2. FAISS trouve les 3 passages Wikipedia les plus pertinents
        3. Llama-3.1 génère une réponse basée sur ces passages
        4. Les sources sont citées pour vérification
        
        **📊 Corpus** : 17 articles Wikipedia (FR/EN) | **🤖 LLM** : Llama-3.1-8B (Groq)
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )