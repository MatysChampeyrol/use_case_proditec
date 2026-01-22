import gradio as gr
from pathlib import Path
from src.main.service.rag_service import RagService  # Suppose que ton code est dans rag_service.py
import json

# Initialiser le service RAG
rag = RagService()

# Fonction pour l'upload via Gradio
def upload_file(file, chunk_size=1000, chunk_overlap=200, collection_name=None):
    if file is None:
        return "❌ Aucun fichier sélectionné"

    # Gradio fournit maintenant un chemin temporaire dans file.name
    # Pas besoin de faire file.read()
    if hasattr(file, "name"):
        file_path = Path(file.name)
    else:
        file_path = Path(file)

    try:
        result = rag.upload(
            file_path=file_path,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            collection_name=collection_name
        )
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"❌ Erreur: {e}"

# Fonction pour poser une question via Gradio
def ask_question(question, collection_name=None, top_k=5):
    if not question:
        return "❌ Veuillez entrer une question"
    
    try:
        response = rag.ask(
            question=question,
            collection_name=collection_name,
            top_k=int(top_k)
        )
        # On renvoie uniquement la réponse et le contexte
        answer_text = f"💬 Réponse du LLM:\n{response['answer']}\n\n📄 Contexte (top {len(response['context'])}):\n"
        for i, ctx in enumerate(response['context'], 1):
            answer_text += f"[{i}] Distance: {ctx['distance']:.4f}\n{ctx['document'][:200]}...\n\n"
        return answer_text
    except Exception as e:
        return f"❌ Erreur: {e}"

# Gradio UI
with gr.Blocks(title="RAG Service ChatGPT-like") as demo:
    gr.Markdown("# 📚 RAG Service Interface")
    
    with gr.Tab("Upload Document"):
        file_input = gr.File(label="Sélectionner un fichier", file_types=[".pdf", ".txt", ".md", ".docx"])
        chunk_size = gr.Number(value=1000, label="Taille des chunks")
        chunk_overlap = gr.Number(value=200, label="Chevauchement des chunks")
        collection_name_input = gr.Textbox(label="Nom de la collection (optionnel)")
        upload_btn = gr.Button("⬆️ Upload")
        upload_output = gr.Textbox(label="Résultat de l'upload", lines=20)
        upload_btn.click(
            upload_file,
            inputs=[file_input, chunk_size, chunk_overlap, collection_name_input],
            outputs=upload_output
        )
    
    with gr.Tab("Posez une question"):
        question_input = gr.Textbox(label="Votre question", lines=2)
        collection_name_q = gr.Textbox(label="Collection (optionnel)")
        top_k_input = gr.Number(value=5, label="Nombre de documents similaires à récupérer")
        ask_btn = gr.Button("❓ Poser la question")
        ask_output = gr.Textbox(label="Réponse du LLM + contexte", lines=20)
        ask_btn.click(
            ask_question,
            inputs=[question_input, collection_name_q, top_k_input],
            outputs=ask_output
        )

# Lancer l'interface
demo.launch()
