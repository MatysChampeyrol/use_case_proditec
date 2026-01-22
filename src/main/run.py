import gradio as gr
import os
from pathlib import Path
from src.main.service.rag_service import RagService 

# Initialisation du service
rag = RagService()

def process_upload(file, chunk_size, chunk_overlap):
    if file is None:
        return gr.update(value="⚠️ Veuillez sélectionner un fichier.", visible=True), "Aucun document"
    
    try:
        result = rag.upload(
            file_path=file.name,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap)
        )
        stats = rag.get_collection_stats()
        
        msg = f"✅ **{result['filename']}** indexé !\n\n📦 {result['num_chunks']} chunks créés."
        return gr.update(value=msg, visible=True), f"Total chunks: {stats['total_documents']}"
    except Exception as e:
        return gr.update(value=f"❌ Erreur : {str(e)}", visible=True), "Erreur"

def respond(message, history, top_k):
    if not message:
        return "", history
    
    try:
        # Appel du service RAG
        response = rag.ask(question=message, top_k=int(top_k))
        
        answer = response['answer']
        sources = response['context']
        
        # --- Formatage enrichi des sources avec extraits ---
        source_text = "\n\n---\n### 🔍 Sources et extraits consultés\n"
        
        for i, src in enumerate(sources, 1):
            name = src['metadata'].get('source', 'Inconnu')
            score = src.get('distance', 0)
            # On récupère le contenu du chunk
            content = src.get('document', 'Contenu non disponible')
            
            # On crée un bloc repliable (Accordion) en Markdown/HTML
            source_text += f"""
<details>
  <summary><b>{i}. {name}</b> (Score: {score:.3f})</summary>
  <div style="background-color: rgba(0,0,0,0.05); padding: 10px; border-left: 4px solid #2196F3; margin-top: 5px; font-size: 0.9em;">
    <i>"{content}"</i>
  </div>
</details>
"""
        
        full_response = f"{answer}{source_text}"
        
        # Structure imposée par Gradio 5/6 (Dictionnaires)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_response})
        
        return "", history
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Erreur : {str(e)}"})
        return "", history

# --- Interface Gradio ---

custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks(title="Expert RAG Assistant") as demo:
    
    gr.Markdown("# 🤖 Assistant Documentaire Intelligent")
    
    with gr.Row():
        # Panneau de gauche
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📄 Configuration")
                file_input = gr.File(label="Charger un document")
                
                with gr.Accordion("Réglages avancés", open=False):
                    chunk_size = gr.Slider(500, 2000, value=1000, step=100, label="Taille")
                    chunk_overlap = gr.Slider(0, 500, value=200, step=50, label="Overlap")
                
                upload_btn = gr.Button("🚀 Indexer", variant="primary")
                status_box = gr.Markdown(visible=False)
                db_stats = gr.Label(value="En attente...", label="Statut Base")

        # Panneau de droite
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(show_label=False, height=600)
            
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Posez votre question...", show_label=False, scale=9)
                submit_btn = gr.Button("Envoyer", variant="primary", scale=1)
            
            with gr.Row():
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Nombre de sources (Top-K)")
                clear = gr.ClearButton([msg_input, chatbot], value="Effacer le chat")

    # --- Événements ---
    upload_btn.click(
        fn=process_upload,
        inputs=[file_input, chunk_size, chunk_overlap],
        outputs=[status_box, db_stats]
    )
    
    submit_btn.click(respond, [msg_input, chatbot, top_k_slider], [msg_input, chatbot])
    msg_input.submit(respond, [msg_input, chatbot, top_k_slider], [msg_input, chatbot])

if __name__ == "__main__":
    demo.launch(theme=custom_theme, debug=True)