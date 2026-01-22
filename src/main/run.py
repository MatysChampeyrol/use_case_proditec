import gradio as gr
from pathlib import Path
from src.main.service.rag_service import RagService 

# --- Initialisation du service RAG ---
rag = RagService()

# --- Pré-chargement automatique du document ---
preload_path = Path("ReferenceManualVision.pdf")
collection_name = "documents"

if preload_path.exists():
    stats = rag.get_collection_stats(collection_name=collection_name)
    already_indexed = any(
        md.get("source") == preload_path.name for md in stats.get("sample_metadata", [])
    )

    if not already_indexed:
        print(f"📄 Indexation automatique de {preload_path}...")
        try:
            rag.upload(
                file_path=preload_path,
                chunk_size=1000,
                chunk_overlap=200,
                collection_name=collection_name
            )
            print(f"✅ Document {preload_path.name} indexé dans la collection '{collection_name}'")
        except Exception as e:
            print(f"❌ Erreur lors de l'indexation automatique: {e}")
    else:
        print(f"ℹ️ {preload_path.name} est déjà indexé, pas besoin de re-upload.")
else:
    print(f"⚠️ Fichier {preload_path} introuvable, indexation automatique ignorée.")


# --- Fonction de réponse simplifiée ---
def respond(message, history):
    if not message:
        return "", history
    try:
        response = rag.ask(question=message, top_k=10)
        answer = response['answer']
        sources = response['context']

        # Sources repliables
        source_text = "\n\n---\n### 🔍 Sources consultées\n"
        for i, src in enumerate(sources, 1):
            name = src['metadata'].get('source', 'Inconnu')
            score = src.get('distance', 0)
            content = src.get('document', 'Contenu non disponible')
            source_text += f"""
<details>
  <summary><b>{i}. {name}</b> (Score: {score:.3f})</summary>
  <div style="background-color: rgba(0,0,0,0.05); padding: 10px; border-left: 4px solid #2196F3; margin-top: 5px; font-size: 0.9em;">
    <i>{content}</i>
  </div>
</details>
"""

        full_response = f"{answer}{source_text}"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_response})
        return "", history

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Erreur : {str(e)}"})
        return "", history


# --- Interface Gradio professionnelle ---
custom_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks(title="Expert RAG Assistant") as demo:
    gr.Markdown("# 🤖 Assistant Documentaire Intelligent")
    gr.Markdown("Posez vos questions au document préchargé.")

    # Chatbot avec message d'accueil
    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": "Bonjour, je suis un chatbot dédié au support technique, posez votre question."}],
        show_label=False,
        height=400
    )

    msg_input = gr.Textbox(placeholder="Tapez votre question ici...", show_label=False)
    submit_btn = gr.Button("Envoyer", variant="primary")

    submit_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
    msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])


if __name__ == "__main__":
    demo.launch(
        theme=custom_theme,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
