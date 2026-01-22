"""
RAG Service - Complete implementation with upload and ask functions
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Remonte à la racine du projet
sys.path.insert(0, str(project_root))

from src.main.service.database_vect_service.database_vect_service import DatabaseVectService
from src.main.service.llm_service.llm_service import LlmService
from src.main.service.chunk_service.chunk_service import ChunkService
from src.main.service.embedding_service.embedding_service import EmbeddingService
from src.main.model.config import Config
import json
import shutil


def load_file(file):
    """Load JSON configuration file"""
    path = f"{os.getcwd()}"
    with open(f"{path}/{file}", 'r', encoding='utf-8') as read_file:
        return json.load(read_file)


class RagService:
    """
    Service RAG principal avec deux fonctions principales:
    - upload(): charge un document, le chunke, l'embedde et le stocke
    - ask(): répond à une question en utilisant la base vectorielle
    """
    
    def __init__(self):
        # Load configuration
        self.config = Config(load_file("src/main/config/config.json"))
        
        # Initialize services
        self.chunk_service = ChunkService(self.config)
        self.embedding_service = EmbeddingService()
        self.database_vect_service = DatabaseVectService(self.config)
        self.llm_service = LlmService()
        
        # Default collection name (peut être configuré)
        self.collection_name = "documents"
        self.collection = self.database_vect_service.get_or_create_collection(
            self.collection_name
        )
        
        print("✓ RAG Service initialized")
    
    def upload(
        self, 
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = None
    ) -> dict:
        """
        Upload un document depuis un chemin fichier, le convertit en chunks, 
        génère les embeddings et stocke tout dans la base vectorielle.
        
        Args:
            file_path: Chemin vers le fichier (str ou Path)
            chunk_size: Taille des chunks en caractères
            chunk_overlap: Chevauchement entre chunks
            collection_name: Nom de la collection (optionnel)
        
        Returns:
            dict: Résumé de l'upload avec statistiques
        """
        
        # Convertir en Path
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")
        
        print(f"\n{'='*60}")
        print(f"UPLOAD: {file_path.name}")
        print(f"{'='*60}")
        
        # Utilise la collection spécifiée ou celle par défaut
        if collection_name:
            collection = self.database_vect_service.get_or_create_collection(
                collection_name
            )
        else:
            collection = self.collection
        
        try:
            # 1. Convertir en Markdown si nécessaire
            file_extension = file_path.suffix.lower()
            
            if file_extension in ['.pdf', '.docx', '.doc', '.xlsx', '.pptx']:
                print(f"→ Conversion en Markdown ({file_extension})...")
                markdown_content = self._convert_to_markdown(file_path)
            elif file_extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            else:
                # Texte brut
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            print(f"✓ Document chargé ({len(markdown_content)} caractères)")
            
            # 2. Chunker le document
            print(f"→ Chunking (size={chunk_size}, overlap={chunk_overlap})...")
            chunks = self.chunk_service.chunk(
                pages=markdown_content,
                size=chunk_size,
                overlap=chunk_overlap
            )
            
            num_chunks = len(chunks)
            print(f"✓ {num_chunks} chunks créés")
            
            # 3. Générer les embeddings
            print(f"→ Génération des embeddings...")
            
            # Extraire le texte des chunks
            chunk_texts = [chunk if isinstance(chunk, str) else chunk.page_content 
                          for chunk in chunks]
            
            # Embedder par batch
            embeddings = self.embedding_service.embed_batch(
                texts=chunk_texts,
                batch_size=32,
                is_query=False,
                show_progress=True
            )
            
            print(f"✓ {len(embeddings)} embeddings générés")
            
            # 4. Stocker dans ChromaDB
            print(f"→ Stockage dans ChromaDB...")
            
            # Obtenir le dernier ID
            all_ids = collection.get()["ids"]
            numeric_ids = [int(i) for i in all_ids if i.isdigit()]
            last_id = max(numeric_ids) if numeric_ids else 0
            
            # Stocker chaque chunk avec son embedding
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                chunk_id = str(last_id + i + 1)
                
                # Métadonnées
                metadata = {
                    "source": file_path.name,
                    "source_path": str(file_path),
                    "chunk_index": i,
                    "total_chunks": num_chunks,
                    "file_type": file_extension
                }
                
                # Si c'est un objet Document de langchain
                if hasattr(chunks[i], 'metadata'):
                    metadata.update(chunks[i].metadata)
                
                # Stocker
                self.database_vect_service.collection_add_or_update(
                    collection=collection,
                    id=chunk_id,
                    embedding_vector=embedding,
                    documents=chunk_text,
                    metadatas=metadata
                )
            
            print(f"✓ {num_chunks} chunks stockés dans '{collection.name}'")
            
            # Résumé
            result = {
                "status": "success",
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": file_extension,
                "num_chunks": num_chunks,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "collection": collection.name,
                "embedding_dimension": len(embeddings[0]),
                "total_characters": len(markdown_content)
            }
            
            print(f"\n{'='*60}")
            print("✓ UPLOAD TERMINÉ")
            print(json.dumps(result, indent=2))
            print(f"{'='*60}\n")
            
            return result
        
        except Exception as e:
            print(f"✗ ERREUR lors de l'upload: {e}")
            raise
    
    def ask(
        self,
        question: str,
        collection_name: str = None,
        top_k: int = 5
    ) -> dict:
        """
        Répond à une question en utilisant le RAG.
        
        Args:
            question: La question de l'utilisateur
            collection_name: Collection à interroger (optionnel)
            top_k: Nombre de documents similaires à récupérer
        
        Returns:
            dict: Réponse avec contexte et métadonnées
        """
        
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")
        
        # Utilise la collection spécifiée ou celle par défaut
        if collection_name:
            collection = self.database_vect_service.get_or_create_collection(
                collection_name
            )
        else:
            collection = self.collection
        
        # 1. Embedder la question
        print(f"→ Embedding de la question...")
        question_embedding = self.embedding_service.embed(
            texts=question,
            is_query=True,
            task_instruction="Given a question, retrieve passages that answer the question"
        )
        
        print(f"✓ Question embeddée (dim={len(question_embedding[0])})")
        
        # 2. Recherche de similarité dans ChromaDB
        print(f"→ Recherche des {top_k} documents les plus similaires...")
        results = self.database_vect_service.query(
            collection=collection,
            embedded_question=question_embedding,
            number_results=top_k
        )
        
        # Formater les résultats
        formatted_results = self.database_vect_service.format_chroma_results(results)
        
        print(f"✓ {len(formatted_results)} résultats trouvés")
        
        # Afficher les distances
        for i, res in enumerate(formatted_results):
            print(f"  [{i+1}] Distance: {res['distance']:.4f} | Source: {res['metadata'].get('source', 'unknown')}")
        
        # 3. Construire le contexte pour le LLM
        context_items = []
        for item in formatted_results:
            context_items.append({
                "page_content": item["document"],
                "metadata": item["metadata"],
                "distance": item["distance"]
            })
        
        # 4. Générer la réponse avec le LLM
        print(f"→ Génération de la réponse avec le LLM...")
        llm_response = self.llm_service.rag_completion(
            question=question,
            similarity_query=context_items
        )
        
        print(f"✓ Réponse générée")
        
        # Résultat final
        result = {
            "question": question,
            "answer": llm_response,
            "context": formatted_results,
            "num_sources": len(formatted_results),
            "collection": collection.name
        }
        
        print(f"\n{'='*60}")
        print(f"RÉPONSE:")
        print(f"{'='*60}")
        print(llm_response)
        print(f"{'='*60}\n")
        
        return result
    
    def _convert_to_markdown(self, file_path: Path) -> str:
        """
        Convertit un fichier en Markdown en utilisant markitdown.
        
        Args:
            file_path: Chemin du fichier à convertir
        
        Returns:
            str: Contenu Markdown
        """
        from markitdown import MarkItDown
        
        md_converter = MarkItDown()
        result = md_converter.convert(str(file_path))
        
        return result.text_content
    
    def list_collections(self):
        """Liste toutes les collections disponibles"""
        self.database_vect_service.get_list_collections()
    
    def delete_collection(self, collection_name: str):
        """Supprime une collection"""
        self.database_vect_service.delete_collection(collection_name)
        print(f"✓ Collection '{collection_name}' supprimée")
    
    def get_collection_stats(self, collection_name: str = None) -> dict:
        """Obtient les statistiques d'une collection"""
        if collection_name:
            collection = self.database_vect_service.get_or_create_collection(
                collection_name
            )
        else:
            collection = self.collection
        
        all_data = collection.get()
        
        stats = {
            "collection_name": collection.name,
            "total_documents": len(all_data["ids"]),
            "sample_metadata": all_data["metadatas"][:3] if all_data["metadatas"] else []
        }
        
        return stats


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    
    # Initialiser le service
    rag = RagService()
    
    # Exemple 1: Upload d'un document
    print("\n" + "="*80)
    print("EXEMPLE 1: Upload d'un document")
    print("="*80)
    
    # Créer un fichier test
    test_content = """
    # Introduction au Machine Learning
    
    Le machine learning est une branche de l'intelligence artificielle.
    
    ## Types d'apprentissage
    
    Il existe trois types principaux:
    - Apprentissage supervisé
    - Apprentissage non supervisé
    - Apprentissage par renforcement
    
    ## Applications
    
    Le ML est utilisé dans de nombreux domaines comme la reconnaissance d'images,
    le traitement du langage naturel, et les systèmes de recommandation.
    """
    
    # Créer un fichier temporaire pour le test
    temp_file = Path("test_document.md")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    # Upload avec le chemin du fichier
    upload_result = rag.upload(
        file_path="test_document.md",  # Juste le chemin!
        chunk_size=200,
        chunk_overlap=50
    )
    
    # Exemple 2: Poser une question
    print("\n" + "="*80)
    print("EXEMPLE 2: Poser une question")
    print("="*80)
    
    response = rag.ask(
        question="Quels sont les types d'apprentissage en machine learning?",
        top_k=3
    )
    
    print("\nRéponse structurée:")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    
    # Exemple 3: Statistiques de collection
    print("\n" + "="*80)
    print("EXEMPLE 3: Statistiques de la collection")
    print("="*80)
    
    stats = rag.get_collection_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Nettoyer
    temp_file.unlink()