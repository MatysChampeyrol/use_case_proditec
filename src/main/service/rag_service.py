"""
RAG Service - Version complète avec conversion et parsing intégrés
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ajouter le répertoire parent au PYTHONPATH
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.main.service.database_vect_service.database_vect_service import DatabaseVectService
from src.main.service.llm_service.llm_service import LlmService
from src.main.service.chunk_service.chunk_service import ChunkService
from src.main.service.embedding_service.embedding_service import EmbeddingService
from src.main.model.config import Config
import json

# Import pour conversion
from markitdown import MarkItDown


def load_file(file):
    """Load JSON configuration file"""
    path = f"{os.getcwd()}"
    with open(f"{path}/{file}", 'r', encoding='utf-8') as read_file:
        return json.load(read_file)


class RagService:
    """Parser Markdown intégré avec détection et suppression du sommaire"""
    
    def __init__(self):
        pass
    
    def clean_markdown(self, content: str) -> str:
        """
        Nettoyage avancé optimisé pour le RAG.
        Supprime le bruit de conversion, les numéros de page et reconstruit les phrases.
        """
        
        # 1. Suppression des caractères de contrôle bizarres (sauf newlines/tabs)
        content = "".join(ch for ch in content if ch.isprintable() or ch in '\n\t')
        
        # 1.5 NOUVEAU: Retirer le sommaire
        content = self.detect_and_remove_toc(content)

        # 2. Nettoyage ligne par ligne (bruit spécifique et pieds de page)
        lines = content.split('\n')
        cleaned_lines = []
        
        # On utilise _remove_repeating_lines d'abord pour virer les headers/footers récurrents
        lines = self._remove_repeating_lines(lines)

        for line in lines:
            # Supprimer les lignes de points (ex: "..........." ou ". . . .")
            if re.match(r'^\s*[\.\-_]{3,}\s*$', line):
                continue
            
            # Supprimer les patterns de Table des Matières (ex: "Introduction ....... 5")
            line = re.sub(r'[\.\-_]{3,}\s*\d+\s*$', '', line)
            
            # Supprimer les numéros de page isolés (ex: ligne contenant juste "42" ou "Page 12")
            if re.match(r'^\s*(Page\s*)?\d+\s*$', line, re.IGNORECASE):
                continue
            
            # Supprimer les lignes contenant trop de symboles par rapport au texte (bruit OCR)
            if self._is_noise_line(line):
                continue
                
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)

        # 3. Fusionner les lignes brisées (Reconstruction de phrases)
        content = self._merge_broken_lines(content)

        # 4. Normalisation des espaces et headers
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def detect_and_remove_toc(self, content: str) -> str:
        """Retire les sections de table des matières"""
        lines = content.split('\n')
        cleaned_lines = []
        skip_until = -1
        
        for i, line in enumerate(lines):
            if i < skip_until:
                continue
                
            # Détecter début de TOC
            if re.search(r'(table des matières|sommaire|table of contents|contents)', line.lower()):
                # Chercher la fin (section suivante ou 50 lignes)
                for j in range(i, min(i + 50, len(lines))):
                    if re.match(r'^#{1,3}\s', lines[j]) and j > i:
                        skip_until = j
                        break
                else:
                    skip_until = i + 50
                continue
            
            # Ignorer les lignes type TOC (avec points + numéro)
            if re.search(r'\.{3,}.*\d+\s*$', line):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_noise_line(self, line: str) -> bool:
        """Détecte si une ligne est purement du bruit"""
        if len(line.strip()) == 0:
            return False
        
        text_chars = len(re.findall(r'[a-zA-Z0-9À-ÿ]', line))
        total_chars = len(line.strip())
        
        if line.strip().startswith('#'):
            return False
            
        if total_chars > 0 and (text_chars / total_chars) < 0.3 and total_chars < 50:
            return True
            
        return False

    def _merge_broken_lines(self, content: str) -> str:
        """Fusionne les lignes qui ont été coupées arbitrairement"""
        lines = content.split('\n')
        merged_lines = []
        buffer = ""

        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                if buffer:
                    merged_lines.append(buffer)
                    buffer = ""
                merged_lines.append("")
                continue

            if re.match(r'^(#|\*|-|\+|\|)', line):
                if buffer:
                    merged_lines.append(buffer)
                    buffer = ""
                merged_lines.append(line)
                continue

            if buffer:
                if buffer.endswith('-'):
                    buffer = buffer[:-1] + line
                else:
                    buffer += " " + line
            else:
                buffer = line

            ends_with_punctuation = re.search(r'[.?!:]$', line) is not None
            
            next_line_is_start = False
            if i + 1 < len(lines):
                next_l = lines[i+1].strip()
                if next_l and (next_l[0].isupper() or re.match(r'^(#|\*|-)', next_l)):
                    next_line_is_start = True
            
            if ends_with_punctuation or next_line_is_start:
                merged_lines.append(buffer)
                buffer = ""

        if buffer:
            merged_lines.append(buffer)

        return '\n'.join(merged_lines)
    
    def _remove_repeating_lines(self, lines: List[str], threshold: int = 3) -> List[str]:
        """Remove lines that appear too frequently (likely headers/footers)"""
        from collections import Counter
        
        short_lines = [line.strip() for line in lines if 0 < len(line.strip()) < 100]
        line_counts = Counter(short_lines)
        
        repeating = {line for line, count in line_counts.items() if count >= threshold}
        
        return [line for line in lines if line.strip() not in repeating]


class RagService:
    """
    Service RAG avec conversion, parsing et filtrage intégrés
    """
    
    def __init__(self):
        # Load configuration
        self.config = Config(load_file("src/main/config/config.json"))
        
        # Initialize services
        self.chunk_service = ChunkService(self.config)
        self.embedding_service = EmbeddingService()
        self.database_vect_service = DatabaseVectService(self.config)
        self.llm_service = LlmService()
        
        # Initialize conversion and parsing
        self.md_converter = MarkItDown()
        self.markdown_parser = RagService()
        
        # Default collection name
        self.collection_name = "documents"
        self.collection = self.database_vect_service.get_or_create_collection(
            self.collection_name
        )
        
        # Paramètres de filtrage TOC
        self.enable_toc_filtering = True
        self.quality_threshold = 0.3
        self.enable_reranking = True
        
        print("✓ RAG Service initialized with integrated conversion and parsing")
    
    def upload(
        self, 
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = None,
        clean_markdown: bool = True
    ) -> dict:
        """
        Upload un document avec conversion et nettoyage intégrés
        
        Args:
            file_path: Chemin vers le fichier
            chunk_size: Taille des chunks en caractères
            chunk_overlap: Chevauchement entre chunks
            collection_name: Nom de la collection (optionnel)
            clean_markdown: Appliquer le nettoyage avancé (recommandé)
        
        Returns:
            dict: Résumé de l'upload avec statistiques
        """
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")
        
        print(f"\n{'='*60}")
        print(f"UPLOAD: {file_path.name}")
        print(f"{'='*60}")
        
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
                result = self.md_converter.convert(str(file_path))
                markdown_content = result.text_content
                print(f"✓ Conversion terminée")
            elif file_extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            print(f"✓ Document chargé ({len(markdown_content)} caractères)")
            
            # 2. Nettoyer le Markdown (NOUVEAU)
            if clean_markdown:
                print(f"→ Nettoyage du Markdown (suppression TOC, bruit)...")
                original_length = len(markdown_content)
                markdown_content = self.markdown_parser.clean_markdown(markdown_content)
                cleaned_length = len(markdown_content)
                removed = original_length - cleaned_length
                print(f"✓ Nettoyage terminé ({removed} caractères supprimés)")
            
            # 3. Chunker le document
            print(f"→ Chunking (size={chunk_size}, overlap={chunk_overlap})...")
            chunks = self.chunk_service.chunk(
                pages=markdown_content,
                size=chunk_size,
                overlap=chunk_overlap
            )
            
            num_chunks = len(chunks)
            print(f"✓ {num_chunks} chunks créés")
            
            # 4. Générer les embeddings
            print(f"→ Génération des embeddings...")
            
            chunk_texts = [chunk if isinstance(chunk, str) else chunk.page_content 
                          for chunk in chunks]
            
            embeddings = self.embedding_service.embed_batch(
                texts=chunk_texts,
                batch_size=32,
                is_query=False,
                show_progress=True
            )
            
            print(f"✓ {len(embeddings)} embeddings générés")
            
            # 5. Stocker dans ChromaDB
            print(f"→ Stockage dans ChromaDB...")
            
            all_ids = collection.get()["ids"]
            numeric_ids = [int(i) for i in all_ids if i.isdigit()]
            last_id = max(numeric_ids) if numeric_ids else 0
            
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
                chunk_id = str(last_id + i + 1)
                
                metadata = {
                    "source": file_path.name,
                    "source_path": str(file_path),
                    "chunk_index": i,
                    "total_chunks": num_chunks,
                    "file_type": file_extension,
                    "cleaned": clean_markdown
                }
                
                if hasattr(chunks[i], 'metadata'):
                    metadata.update(chunks[i].metadata)
                
                self.database_vect_service.collection_add_or_update(
                    collection=collection,
                    id=chunk_id,
                    embedding_vector=embedding,
                    documents=chunk_text,
                    metadatas=metadata
                )
            
            print(f"✓ {num_chunks} chunks stockés dans '{collection.name}'")
            
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
                "total_characters": len(markdown_content),
                "cleaned": clean_markdown
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
        top_k: int = 5,
        filter_toc: bool = True,
        show_details: bool = False
    ) -> dict:
        """
        Répond à une question avec filtrage intelligent du sommaire
        
        Args:
            question: La question de l'utilisateur
            collection_name: Collection à interroger (optionnel)
            top_k: Nombre de documents à retourner
            filter_toc: Activer le filtrage des chunks de sommaire
            show_details: Afficher les détails du filtrage
        
        Returns:
            dict: Réponse avec contexte et métadonnées
        """
        
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")
        
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
        
        # 2. Recherche (chercher plus si filtrage activé)
        search_multiplier = 3 if filter_toc else 1
        initial_top_k = top_k * search_multiplier
        
        print(f"→ Recherche des {initial_top_k} documents les plus similaires...")
        results = self.database_vect_service.query(
            collection=collection,
            embedded_question=question_embedding,
            number_results=initial_top_k
        )
        
        formatted_results = self.database_vect_service.format_chroma_results(results)
        
        print(f"✓ {len(formatted_results)} résultats bruts trouvés")
        
        # 3. FILTRAGE INTELLIGENT
        if filter_toc:
            print(f"\n{'─'*60}")
            print("FILTRAGE DES CHUNKS DE SOMMAIRE")
            print(f"{'─'*60}")
            
            scored_results = self._score_chunk_quality(formatted_results)
            
            if show_details:
                print("\n📊 Analyse de qualité:")
                for i, res in enumerate(scored_results[:10], 1):
                    quality = res.get('quality_score', 0)
                    is_filtered = quality < self.quality_threshold
                    status = "❌ FILTRÉ" if is_filtered else "✅ GARDÉ"
                    print(f"  [{i}] Distance: {res['distance']:.4f} | Qualité: {quality:.2f} | {status}")
                    if show_details and i <= 3:
                        preview = res['document'][:100].replace('\n', ' ')
                        print(f"       Preview: {preview}...")
            
            filtered_results = [
                r for r in scored_results 
                if r.get('quality_score', 0) >= self.quality_threshold
            ]
            
            filtered_count = len(formatted_results) - len(filtered_results)
            print(f"\n🗑️  {filtered_count} chunks filtrés (sommaire/bruit)")
            print(f"📄 {len(filtered_results)} chunks de qualité restants")
            
            if self.enable_reranking and len(filtered_results) > 0:
                print(f"→ Reranking par score combiné...")
                filtered_results = self._rerank_by_combined_score(filtered_results)
                print(f"✓ Résultats rerankés")
        else:
            filtered_results = formatted_results
        
        # 4. Prendre les top_k meilleurs
        final_results = filtered_results[:top_k]
        
        if len(final_results) == 0:
            print("\n⚠️  AUCUN RÉSULTAT de qualité trouvé!")
            return {
                "question": question,
                "answer": "Je n'ai pas trouvé d'information pertinente dans la base de données.",
                "context": [],
                "num_sources": 0,
                "collection": collection.name,
                "warning": "no_quality_results"
            }
        
        print(f"\n{'─'*60}")
        print(f"TOP {len(final_results)} RÉSULTATS FINAUX")
        print(f"{'─'*60}")
        
        for i, res in enumerate(final_results, 1):
            quality = res.get('quality_score', 'N/A')
            combined = res.get('combined_score', 'N/A')
            source = res['metadata'].get('source', 'unknown')
            
            print(f"[{i}] Distance: {res['distance']:.4f} | Qualité: {quality:.2f} | "
                  f"Score: {combined:.2f} | Source: {source}")
        
        # 5. Construire le contexte pour le LLM
        context_items = []
        for item in final_results:
            context_items.append({
                "page_content": item["document"],
                "metadata": item["metadata"],
                "distance": item["distance"],
                "quality_score": item.get("quality_score", 1.0)
            })
        
        # 6. Générer la réponse avec le LLM
        print(f"\n{'─'*60}")
        print(f"→ Génération de la réponse avec le LLM...")
        llm_response = self.llm_service.rag_completion(
            question=question,
            similarity_query=context_items
        )
        
        print(f"✓ Réponse générée")
        
        result = {
            "question": question,
            "answer": llm_response,
            "context": final_results,
            "num_sources": len(final_results),
            "collection": collection.name,
            "filtering_stats": {
                "initial_results": len(formatted_results),
                "filtered_out": len(formatted_results) - len(filtered_results) if filter_toc else 0,
                "final_results": len(final_results),
                "quality_threshold": self.quality_threshold if filter_toc else None
            }
        }
        
        print(f"\n{'='*60}")
        print(f"RÉPONSE:")
        print(f"{'='*60}")
        print(llm_response)
        print(f"{'='*60}\n")
        
        return result
    
    def _score_chunk_quality(self, results: List[Dict]) -> List[Dict]:
        """Score la qualité de chaque chunk (0 = mauvais, 1 = excellent)"""
        
        for res in results:
            doc = res['document']
            score = 1.0
            
            # Pattern TOC: ".... 42" ou "--- 15"
            if re.search(r'[\.\-_]{3,}\s*\d+\s*$', doc):
                score *= 0.1
            
            # Ligne avec beaucoup de points
            dots_ratio = doc.count('.') / max(len(doc), 1)
            if dots_ratio > 0.05:
                score *= 0.3
            
            # Pattern: "Chapitre X ..... Y"
            if re.search(r'(chapitre|section|partie|chapter)\s+\d+.*[\.\-_]+.*\d+', doc.lower()):
                score *= 0.1
            
            # Numéro de page isolé à la fin
            if re.search(r'\b(page\s*)?\d+\s*$', doc.strip(), re.IGNORECASE):
                score *= 0.4
            
            # Longueur du texte
            text_length = len(doc.strip())
            if text_length < 50:
                score *= 0.3
            elif text_length < 100:
                score *= 0.6
            
            # Densité d'information
            words = doc.split()
            if len(words) > 0:
                significant_words = [w for w in words if len(w) > 4]
                word_ratio = len(significant_words) / len(words)
                if word_ratio < 0.2:
                    score *= 0.5
            
            # Présence de verbes
            verb_patterns = [
                r'\best\b', r'\bsont\b', r'\bfait\b', r'\bfont\b',
                r'\bdoit\b', r'\bpeut\b', r'\bpeuvent\b', r'\bsera\b',
                r'\bétait\b', r'\bpermet\b', r'\butilise\b', r'\bcontient\b'
            ]
            has_verbs = any(re.search(pattern, doc.lower()) for pattern in verb_patterns)
            if not has_verbs and len(doc) > 50:
                score *= 0.6
            
            # Ratio sauts de ligne
            newline_ratio = doc.count('\n') / max(len(doc), 1)
            if newline_ratio > 0.05:
                score *= 0.4
            
            # Listes numérotées
            lines = doc.split('\n')
            numbered_lines = sum(1 for line in lines if re.match(r'^\d+[\.\)]\s', line.strip()))
            if len(lines) > 2 and numbered_lines / len(lines) > 0.5:
                score *= 0.3
            
            res['quality_score'] = max(0.0, min(1.0, score))
        
        return results
    
    def _rerank_by_combined_score(self, results: List[Dict]) -> List[Dict]:
        """Reranke par score combiné: 60% distance + 40% qualité"""
        
        for res in results:
            distance = res['distance']
            quality = res.get('quality_score', 1.0)
            
            normalized_distance = max(0, 1 - (distance / 2.0))
            combined_score = (0.6 * normalized_distance) + (0.4 * quality)
            
            res['combined_score'] = combined_score
        
        return sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True)
    
    def set_filtering_params(
        self, 
        enable_filtering: bool = True,
        quality_threshold: float = 0.3,
        enable_reranking: bool = True
    ):
        """Configure les paramètres de filtrage"""
        self.enable_toc_filtering = enable_filtering
        self.quality_threshold = quality_threshold
        self.enable_reranking = enable_reranking
        
        print(f"⚙️  Paramètres de filtrage mis à jour:")
        print(f"   • Filtrage: {'✓' if enable_filtering else '✗'}")
        print(f"   • Seuil qualité: {quality_threshold}")
        print(f"   • Reranking: {'✓' if enable_reranking else '✗'}")
    
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
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║          RAG SERVICE INTÉGRÉ - GUIDE D'UTILISATION            ║
    ╚════════════════════════════════════════════════════════════════╝
    
    ✨ NOUVEAUTÉS:
    
    1️⃣  CONVERSION AUTOMATIQUE:
       - Upload de PDF, DOCX, XLSX, PPTX directement
       - Conversion en Markdown intégrée
    
    2️⃣  NETTOYAGE INTELLIGENT:
       - Suppression automatique du sommaire
       - Fusion des lignes cassées
       - Suppression du bruit (headers/footers répétitifs)
    
    3️⃣  FILTRAGE AVANCÉ:
       - Détection des chunks de type sommaire
       - Score de qualité pour chaque chunk
       - Reranking intelligent (distance + qualité)
    
    ════════════════════════════════════════════════════════════════
    """)
    
    # Exemple 1: Upload avec nettoyage
    print("\n" + "="*80)
    print("EXEMPLE 1: Upload d'un PDF avec nettoyage")
    print("="*80)
    
    pdf_path = "ReferenceManualVision.pdf"
    
    if Path(pdf_path).exists():
        upload_result = rag.upload(
            file_path=pdf_path,
            chunk_size=1000,
            chunk_overlap=200,
            clean_markdown=True  # Active le nettoyage
        )
    else:
        print(f"⚠️  Fichier '{pdf_path}' non trouvé, skip upload example")
    
    # Exemple 2: Question avec filtrage
    print("\n" + "="*80)
    print("EXEMPLE 2: Question avec filtrage du sommaire")
    print("="*80)
    
    response = rag.ask(
        question="Code erreur 110",
        top_k=5,
        filter_toc=True,      # Active le filtrage
        show_details=True     # Affiche les scores
    )
    
    print("\n📊 Statistiques de filtrage:")
    print(f"  • Résultats initiaux: {response['filtering_stats']['initial_results']}")
    print(f"  • Filtrés: {response['filtering_stats']['filtered_out']}")
    print(f"  • Finaux: {response['filtering_stats']['final_results']}")
    
    # Exemple 3: Ajuster les paramètres
    print("\n" + "="*80)
    print("EXEMPLE 3: Personnalisation du filtrage")
    print("="*80)
    
    rag.set_filtering_params(
        enable_filtering=True,
        quality_threshold=0.5,  # Plus strict
        enable_reranking=True
    )
    
    print("""
    ════════════════════════════════════════════════════════════════
    
    💡 CONSEILS D'UTILISATION:
    
    • quality_threshold = 0.3  → Équilibré (recommandé)
    • quality_threshold = 0.5  → Strict (peu de faux positifs)
    • quality_threshold = 0.2  → Permissif (plus de résultats)
    
    • show_details=True pour voir les scores et déboguer
    • clean_markdown=True pour un meilleur RAG (recommandé)
    
    ════════════════════════════════════════════════════════════════
    """)