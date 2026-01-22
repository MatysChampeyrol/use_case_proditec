"""
Fonctions simples pour faire de l'embedding avec llama-embed-nemotron-8b
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Union

#"intfloat/e5-mistral-7b-instruct"

class EmbeddingService:
    """Classe simple pour générer des embeddings avec llama-embed-nemotron-8b."""
    
    def __init__(
        self, 
        model_name: str = "intfloat/e5-mistral-7b-instruct",
        device: str = None
    ):
        """
        Initialise le modèle.
        
        Args:
            model_name: Nom du modèle HuggingFace
            device: 'cuda' ou 'cpu' (None = auto-détection)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"
        
        print(f"Chargement du modèle sur {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
        ).eval().to(self.device)
        
        print("Modèle chargé ✓")
    
    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooling pour obtenir un vecteur par texte."""
        last_hidden_states = last_hidden_states.to(torch.float32)
        last_hidden_states_masked = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return F.normalize(embedding, dim=-1)
    
    def embed(
        self, 
        texts: Union[str, List[str]],
        is_query: bool = False,
        task_instruction: str = "Given a question, retrieve passages that answer the question",
        max_length: int = 4096
    ) -> List[List[float]]:
        """
        Génère des embeddings pour un ou plusieurs textes.
        
        Args:
            texts: Un texte (str) ou une liste de textes (List[str])
            is_query: Si True, ajoute l'instruction (pour les questions)
                     Si False, pas d'instruction (pour les documents)
            task_instruction: L'instruction à utiliser pour les queries
            max_length: Longueur max en tokens
        
        Returns:
            List[List[float]]: Liste de vecteurs de dimension 4096
            
        Exemples:
            # Un seul texte (document)
            >>> emb = embedder.embed("Python est un langage")
            >>> len(emb)  # 1 vecteur
            1
            >>> len(emb[0])  # 4096 dimensions
            4096
            
            # Plusieurs documents
            >>> emb = embedder.embed(["texte 1", "texte 2", "texte 3"])
            >>> len(emb)  # 3 vecteurs
            3
            
            # Query (avec instruction)
            >>> emb = embedder.embed("Comment apprendre Python?", is_query=True)
        """
        # Convertit en liste si c'est un seul texte
        if isinstance(texts, str):
            texts = [texts]
        
        # Formate avec instruction si c'est une query
        if is_query:
            formatted_texts = [
                f"Instruct: {task_instruction}\nQuery: {text}" 
                for text in texts
            ]
        else:
            formatted_texts = texts
        
        # Tokenization
        batch_dict = self.tokenizer(
            text=formatted_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        
        # Pooling
        embeddings = self._average_pool(
            outputs.last_hidden_state,
            batch_dict["attention_mask"]
        )
        
        # Conversion en liste Python
        return embeddings.cpu().numpy().tolist()
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
        task_instruction: str = "Given a question, retrieve passages that answer the question",
        max_length: int = 4096,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Génère des embeddings par batch (pour grandes quantités de textes).
        
        Args:
            texts: Liste de textes à encoder
            batch_size: Nombre de textes par batch
            is_query: Si True, ajoute l'instruction
            task_instruction: Instruction pour les queries
            max_length: Longueur max en tokens
            show_progress: Affiche la progression
        
        Returns:
            List[List[float]]: Liste de vecteurs de dimension 4096
            
        Exemple:
            # Encoder 1000 documents par batch de 32
            >>> docs = ["doc " + str(i) for i in range(1000)]
            >>> embeddings = embedder.embed_batch(docs, batch_size=32)
            >>> len(embeddings)
            1000
        """
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if show_progress:
                batch_num = i // batch_size + 1
                print(f"Batch {batch_num}/{num_batches} ({len(batch)} textes)...")
            
            batch_embeddings = self.embed(
                batch,
                is_query=is_query,
                task_instruction=task_instruction,
                max_length=max_length
            )
            
            all_embeddings.extend(batch_embeddings)
        
        if show_progress:
            print(f"✓ {len(all_embeddings)} embeddings générés")
        
        return all_embeddings


# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    
    # Initialise le modèle
    embedder = EmbeddingService()
    
    print("\n" + "="*80)
    print("EXEMPLE 1: Embedding d'un seul texte")
    print("="*80)
    
    text = "Python est un langage de programmation"
    emb = embedder.embed(text)
    print(f"Texte: {text}")
    print(f"Embedding shape: [{len(emb)}, {len(emb[0])}]")
    print(f"Premiers 5 floats: {emb[0][:5]}")
    
    
    print("\n" + "="*80)
    print("EXEMPLE 2: Embedding de plusieurs documents")
    print("="*80)
    
    documents = [
        "Les réseaux de neurones apprennent via la backpropagation",
        "ChromaDB est une base de données vectorielle",
        "Le transformer utilise le mécanisme d'attention"
    ]
    
    doc_embeddings = embedder.embed(documents, is_query=False)
    print(f"Nombre de documents: {len(documents)}")
    print(f"Nombre d'embeddings: {len(doc_embeddings)}")
    print(f"Dimension: {len(doc_embeddings[0])}")
    
    
    print("\n" + "="*80)
    print("EXEMPLE 3: Embedding d'une query (avec instruction)")
    print("="*80)
    
    query = "Comment fonctionnent les réseaux de neurones?"
    query_emb = embedder.embed(query, is_query=True)
    print(f"Query: {query}")
    print(f"Embedding shape: [{len(query_emb)}, {len(query_emb[0])}]")
    
    # Calcul de similarité
    import numpy as np
    query_vec = np.array(query_emb[0])
    similarities = [np.dot(query_vec, np.array(doc_emb)) for doc_emb in doc_embeddings]
    
    print("\nSimilarités avec les documents:")
    for i, (doc, sim) in enumerate(zip(documents, similarities)):
        print(f"  [{i}] {sim:.4f} - {doc}")
    
    
    print("\n" + "="*80)
    print("EXEMPLE 4: Embedding par batch (pour grandes quantités)")
    print("="*80)
    
    # Simule 100 documents
    large_docs = [f"Document numéro {i} avec du contenu..." for i in range(100)]
    
    batch_embeddings = embedder.embed_batch(
        large_docs,
        batch_size=16,
        is_query=False,
        show_progress=True
    )
    
    print(f"\nTotal embeddings: {len(batch_embeddings)}")
    print(f"Dimension: {len(batch_embeddings[0])}")
    
    
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    print("""
    embed():
    --------
    - Pour 1 texte ou quelques textes
    - Simple et direct
    - Retourne List[List[float]]
    
    embed_batch():
    --------------
    - Pour beaucoup de textes (100+)
    - Traite par batch pour économiser la mémoire
    - Affiche la progression
    - Retourne List[List[float]]
    
    Dimension des embeddings: 4096 (toujours)
    """)