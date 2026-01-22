from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re


class ChunkService:
    def __init__(self, config):
        self.config = config

        # <<< MODIF : limite de sécurité pour embeddings
        self.MAX_CHARS_FOR_EMBEDDING = 900

    def _add_context_metadata(self, docs):
        """
        Ajoute un résumé de contexte au début de chaque chunk
        """
        enriched_docs = []

        for doc in docs:
            meta = doc.metadata
            context = []

            if meta.get("chapter"):
                context.append(f"Chapitre : {meta['chapter']}")
            if meta.get("section"):
                context.append(f"Section : {meta['section']}")

            if context:
                doc.page_content = (
                    "Contexte du document:\n"
                    + " | ".join(context)
                    + "\n\n"
                    + doc.page_content
                )

            enriched_docs.append(doc)

        return enriched_docs

    def _extract_structure(self, text):
        """
        Découpe grossièrement par chapitres / sections
        """
        pattern = r"(?:\n|^)([IVX]+|\d+(?:\.\d+)*)\s+([A-ZÀ-ÿ].+)"
        splits = re.split(pattern, text)

        documents = []
        current_chapter = None
        current_section = None

        i = 0
        while i < len(splits) - 2:
            identifier = splits[i + 1]
            title = splits[i + 2]
            content = splits[i + 3] if i + 3 < len(splits) else ""

            if identifier.isupper():
                current_chapter = title
                current_section = None
            else:
                current_section = title

            if content.strip():
                documents.append(
                    Document(
                        page_content=content.strip(),
                        metadata={
                            "chapter": current_chapter,
                            "section": current_section,
                        },
                    )
                )
            i += 3

        return documents

    def _merge_small_chunks(self, chunks, min_size):
        """
        Fusionne les chunks trop petits pour garantir du contexte
        """
        merged = []
        buffer = None

        for chunk in chunks:
            if buffer is None:
                buffer = chunk
                continue

            if len(buffer.page_content) < min_size:
                buffer.page_content += "\n\n" + chunk.page_content
            else:
                merged.append(buffer)
                buffer = chunk

        if buffer:
            merged.append(buffer)

        return merged

    def _clamp_chunks_for_embedding(self, chunks):
        """
        <<< MODIF : Sécurité ABSOLUE contre les dépassements token
        """
        safe_chunks = []

        for doc in chunks:
            if len(doc.page_content) > self.MAX_CHARS_FOR_EMBEDDING:
                doc.page_content = doc.page_content[: self.MAX_CHARS_FOR_EMBEDDING]
            safe_chunks.append(doc)

        return safe_chunks

    def chunk(self, pages, size=1000, overlap=150):
        print("Try to split file with contextual chunking...")

        if isinstance(pages, str):
            base_docs = self._extract_structure(pages)
        else:
            base_docs = pages

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
            ],
        )

        chunks = splitter.split_documents(base_docs)

        # 🔥 GARANTIE DE CONTEXTE
        min_chunk_size = int(size * 0.6)
        chunks = self._merge_small_chunks(chunks, min_chunk_size)

        # 🔥 AJOUT CONTEXTE
        chunks = self._add_context_metadata(chunks)

        # 🔥 SÉCURITÉ EMBEDDINGS (CRITIQUE)
        chunks = self._clamp_chunks_for_embedding(chunks)

        print(f"OK, Number of chunks: {len(chunks)}")
        return chunks
