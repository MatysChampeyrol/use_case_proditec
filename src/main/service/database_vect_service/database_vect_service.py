import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

def get_chroma_client():
    # On récupère les infos injectées par Docker
    host = os.getenv('CHROMA_HOST', 'localhost')
    port = os.getenv('CHROMA_PORT', '5435')      
    token = os.getenv('CHROMA_TOKEN')            

    print(f"Connexion à ChromaDB sur {host}:{port}...")

    return chromadb.HttpClient(
        host=host,
        port=int(port),
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=token
        )
    )

# Utilisation
client = get_chroma_client()

class DatabaseVectService():
    def __init__(self, config):
        self.config = config

    def get_or_create_collection(self, collection_name):
        collection = client.get_or_create_collection(name=f"{collection_name}")
        return collection

    # to add one element
    def collection_add_or_update(self, collection, id, embedding_vector, documents = None, metadatas=None):
        #upsert pour contracter create et update
        collection.upsert(
            ids=[id],
            embeddings=[embedding_vector],
            documents=[documents],
            metadatas=[metadatas]
        )

    # to store a whole embedded document
    def collection_store_embedded_document(self, collection, document_chunked, document_embedded):
        all_ids = collection.get()["ids"]
        numeric_ids = [int(i) for i in all_ids if i.isdigit()]
        last_id = max(numeric_ids) if numeric_ids else 0
        for i in range(len(document_chunked)):
            next_id = str(last_id + 1)
            self.collection_add_or_update(collection, next_id, document_embedded[i], document_chunked[i].page_content, document_chunked[i].metadata)
            last_id += 1

    def get_list_collections(self):
        print(client.list_collections())

    def get_element_collection_by_id(self, collection,id):
    #Il faut préciser les éléments à afficher comme embeddings et metadatas
        return collection.get(ids=[id], include=["embeddings", "metadatas", "documents"])

    def get_all_elements_collection(self, collection):
        return collection.get(include=["documents", "metadatas", "embeddings"])

    def delete_collection(self, collection_name):
        client.delete_collection(collection_name)

    # similarity query
    def query(self, collection, embedded_question, number_results):
        return collection.query(
                query_embeddings=embedded_question,
                n_results=number_results,
        )

    # To order results of the query into a list
    def format_chroma_results(self, results):
        formatted = []
        for idx, doc_id in enumerate(results["ids"][0]):
            item = {
                "id": doc_id,
                "document": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
            }
            formatted.append(item)
        return formatted


    # persistent client uses

    def drop_database(self):
        client.reset()

    # to remain client connected

    def heartbeat(self):
        client.client.heartbeat()

