
from service.database_vect_service.database_vect_service import  DatabaseVectService
from service.llm_service.llm_service import LlmService
import os
from service.document_service.pdf_service import PdfService
import io
from service.bdd_service.bdd_service import PostgresService
from service.chunk_service.chunk_service import ChunkService
from service.embedding_service.embedding_service import EmbeddingService
import json
from model.config import Config
from constant import rag_constant
import json
from fastapi import UploadFile


def load_file(file):
    path = f"{os.getcwd()}"
    with open(f"{path}/{file}", 'r', encoding='utf-8') as read_file:
        return json.load(read_file)
        

class rag_service():
    def __init__(self):  #remettre src/main
        self.config = Config(load_file("src/main/config/config.json"))

        # services declaration
        self.chunk_service = ChunkService(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.database_vect_service = DatabaseVectService(self.config) 
        self.llm_service = LlmService(self.config)

    
    def upload(self, file):    
        filename = file.filename
        ## on crée une collection chroma
        collection = self.database_vect_service.get_or_create_collection(filename)
        
        document_to_load = PdfService(file, self.config)

        ##On extrait la donnée du pdf
        extract = document_to_load.extract_data()
        
        ##Contient une liste de ProcessData (page_content, metadata) les éléments de la liste correspondent aux pages du pdf
        proceed = document_to_load.proceed_data(extract)
        ## chunk media
        document_chunked = self.chunk_service.chunk(proceed, rag_constant.CHUNK_SIZE,rag_constant.OVERLAP)
        print(f"document chunked :{document_chunked}\n")
        # embed media
        document_embedded = self.embedding_service.embedding_bge_multilingual(document_chunked)

        ## store in db vect
        self.database_vect_service.collection_store_embedded_document(collection, document_chunked, document_embedded)
        

    def ask(self, question):
         # access collection
        collection = self.database_vect_service.get_or_create_collection("default_database")

        # (facultatif) Question filter

        # Chunk question
        question_chunked = self.chunk_service.chunk(question, rag_constant.CHUNK_SIZE, rag_constant.OVERLAP)

        # Embedding question
        embedded_question = self.embedding_service.embedding_bge_multilingual(question_chunked)

        # similarity query
        similarity_query = self.database_vect_service.query(collection, embedded_question, rag_constant.NUMBER_RESULTS)

        #TODO reorganisation of the data bejore inject to the llm
        

        # print llm response
        llm_completion = self.llm_service.rag_completion(question, similarity_query)
        print(llm_completion)
        