import os
import logging
import json
from typing import Optional
# from model.config import Config
from src.main.model.config import Config
from src.main.service.embedding_service.embedding_service import EmbeddingService
# from src.main.service.database_vect_service.database_vect_service import DatabaseVectService
# from src.main.service.document_service import PdfService
from ollama import chat
from langchain_ollama import OllamaLLM


class LlmService:
    def __init__(self):
        # self.config = config
        # local llm model initialization 
        logging.info("LLM Service initialized with config.")
        self.llm_model = self.initialize_llm_model()

    def rag_completion(self, question, similarity_query: Optional[list] = None) -> str:
        """
        Generate a response to the question using RAG approach with optional similarity context.
        """
        prompt = self.build_prompt(question, similarity_query)
        logging.info(f"Generated prompt for LLM: {prompt}")

        response = self.llm_model.invoke(prompt)
        logging.info(f"LLM response: {response}")

        return response

    def initialize_llm_model(self):    
        return OllamaLLM(
            model="mistral",
            temperature=0.7,
        ) 

    def build_prompt(self, question: str, similarity_query: Optional[list] = None) -> str:
        """
        Build the prompt for the LLM using the question and optional similarity context.
        """
        context = ""
        if similarity_query:
            context = "\n".join([f"Context {i+1}: {item['page_content']}" for i, item in enumerate(similarity_query)])
        
        prompt = f"""Tu es expert technique d'un service support client. Réponds à la question en te basant uniquement sur le CONTEXTE TECHNIQUE fourni. Réponds en français. Si l'information manque, admets-le.
        
        {context}
        
        Question: {question}
        
        Answer:"""
        return prompt
    


# example usage
if __name__ == "__main__":
    #config = Config()
    llm_service = LlmService()

    question = "What is the capital of France?"
    response = llm_service.rag_completion(question)
    print("Response:", response)