import os

class Config:
    def __init__(self, jsonData):
        self._access_token = os.getenv("OVH_API_KEY") if os.getenv("OVH_API_KEY") else jsonData['access-token']
        
        self._url_model_llm = jsonData['url_model_llm']
        self._model_llm = jsonData['model_llm']
        self._max_token_llm = jsonData['max_token_llm']
        self._temperature_llm = jsonData['temperature_llm']
        
        self._url_embedding_model = jsonData['url_embedding_model']

    # access_token
    @property
    def access_token(self):
        return self._access_token

    @access_token.setter
    def access_token(self, value):
        self._access_token = value

    # url_model_llm
    @property
    def url_model_llm(self):
        return self._url_model_llm

    @url_model_llm.setter
    def url_model_llm(self, value):
        self._url_model_llm = value

    # model_llm
    @property
    def model_llm(self):
        return self._model_llm

    @model_llm.setter
    def model_llm(self, value):
        self._model_llm = value

    # max_token_llm
    @property
    def max_token_llm(self):
        return self._max_token_llm

    @max_token_llm.setter
    def max_token_llm(self, value):
        self._max_token_llm = value

    # temperature_llm
    @property
    def temperature_llm(self):
        return self._temperature_llm

    @temperature_llm.setter
    def temperature_llm(self, value):
        self._temperature_llm = value
        
    # url_embedding_model
    @property
    def url_embedding_model(self):
        return self._url_embedding_model

    @url_embedding_model.setter
    def url_embedding_model(self, value):
        self._url_embedding_model = value
