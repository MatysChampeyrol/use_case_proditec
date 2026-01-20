from abc import ABC, abstractmethod


class BaseService(ABC):
    @abstractmethod
    def extract_data(self):
        pass

    @abstractmethod
    def proceed_data(self, extract_data):
        pass
