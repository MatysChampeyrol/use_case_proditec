class Extractdata:
    def __init__(self, data, data_type, file_name=None):
        self._extract_data=data
        self._data_type = data_type
        self._file_name = file_name

    @property
    def extract_data(self):
        return self._extract_data

    @extract_data.setter
    def extract_data(self, value):
        self._extract_data = value

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        self._data_type = value

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        self._file_name = value
    
