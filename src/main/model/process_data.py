class ProcessData:
    def __init__(self, page_content, metadata :dict = None):
        self._page_content=page_content
        self._metadata = metadata

    @property
    def page_content(self):
        return self._page_content

    @page_content.setter
    def page_content(self, value):
        self._page_content = value

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
    
