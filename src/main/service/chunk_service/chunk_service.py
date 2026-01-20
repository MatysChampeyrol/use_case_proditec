from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkService():
      def __init__(self, config):
            self.config = config

      def chunk(self, pages, size, overlap):
            # Split text in chunk
            print("try to split file...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=size,
                                                      chunk_overlap=overlap,
                                                      separators=["\n\n", "\n", " "])
            if isinstance(pages, str) :
                  # Pour les questions
                  splitted_chunks = splitter.split_text(pages)
            else:
                  # Pour les documents
                  splitted_chunks = splitter.split_documents(pages)
            print("OK, Number of chunks : " + str(len(splitted_chunks)))
            print(f"Result of chunking : {splitted_chunks}")
            return splitted_chunks



