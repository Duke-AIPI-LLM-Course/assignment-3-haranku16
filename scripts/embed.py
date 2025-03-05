# Embedding wrapper implementation on fastembed
from fastembed import TextEmbedding

class Embedder:
    def __init__(self):
        self.embedder = TextEmbedding()

    def embed(self, chunks):
        """
        Embed the text chunks
        """
        return self.embedder.embed(chunks)

