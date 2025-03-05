# Embedding wrapper implementation on fastembed
from fastembed import TextEmbedding

class Embedder:
    """
    A wrapper class for text embedding functionality using the fastembed library.
    Provides a simplified interface for generating embeddings from text chunks.
    """
    def __init__(self):
        self.embedder = TextEmbedding()

    def embed(self, chunks):
        """
        Embed the text chunks
        """
        return self.embedder.embed(chunks)

