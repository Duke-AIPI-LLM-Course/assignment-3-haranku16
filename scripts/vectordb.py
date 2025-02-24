# Vector database implementation
import numpy as np
import sqlite3
import os
import uuid

from scripts.embed import Embedder
from scripts.cosine_similarity import cosine_similarity

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class VectorDatabase:
    def __init__(self, db_path="vector_store.db"):
        """
        Initialize the vector database
        Args:
            db_path: Path to SQLite database file
        """
        # Create data directory structure
        self.data_dir = "data"
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.chunks_dir = os.path.join(self.data_dir, "chunks")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)

        # Initialize embedder
        self.embedder = Embedder()

        # Setup for database connection in __enter__
        self.db_path = db_path
        self.connection = None
        self.cursor = None
    
    def __enter__(self):
        """
        Enter the context of the vector database
        """
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        # Create VECTORS table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS VECTORS (
                id TEXT PRIMARY KEY,  -- filename#chunk_id
                filename TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                embedding TEXT
            )
        ''')
        self.connection.commit()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback): 
        """
        Exit the context of the vector database
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.commit()
            self.connection.close()
        self.cursor = None
        self.connection = None

    def put(self, filename, text):
        """
        Put the text into the vector database
        Args:
            filename: Name of the file to store
            text: Text content to store and chunk
        """
        # Store original text in raw directory
        raw_path = os.path.join(self.raw_dir, filename)
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # Chunk the text
        chunks = self.__chunk(text)

        embeddings = self.embedder.embed(chunks)

        # Store chunks with UUID filenames and vector embeddings into SQLite
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.txt")
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # Generate and store vector embeddings
            vector_id = f"{filename}#{chunk_id}"
            self.cursor.execute('''
                INSERT INTO VECTORS (id, filename, chunk_id, embedding)
                VALUES (?, ?, ?, ?)
            ''', (vector_id, filename, chunk_id, np.array2string(embedding, separator=',', precision=3, suppress_small=True)))

    def delete(self, filename):
        """
        Delete the text and its chunks from the vector database
        Args:
            filename: Name of the file to delete
        """
        # Get all chunk_ids for this filename from the VECTORS table
        self.cursor.execute('SELECT chunk_id FROM VECTORS WHERE filename = ?', (filename,))
        chunk_ids = [row[0] for row in self.cursor.fetchall()]
        
        # Delete chunks from filesystem
        for chunk_id in chunk_ids:
            chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.txt")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        
        # Delete raw file
        raw_path = os.path.join(self.raw_dir, filename)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        
        # Delete all vectors for this file from the database
        self.cursor.execute('DELETE FROM VECTORS WHERE filename = ?', (filename,))
        self.connection.commit()
    
    def search(self, query, k=10):
        """
        Search the vector database for the query and returns the top k results
        Args:
            query: Query to search for
            k: Number of results to return
        """
        # Embed the query
        query_embedding = [ embedding for embedding in self.embedder.embed([query]) ][0]

        # Get all vectors from the database and calculate cosine similarity with the query embedding
        self.cursor.execute('SELECT chunk_id, embedding FROM VECTORS')
        rows = [row for row in self.cursor.fetchall()]
        embeddings = [np.fromstring(row[1], sep=',') for row in rows]
        similarities = [cosine_similarity(query_embedding, vector) for vector in embeddings]

        # Get the top k results
        top_k_results = sorted(zip(similarities, rows), key=lambda x: x[0], reverse=True)[:k]
        
        # Get the chunks from the filesystem
        chunks = []
        for similarity, row in top_k_results:
            chunk_id = row[0]
            chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.txt")
            with open(chunk_path, 'r', encoding='utf-8') as f:
                chunks.append(f.read())

        return chunks

    def __chunk(self, text):
        """
        Chunk the text into smaller pieces with overlap
        """
        words = text.split()  # Split by all whitespace characters
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunks.append(" ".join(words[i:i+CHUNK_SIZE]))
        return chunks
