# Vector database implementation
import numpy as np
import sqlite3
import os
import uuid
import pickle

from scripts.embed import Embedder
from scripts.similarity import similarity_score

CHUNK_SIZE = 1000

class VectorDatabase:
    def __init__(self, db_path="vector_store.db"):
        """
        Initialize the vector database with the specified database path.
        
        Args:
            db_path (str, optional): Path to SQLite database file. Defaults to "vector_store.db".
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
        Enter the context of the vector database, establishing database connection.
        
        Returns:
            VectorDatabase: The database instance with an active connection.
        """
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        # Create VECTORS table if it doesn't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS VECTORS (
                id TEXT PRIMARY KEY,  -- filename#chunk_id
                filename TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                embedding BLOB
            )
        ''')
        self.connection.commit()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context of the vector database, closing all connections.
        
        Args:
            exc_type: The type of the exception that was raised
            exc_value: The instance of the exception that was raised
            traceback: The traceback of the exception that was raised
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
        Store text content in the vector database with associated embeddings.
        
        Args:
            filename (str): Name of the file to store
            text (str): Text content to store and chunk
        """
        # Check if file already exists in database
        self.cursor.execute('SELECT COUNT(*) FROM VECTORS WHERE filename = ?', (filename,))
        if self.cursor.fetchone()[0] > 0:
            return  # Skip if file already exists

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
            ''', (vector_id, filename, chunk_id, pickle.dumps(embedding)))

    def delete(self, filename):
        """
        Remove a file and its associated chunks from the vector database.
        
        Args:
            filename (str): Name of the file to delete
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
    
    def search(self, query):
        """
        Search the vector database for content similar to the query.
        
        Args:
            query (str): Query text to search for
            
        Yields:
            tuple: A tuple containing:
                - float: Similarity score between the query and the chunk
                - str: The text content of the matching chunk
        """
        # Embed the query
        query_embedding = [ embedding for embedding in self.embedder.embed([query]) ][0]

        # Get all vectors from the database and calculate cosine similarity with the query embedding
        self.cursor.execute('SELECT chunk_id, embedding FROM VECTORS')
        rows = [row for row in self.cursor.fetchall()]
        embeddings = [pickle.loads(row[1]) for row in rows]
        similarities = [similarity_score(query_embedding, vector) for vector in embeddings]

        # Get the top k results
        sorted_results = sorted(zip(similarities, rows), key=lambda x: x[0], reverse=True)
        
        # Get the chunks from the filesystem
        for similarity, row in sorted_results:
            chunk_id = row[0]
            chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.txt")
            with open(chunk_path, 'r', encoding='utf-8') as f:
                yield similarity, f.read()

    def __chunk(self, text):
        """
        Split text into chunks based on paragraphs with intelligent combining and splitting.
        
        Args:
            text (str): The input text to be chunked
            
        Returns:
            list: A list of text chunks, each approximately CHUNK_SIZE words or less
            
        Notes:
            - Paragraphs < CHUNK_SIZE are kept as single chunks
            - Small paragraphs (<100 words) are combined with subsequent paragraphs
            - Paragraphs > CHUNK_SIZE are split into roughly equal chunks
        """
        # Split into paragraphs (handling different line ending styles)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_para = []
        current_word_count = 0
        
        def split_large_paragraph(paragraph):
            words = paragraph.split()
            total_words = len(words)
            # Calculate number of chunks needed
            num_chunks = (total_words + CHUNK_SIZE - 1) // CHUNK_SIZE
            # Calculate target size for roughly equal chunks
            target_size = total_words // num_chunks
            
            para_chunks = []
            for i in range(0, total_words, target_size):
                end_idx = min(i + target_size, total_words)
                para_chunks.append(' '.join(words[i:end_idx]))
            return para_chunks
        
        for para in paragraphs:
            para_words = para.split()
            word_count = len(para_words)
            
            # If we have accumulated paragraphs and the new one would exceed CHUNK_SIZE
            if current_para and (current_word_count + word_count > CHUNK_SIZE):
                chunks.append(' '.join(current_para))
                current_para = []
                current_word_count = 0
            
            # Handle the current paragraph
            if word_count < 100:  # Small paragraph
                current_para.append(para)
                current_word_count += word_count
            elif word_count <= CHUNK_SIZE:  # Medium paragraph
                if current_para:  # Flush any accumulated paragraphs
                    chunks.append(' '.join(current_para))
                    current_para = []
                    current_word_count = 0
                chunks.append(para)
            else:  # Large paragraph
                if current_para:  # Flush any accumulated paragraphs
                    chunks.append(' '.join(current_para))
                    current_para = []
                    current_word_count = 0
                chunks.extend(split_large_paragraph(para))
        
        # Add any remaining accumulated paragraphs
        if current_para:
            chunks.append(' '.join(current_para))
        
        return chunks
