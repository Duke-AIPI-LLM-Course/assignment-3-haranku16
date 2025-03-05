from openai import OpenAI

# RAG system implementation
class RetrievalAugmentedGenerator:
    """
    A class that implements Retrieval Augmented Generation using OpenAI's API and a vector database.
    This class combines document retrieval with language model generation to provide context-aware responses.
    """
    def __init__(self, db):
        """
        Initialize the RAG system.
        
        Args:
            db: A vector database instance used for document retrieval
        """
        self.client = OpenAI()
        self.db = db

    def generate(self, query):
        """
        Generate a response to a query using retrieved context from the vector database.
        
        Args:
            query (str): The user's question or query
            
        Returns:
            dict: A dictionary containing:
                - response (str): The generated response from the language model
                - context (str): The retrieved context used to generate the response
        """
        # Retrieve relevant chunks from the database
        chunks = list(self.db.search(query))
        
        # Filter chunks by similarity and limit to top 50
        filtered_chunks = [(sim, chunk) for sim, chunk in chunks if sim > 0.5][:50]
        
        # Format the context for the model
        context = "\n".join([f"Chunk {i+1} with similarity {similarity}:\n{chunk}" 
                           for i, (similarity, chunk) in enumerate(filtered_chunks)])

        # Generate the response using OpenAI's API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ]
        )

        # Return both the response and context
        return {
            'response': response.choices[0].message.content,
            'context': context
        }
