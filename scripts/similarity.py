import numpy as np

def similarity_score(query_embedding, vector):
    """
    Calculate the cosine similarity between a query embedding and a vector
    """
    return np.dot(query_embedding, vector) / (np.linalg.norm(query_embedding) * np.linalg.norm(vector))
