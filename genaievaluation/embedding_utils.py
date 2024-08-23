# genaievaluation/genaievaluation/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using the provided model.
    
    Args:
        texts (list of str): List of texts to generate embeddings for.
    
    Returns:
        np.ndarray: Array of embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings