import faiss
import numpy as np
from utils.embedding_model import model

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Split text into overlapping chunks using character length and space boundaries.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            # try to find a space to not break words
            while end > start and text[end] != ' ':
                end -= 1
            if end == start: # fallback if no space found
                end = start + chunk_size
        
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= text_len or end >= text_len:
            break
            
    # Filter empty chunks
    return [c for c in chunks if c]

def create_vector_store(chunks: list):
    """
    Create a FAISS vector index from text chunks.
    Returns the index.
    """
    if not chunks:
        return None
        
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index

def retrieve_context(query: str, index, chunks: list, top_k: int = 3) -> list:
    """
    Retrieve top_k relevant chunks from the FAISS index for the given query.
    """
    if not index or not chunks:
        return []
        
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), min(top_k, len(chunks)))
    
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return retrieved_chunks
