import numpy as np
from sentence_transformers import util

def calculate_similarity(query, documents, embedding_model):
    """Calculate cosine similarity between query and documents"""
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)
    return util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

def format_project_suggestions(suggestions, max_suggestions=3):
    """Format project suggestions for display"""
    if not suggestions:
        return "No projects found"
    
    # Sort by similarity (descending)
    sorted_suggestions = sorted(suggestions, key=lambda x: x["similarity"], reverse=True)
    
    # Format the top suggestions
    formatted = []
    for i, suggestion in enumerate(sorted_suggestions[:max_suggestions]):
        formatted.append(f"{suggestion['name']} (ID: {suggestion['id']})")
    
    return ", ".join(formatted)