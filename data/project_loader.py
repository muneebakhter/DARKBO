import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

class ProjectLoader:
    """Handles project mapping and identification with enhanced ambiguity handling"""
    
    def __init__(self, settings):
        self.settings = settings
        self.projects = {}  # {project_id: project_name}
        self.project_embeddings = None
        self.embedding_model = None
        
        # Load project mapping
        self._load_project_mapping()
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Create project embeddings
        if self.projects:
            self._create_project_embeddings()
    
    def _load_project_mapping(self):
        """Load project mapping from file with robust error handling"""
        mapping_file = Path(self.settings.PROJECT_MAPPING_FILE)
        if not mapping_file.exists():
            raise FileNotFoundError(f"Project mapping file not found: {self.settings.PROJECT_MAPPING_FILE}")
        
        self.projects = {}
        for line in mapping_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
                
            # Handle multiple tabs by splitting only on first tab
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
                
            project_id = parts[0].strip()
            project_name = parts[1].strip()
            self.projects[project_id] = project_name
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        self.embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL)
    
    def _create_project_embeddings(self):
        """Create embeddings for all project names"""
        project_names = list(self.projects.values())
        self.project_embeddings = self.embedding_model.encode(project_names, convert_to_tensor=True)
    
    def identify_project(self, query):
        """
        Identify the most relevant project for a query with enhanced ambiguity handling
        
        Returns:
            {
                "status": "selected" | "ambiguous" | "not_found",
                "project_id": str | None,
                "project_name": str | None,
                "confidence": float,
                "suggestions": list[{"id": str, "name": str, "similarity": float}]
            }
        """
        if not self.projects:
            return {
                "status": "not_found",
                "project_id": None,
                "project_name": None,
                "confidence": 0.0,
                "suggestions": []
            }
        
        # Encode the query
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        
        # Calculate cosine similarities
        cos_scores = util.cos_sim(query_embedding, self.project_embeddings)[0]
        
        # Get top matches with their similarities
        top_matches = [
            (idx, self.projects[list(self.projects.keys())[idx]], cos_scores[idx].item())
            for idx in torch.topk(cos_scores, k=min(10, len(self.projects)))[1]
        ]
        
        # Filter by threshold
        valid_matches = [(idx, name, sim) for idx, name, sim in top_matches 
                         if sim >= self.settings.PROJECT_SIMILARITY_THRESHOLD]
        
        if not valid_matches:
            return {
                "status": "not_found",
                "project_id": None,
                "project_name": None,
                "confidence": 0.0,
                "suggestions": []
            }
        
        # Get the best match
        best_idx, best_name, best_sim = valid_matches[0]
        project_id = list(self.projects.keys())[best_idx]
        
        # Check for ambiguity
        ambiguous_matches = [
            {"id": list(self.projects.keys())[idx], "name": name, "similarity": sim}
            for idx, name, sim in valid_matches[1:]
            if sim > self.settings.PROJECT_AMBIGUITY_THRESHOLD * best_sim
        ]
        
        # Format suggestions
        suggestions = [
            {"id": list(self.projects.keys())[idx], "name": name, "similarity": sim}
            for idx, name, sim in valid_matches[:self.settings.MAX_PROJECT_SUGGESTIONS]
        ]
        
        # Determine status
        if len(ambiguous_matches) > 0:
            status = "ambiguous"
        else:
            status = "selected"
        
        return {
            "status": status,
            "project_id": project_id if status == "selected" else None,
            "project_name": best_name if status == "selected" else None,
            "confidence": best_sim,
            "suggestions": suggestions
        }
    
    def get_project_name(self, project_id):
        """Get project name for a project ID"""
        return self.projects.get(project_id, "Unknown Project")
    
    def get_all_projects(self):
        """Get all projects as a list of (id, name) tuples"""
        return [{"id": pid, "name": name} for pid, name in self.projects.items()]