import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # Core settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Project identification
    PROJECT_SIMILARITY_THRESHOLD = float(os.getenv("PROJECT_SIMILARITY_THRESHOLD", "0.65"))
    PROJECT_AMBIGUITY_THRESHOLD = float(os.getenv("PROJECT_AMBIGUITY_THRESHOLD", "0.85"))
    MAX_PROJECT_SUGGESTIONS = int(os.getenv("MAX_PROJECT_SUGGESTIONS", "3"))
    
    # Vector store
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_stores")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # RAG
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    # Text splitting
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # File paths
    PROJECT_MAPPING_FILE = os.getenv("PROJECT_MAPPING_FILE", "proj_mappings.txt")
    
    # Two-stage retrieval
    MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "3"))
    MAX_SECTIONS_PER_ARTICLE = int(os.getenv("MAX_SECTIONS_PER_ARTICLE", "5"))
    
    # Active projects
    MAX_ACTIVE_PROJECTS = int(os.getenv("MAX_ACTIVE_PROJECTS", "5"))
    
    def validate(self):
        """Validate required settings"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")