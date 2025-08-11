import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # Core settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # DARKBO home directory
    DARKBO_HOME = os.getenv("DARKBO_HOME", str(Path.home()))
    
    # API settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Project identification
    PROJECT_SIMILARITY_THRESHOLD = float(os.getenv("PROJECT_SIMILARITY_THRESHOLD", "0.65"))
    PROJECT_AMBIGUITY_THRESHOLD = float(os.getenv("PROJECT_AMBIGUITY_THRESHOLD", "0.85"))
    MAX_PROJECT_SUGGESTIONS = int(os.getenv("MAX_PROJECT_SUGGESTIONS", "3"))
    
    # Vector store and embedding
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_stores")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # RAG settings
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    FAQ_THRESHOLD = float(os.getenv("FAQ_THRESHOLD", "0.85"))
    
    # Text splitting
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))
    
    # File paths
    PROJECT_MAPPING_FILE = os.getenv("PROJECT_MAPPING_FILE", "proj_mapping.txt")
    
    # Retrieval settings
    MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "3"))
    MAX_SECTIONS_PER_ARTICLE = int(os.getenv("MAX_SECTIONS_PER_ARTICLE", "5"))
    SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "20"))
    
    # Active projects
    MAX_ACTIVE_PROJECTS = int(os.getenv("MAX_ACTIVE_PROJECTS", "5"))
    
    # Hybrid search weights
    SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.6"))
    DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.4"))
    
    def validate(self):
        """Validate required settings"""
        # OpenAI API key is optional now since we might use other models
        if self.LLM_MODEL.startswith("gpt") and not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set but using OpenAI model")
        
        # Ensure DARKBO_HOME exists
        darkbo_path = Path(self.DARKBO_HOME)
        darkbo_path.mkdir(exist_ok=True)
        
        return True