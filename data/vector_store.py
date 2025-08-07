import os
import shutil
from functools import lru_cache
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document

class VectorStoreManager:
    """Manages vector stores for multiple projects with memory optimization"""
    
    def __init__(self, settings, project_loader):
        self.settings = settings
        self.project_loader = project_loader
        self.project_stores = {}
        self.active_projects = []
        
        # Ensure vector store directory exists
        os.makedirs(self.settings.VECTOR_STORE_DIR, exist_ok=True)
        
        # Initialize embedding model
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL
        )
    
    @lru_cache(maxsize=5)  # Cache most recently used vector stores
    def get_vector_store(self, project_id):
        """Get or create a vector store for a project"""
        # Check if we already have this store loaded
        if project_id in self.project_stores:
            return self.project_stores[project_id]
        
        # Check if vector store already exists on disk
        store_path = os.path.join(self.settings.VECTOR_STORE_DIR, project_id)
        if os.path.exists(store_path):
            try:
                vector_store = Chroma(
                    persist_directory=store_path,
                    embedding_function=self.embeddings
                )
                self.project_stores[project_id] = vector_store
                self._update_active_projects(project_id)
                return vector_store
            except Exception as e:
                print(f"Error loading existing vector store for {project_id}: {e}")
        
        # Create new vector store
        return self.create_vector_store(project_id)
    
    def create_vector_store(self, project_id):
        """Create a new vector store for a project with comprehensive error handling"""
        from data.data_processor import DataProcessor
        
        print(f"\nCreating vector store for project {project_id}...")
        
        # Process project data
        try:
            data_processor = DataProcessor(self.settings)
            documents = data_processor.process_project_data(project_id)
        except Exception as e:
            print(f"Error processing project data for {project_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if not documents:
            print(f"No documents found for project {project_id}")
            return None
        
        # Ensure all items are Document objects
        valid_docs = []
        for doc in documents:
            if not isinstance(doc, Document):
                print(f"Warning: Skipping non-Document object in project {project_id}: {type(doc)}")
                continue
            
            # Ensure metadata is a dictionary
            if not isinstance(doc.metadata, dict):
                print(f"Warning: Document metadata is not a dict in project {project_id}: {type(doc.metadata)}")
                
                # Handle different metadata types
                if isinstance(doc.metadata, tuple) and len(doc.metadata) == 2:
                    # If it's a (key, value) tuple pair
                    try:
                        doc.metadata = {doc.metadata[0]: doc.metadata[1]}
                    except:
                        doc.metadata = {"project_id": project_id, "source": "unknown"}
                elif isinstance(doc.metadata, tuple):
                    # If it's a tuple of items, create a numbered dictionary
                    try:
                        doc.metadata = {f"metadata_{i}": item for i, item in enumerate(doc.metadata)}
                        doc.metadata["project_id"] = project_id
                        doc.metadata["source"] = "unknown"
                    except:
                        doc.metadata = {"project_id": project_id, "source": "unknown"}
                else:
                    # For any other non-dict type
                    doc.metadata = {"project_id": project_id, "source": "unknown", "raw_metadata": str(doc.metadata)}
            
            valid_docs.append(doc)
        
        if not valid_docs:
            print(f"No valid Document objects after filtering for project {project_id}")
            return None
        
        # Filter out complex metadata
        filtered_docs = []
        for doc in valid_docs:
            try:
                # Make sure metadata is a dictionary
                if not isinstance(doc.metadata, dict):
                    doc.metadata = {"project_id": project_id, "source": "unknown"}
                
                # Make a copy of metadata to avoid modifying the original
                safe_metadata = {}
                for k, v in doc.metadata.items():
                    try:
                        # Only keep simple types
                        if isinstance(v, (str, int, float, bool, type(None))):
                            safe_metadata[k] = v
                    except:
                        pass
                
                # Ensure essential metadata
                if "project_id" not in safe_metadata:
                    safe_metadata["project_id"] = project_id
                if "source" not in safe_metadata:
                    safe_metadata["source"] = "unknown"
                
                # Create a new Document with safe metadata
                filtered_docs.append(Document(
                    page_content=doc.page_content,
                    metadata=safe_metadata
                ))
            except Exception as e:
                print(f"Warning: Error processing document metadata: {e}")
                # Create a safe document with minimal metadata
                filtered_docs.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        "project_id": project_id,
                        "source": "unknown",
                        "error": str(e)
                    }
                ))
        
        if not filtered_docs:
            print(f"No documents after metadata filtering for project {project_id}")
            return None
        
        # Create vector store
        store_path = os.path.join(self.settings.VECTOR_STORE_DIR, project_id)
        if os.path.exists(store_path):
            try:
                shutil.rmtree(store_path)
            except Exception as e:
                print(f"Warning: could not remove existing store {store_path}: {e}")

        try:
            vector_store = Chroma.from_documents(
                documents=filtered_docs,
                embedding=self.embeddings,
                persist_directory=store_path
            )
            
            # Cache the vector store
            self.project_stores[project_id] = vector_store
            self._update_active_projects(project_id)
            
            print(f"Created vector store with {len(filtered_docs)} documents for project {project_id}")
            return vector_store
        except Exception as e:
            print(f"Error creating vector store for project {project_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _update_active_projects(self, project_id):
        """Update the list of active projects (LRU cache)"""
        if project_id in self.active_projects:
            self.active_projects.remove(project_id)
        self.active_projects.append(project_id)
        
        # Remove least recently used project if exceeding limit
        if len(self.active_projects) > self.settings.MAX_ACTIVE_PROJECTS:
            lru_project = self.active_projects.pop(0)
            if lru_project in self.project_stores:
                del self.project_stores[lru_project]
    
    def clear_inactive_stores(self):
        """Clear vector stores that are not in active projects"""
        for project_id in list(self.project_stores.keys()):
            if project_id not in self.active_projects:
                del self.project_stores[project_id]