import os
import json
import ijson
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataProcessor:
    """Processes project data for vector store creation with robust error handling"""
    
    def __init__(self, settings):
        self.settings = settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
    
    def stream_large_json(self, file_path):
        """Stream large JSON files without loading entire file into memory"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Parse array of objects
                objects = ijson.items(f, 'item')
                for obj in objects:
                    yield obj
        except Exception as e:
            print(f"Error streaming JSON from {file_path}: {e}")
            # Fallback to standard loading
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        yield item
            except Exception as e:
                print(f"Error loading JSON with fallback method: {e}")
                return
    
    def _safely_convert_metadata(self, metadata, project_id, source_type):
        """Safely convert metadata to a dictionary with essential fields"""
        safe_metadata = {
            "project_id": project_id,
            "source": source_type
        }
        
        # Handle different metadata types
        if isinstance(metadata, dict):
            # Copy only simple types from the original metadata
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    safe_metadata[k] = v
        elif isinstance(metadata, tuple):
            # Convert tuple to dictionary
            if len(metadata) % 2 == 0:
                # If it's key-value pairs
                for i in range(0, len(metadata), 2):
                    if i+1 < len(metadata) and isinstance(metadata[i], str):
                        safe_metadata[metadata[i]] = metadata[i+1]
            else:
                # If it's a single tuple, use index-based keys
                for i, value in enumerate(metadata):
                    safe_metadata[f"metadata_{i}"] = value
        elif metadata is not None:
            # For any other type, store as string
            safe_metadata["raw_metadata"] = str(metadata)
        
        return safe_metadata
    
    def _ensure_document(self, item, project_id, source_type):
        """Ensure an item is converted to a proper Document object with safe metadata"""
        # Case 1: Already a Document object
        if isinstance(item, Document):
            # Ensure metadata is a dictionary
            if not isinstance(item.metadata, dict):
                item.metadata = self._safely_convert_metadata(item.metadata, project_id, source_type)
            return item
        
        # Case 2: Tuple format (content, metadata)
        if isinstance(item, tuple) and len(item) == 2:
            page_content, metadata = item
            safe_metadata = self._safely_convert_metadata(metadata, project_id, source_type)
            return Document(
                page_content=page_content,
                metadata=safe_metadata
            )
        
        # Case 3: Dict with page_content and metadata
        if isinstance(item, dict) and "page_content" in item:
            safe_metadata = self._safely_convert_metadata(
                item.get("metadata", {}), 
                project_id, 
                source_type
            )
            return Document(
                page_content=item["page_content"],
                metadata=safe_metadata
            )
        
        # Case 4: FAQ format dict
        if isinstance(item, dict) and "question" in item and "answer" in item:
            safe_metadata = {
                "project_id": project_id,
                "source": source_type,
                "question": item["question"]
            }
            return Document(
                page_content=item["answer"],
                metadata=safe_metadata
            )
        
        # Case 5: KB format dict
        if isinstance(item, dict) and "article" in item and "content" in item:
            # Extract article ID and title
            article_parts = item["article"].split('\t')
            article_id = article_parts[0].strip() if article_parts else "unknown"
            article_title = article_parts[1].strip() if len(article_parts) > 1 else item["article"]
            
            safe_metadata = {
                "project_id": project_id,
                "source": source_type,
                "article_id": article_id,
                "article_title": article_title
            }
            return Document(
                page_content=item["content"],
                metadata=safe_metadata
            )
        
        # Case 6: Simple string content
        if isinstance(item, str):
            return Document(
                page_content=item,
                metadata={
                    "project_id": project_id,
                    "source": source_type
                }
            )
        
        # Case 7: Any other format - try to convert to string
        try:
            content = str(item)
            return Document(
                page_content=content,
                metadata={
                    "project_id": project_id,
                    "source": source_type
                }
            )
        except:
            print(f"Warning: Could not convert item to Document: {type(item)}")
            return None
    
    def process_faq_data(self, project_id):
        """Process FAQ data for a project with robust error handling"""
        faq_file = f"{project_id}/{project_id}.faq.json"
        if not os.path.exists(faq_file):
            return []
        
        documents = []
        try:
            # First try standard loading
            with open(faq_file, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
            
            for item in faq_data:
                doc = self._ensure_document(item, project_id, "faq")
                if doc:
                    documents.append(doc)
        except Exception as e:
            print(f"Error processing FAQ file {faq_file} with standard loading: {e}")
            # Fallback to streaming
            try:
                for item in self.stream_large_json(faq_file):
                    doc = self._ensure_document(item, project_id, "faq")
                    if doc:
                        documents.append(doc)
            except Exception as e:
                print(f"Error processing FAQ file {faq_file} with streaming: {e}")
        
        return documents
    
    def process_kb_content(self, project_id):
        """Process KB content with hierarchical chunking and robust error handling"""
        kb_file = f"{project_id}/{project_id}.kb.json"
        if not os.path.exists(kb_file):
            return []
        
        documents = []
        try:
            # First try standard loading
            with open(kb_file, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            for item in kb_data:
                # Convert to Document first
                doc = self._ensure_document(item, project_id, "kb")
                if not doc:
                    continue
                
                # Split content into chunks
                chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    # Extract article ID and title from original metadata
                    article_id = doc.metadata.get("article_id", "unknown")
                    article_title = doc.metadata.get("article_title", "Unknown Article")
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "project_id": project_id,
                            "source": "kb",
                            "article_id": article_id,
                            "article_title": article_title,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    ))
        except Exception as e:
            print(f"Error processing KB file {kb_file} with standard loading: {e}")
            # Fallback to streaming
            try:
                for item in self.stream_large_json(kb_file):
                    # Convert to Document first
                    doc = self._ensure_document(item, project_id, "kb")
                    if not doc:
                        continue
                    
                    # Split content into chunks
                    chunks = self.text_splitter.split_text(doc.page_content)
                    for i, chunk in enumerate(chunks):
                        # Extract article ID and title from original metadata
                        article_id = doc.metadata.get("article_id", "unknown")
                        article_title = doc.metadata.get("article_title", "Unknown Article")
                        
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "project_id": project_id,
                                "source": "kb",
                                "article_id": article_id,
                                "article_title": article_title,
                                "chunk_id": i,
                                "total_chunks": len(chunks)
                            }
                        ))
            except Exception as e:
                print(f"Error processing KB file {kb_file} with streaming: {e}")
        
        return documents
    
    def process_project_data(self, project_id):
        """Process both FAQ and KB data for a project with robust error handling"""
        faq_docs = self.process_faq_data(project_id)
        kb_docs = self.process_kb_content(project_id)
        
        # Filter out any None values
        all_docs = [doc for doc in faq_docs + kb_docs if doc is not None]
        
        print(f"Processed {len(all_docs)} documents for project {project_id} "
              f"({len(faq_docs)} FAQ, {len(kb_docs)} KB)")
        
        return all_docs