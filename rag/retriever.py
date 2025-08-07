from langchain_core.callbacks import CallbackManagerForRetrieverRun, CallbackManager
from typing import List, Optional
from langchain.schema import Document

class TwoStageRetriever:
    """Two-stage retriever for efficient handling of large KB files"""
    
    def __init__(self, vector_store, project_id, settings):
        self.vector_store = vector_store
        self.project_id = project_id
        self.settings = settings
    
    def _get_article_level_results(self, query, k=None, run_manager: Optional[CallbackManagerForRetrieverRun] = None):
        """Get relevant articles (first chunks only)"""
        if k is None:
            k = self.settings.MAX_ARTICLES
            
        # Filter to get the first chunk of each KB article or any FAQ entry
        # Matches either chunk_id == 0 or source == "faq"
        filter_dict = {
            "$and": [
                {"project_id": {"$eq": self.project_id}},
                {
                    "$or": [
                        {"chunk_id": {"$eq": 0}},
                        {"source": {"$eq": "faq"}}
                    ]
                }
            ]
        }
        
        return self.vector_store.similarity_search(
            query, 
            k=k,
            filter=filter_dict
        )
    
    def _get_section_level_results(self, query, article_ids, k=None, run_manager: Optional[CallbackManagerForRetrieverRun] = None):
        """Get relevant sections from specified articles"""
        if k is None:
            k = self.settings.MAX_SECTIONS_PER_ARTICLE
            
        # Filter to get sections from specific articles
        # Updated to use proper ChromaDB filter syntax
        filter_dict = {
            "$and": [
                {"project_id": {"$eq": self.project_id}},
                {"article_id": {"$in": article_ids}}
            ]
        }
        
        return self.vector_store.similarity_search(
            query, 
            k=k,
            filter=filter_dict
        )
    
    def get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        """Retrieve relevant documents using two-stage approach"""
        # Stage 1: Find relevant articles or FAQ entries
        article_results = self._get_article_level_results(query, run_manager=run_manager)

        if not article_results:
            return []

        # Separate FAQ docs from KB article docs
        faq_docs = [doc for doc in article_results if doc.metadata.get("source") == "faq"]
        kb_article_ids = [doc.metadata["article_id"] for doc in article_results if doc.metadata.get("source") != "faq" and "article_id" in doc.metadata]

        final_results = list(faq_docs)

        # Stage 2: Retrieve relevant sections from KB articles
        if kb_article_ids:
            article_ids = list(set(kb_article_ids))
            section_results = self._get_section_level_results(query, article_ids, run_manager=run_manager)
            final_results.extend(section_results)

        return final_results
    
    async def aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        """Asynchronous version of get_relevant_documents"""
        return self.get_relevant_documents(query, run_manager=run_manager)
    
    def test_retrieval(self, query: str) -> List[Document]:
        """Helper method for testing retrieval without requiring run_manager"""
        return self.get_relevant_documents(query)