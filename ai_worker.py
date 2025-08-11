#!/usr/bin/env python3
"""
Simplified AI Worker for DARKBO
Provides minimal endpoints for querying knowledge bases with sources.
"""

import os
import json
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Path as FastAPIPath
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    from whoosh.index import open_dir
    from whoosh.qparser import QueryParser
    import openai
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from api.models_simple import FAQEntry, KBEntry


# Pydantic models for API
class QueryRequest(BaseModel):
    project_id: str
    question: str

class Source(BaseModel):
    id: str
    type: str  # 'faq' or 'kb'
    title: str
    url: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    project_id: str
    timestamp: str


class KnowledgeBaseRetriever:
    """Retrieves information from prebuilt indexes"""
    
    def __init__(self, project_id: str, base_dir: str = "."):
        self.project_id = project_id
        self.base_dir = Path(base_dir)
        self.project_dir = self.base_dir / project_id
        self.index_dir = self.project_dir / "index"
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Initialize indexes
        self.dense_index = None
        self.dense_metadata = None
        self.sparse_index = None
        self.embedding_model = None
        
        self._load_indexes()
    
    def _load_metadata(self) -> Dict:
        """Load index metadata"""
        meta_file = self.index_dir / "meta.json"
        if not meta_file.exists():
            raise ValueError(f"No index metadata found for project {self.project_id}")
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_indexes(self):
        """Load prebuilt indexes"""
        if not HAS_DEPS:
            return
        
        try:
            # Load dense index
            if self.metadata.get('indexes', {}).get('dense', {}).get('available'):
                dense_dir = self.index_dir / "dense"
                index_file = dense_dir / "faiss.index"
                metadata_file = dense_dir / "metadata.json"
                
                if index_file.exists() and metadata_file.exists():
                    self.dense_index = faiss.read_index(str(index_file))
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.dense_metadata = json.load(f)
                    
                    # Load embedding model
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load sparse index
            if self.metadata.get('indexes', {}).get('sparse', {}).get('available'):
                sparse_dir = self.index_dir / "sparse"
                if sparse_dir.exists():
                    self.sparse_index = open_dir(str(sparse_dir))
                    
        except Exception as e:
            print(f"Warning: Could not load indexes for {self.project_id}: {e}")
    
    def search_dense(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using dense vector similarity"""
        if not self.dense_index or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.dense_index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.dense_metadata):
                    result = self.dense_metadata[idx].copy()
                    result['score'] = float(score)
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"Dense search error: {e}")
            return []
    
    def search_sparse(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using sparse text search"""
        if not self.sparse_index:
            return []
        
        try:
            with self.sparse_index.searcher() as searcher:
                parser = QueryParser("content", self.sparse_index.schema)
                parsed_query = parser.parse(query)
                
                results = []
                search_results = searcher.search(parsed_query, limit=top_k)
                
                for hit in search_results:
                    results.append({
                        'id': hit['id'],
                        'type': hit['type'],
                        'score': hit.score,
                        'question': hit.get('question', ''),
                        'answer': hit.get('answer', ''),
                        'article': hit.get('title', ''),
                        'content': hit.get('content', '')
                    })
                
                return results
        except Exception as e:
            print(f"Sparse search error: {e}")
            return []
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search combining dense and sparse results"""
        dense_results = self.search_dense(query, top_k)
        sparse_results = self.search_sparse(query, top_k)
        
        # If no ML-based search available, use basic text matching
        if not dense_results and not sparse_results:
            return self.search_basic(query, top_k)
        
        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []
        
        # Add dense results first (usually better for semantic similarity)
        for result in dense_results:
            if result['id'] not in seen_ids:
                result['search_type'] = 'dense'
                combined_results.append(result)
                seen_ids.add(result['id'])
        
        # Add sparse results
        for result in sparse_results:
            if result['id'] not in seen_ids:
                result['search_type'] = 'sparse'
                combined_results.append(result)
                seen_ids.add(result['id'])
        
        # Sort by score (descending)
        combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return combined_results[:top_k]
    
    def search_basic(self, query: str, top_k: int = 5) -> List[Dict]:
        """Basic text search fallback when dependencies aren't available"""
        try:
            # Load FAQ and KB data directly
            faqs, kb_entries = self._load_raw_data()
            
            results = []
            query_lower = query.lower()
            
            # Search in FAQs
            for faq in faqs:
                score = 0
                question_lower = faq.question.lower()
                answer_lower = faq.answer.lower()
                
                # Simple keyword matching
                for word in query_lower.split():
                    if word in question_lower:
                        score += 2  # Question matches get higher score
                    if word in answer_lower:
                        score += 1
                
                if score > 0:
                    results.append({
                        'id': faq.id,
                        'type': 'faq',
                        'score': score,
                        'question': faq.question,
                        'answer': faq.answer,
                        'search_type': 'basic'
                    })
            
            # Search in KB entries
            for kb in kb_entries:
                score = 0
                article_lower = kb.article.lower()
                content_lower = kb.content.lower()
                
                # Simple keyword matching
                for word in query_lower.split():
                    if word in article_lower:
                        score += 2  # Title matches get higher score
                    if word in content_lower:
                        score += 1
                
                if score > 0:
                    results.append({
                        'id': kb.id,
                        'type': 'kb',
                        'score': score,
                        'article': kb.article,
                        'content': kb.content,
                        'search_type': 'basic'
                    })
            
            # Sort by score (descending) and return top results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Basic search error: {e}")
            return []
    
    def _load_raw_data(self) -> tuple[List[FAQEntry], List[KBEntry]]:
        """Load raw FAQ and KB data for basic search"""
        faqs = []
        kb_entries = []
        
        # Load FAQ data
        faq_file = self.project_dir / f"{self.project_id}.faq.json"
        if faq_file.exists():
            with open(faq_file, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                faqs = [FAQEntry.from_dict(item) for item in faq_data]
        
        # Load KB data  
        kb_file = self.project_dir / f"{self.project_id}.kb.json"
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
                kb_entries = [KBEntry.from_dict(item) for item in kb_data]
        
        return faqs, kb_entries


class AIWorker:
    """Main AI worker class"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.projects = self._load_projects()
        self.retrievers = {}  # Cache retrievers
    
    def _load_projects(self) -> Dict[str, str]:
        """Load project mapping"""
        mapping_file = self.base_dir / "proj_mapping.txt"
        projects = {}
        
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        project_id, name = line.split('\t', 1)
                        projects[project_id.strip()] = name.strip()
        
        return projects
    
    def get_retriever(self, project_id: str) -> KnowledgeBaseRetriever:
        """Get or create retriever for project"""
        if project_id not in self.retrievers:
            if project_id not in self.projects:
                raise ValueError(f"Project {project_id} not found")
            
            self.retrievers[project_id] = KnowledgeBaseRetriever(project_id, str(self.base_dir))
        
        return self.retrievers[project_id]
    
    def get_faq_by_id(self, project_id: str, faq_id: str) -> Optional[FAQEntry]:
        """Get FAQ entry by ID"""
        faq_file = self.base_dir / project_id / f"{project_id}.faq.json"
        if not faq_file.exists():
            return None
        
        with open(faq_file, 'r', encoding='utf-8') as f:
            faqs_data = json.load(f)
            for faq_data in faqs_data:
                if faq_data['id'] == faq_id:
                    return FAQEntry.from_dict(faq_data)
        
        return None
    
    def get_kb_by_id(self, project_id: str, kb_id: str) -> Optional[KBEntry]:
        """Get KB entry by ID"""
        kb_file = self.base_dir / project_id / f"{project_id}.kb.json"
        if not kb_file.exists():
            return None
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
            for kb_item in kb_data:
                if kb_item['id'] == kb_id:
                    return KBEntry.from_dict(kb_item)
        
        return None
    
    def answer_question(self, project_id: str, question: str) -> QueryResponse:
        """Generate answer with sources"""
        # Get retriever
        retriever = self.get_retriever(project_id)
        
        # Search for relevant content
        search_results = retriever.search(question, top_k=5)
        
        # Generate sources
        sources = []
        for result in search_results:
            source_type = result.get('type', 'unknown')
            
            if source_type == 'faq':
                title = f"FAQ: {result.get('question', 'Unknown Question')}"
                url = f"/v1/projects/{project_id}/faqs/{result['id']}"
            else:
                title = result.get('article', 'Unknown Article')
                url = f"/v1/projects/{project_id}/kb/{result['id']}"
            
            sources.append(Source(
                id=result['id'],
                type=source_type,
                title=title,
                url=url,
                relevance_score=result.get('score', 0.0)
            ))
        
        # Generate answer (simple context-based approach)
        if search_results:
            # Use the top result for a simple answer
            top_result = search_results[0]
            
            if top_result.get('type') == 'faq':
                answer = top_result.get('answer', 'No answer available')
            else:
                # For KB entries, use content snippet
                content = top_result.get('content', '')
                # Truncate content to reasonable length
                if len(content) > 200:
                    answer = content[:200] + "..."
                else:
                    answer = content
            
            if not answer.strip():
                answer = "I found some relevant information, but couldn't extract a clear answer. Please check the sources below."
        else:
            answer = "I couldn't find relevant information to answer your question."
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            project_id=project_id,
            timestamp=datetime.utcnow().isoformat()
        )


# Initialize FastAPI app
app = FastAPI(
    title="DARKBO AI Worker",
    description="Simplified AI worker for querying knowledge bases",
    version="2.0.0"
)

# Initialize AI worker
worker = AIWorker()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DARKBO AI Worker",
        "description": "Simplified AI worker for querying knowledge bases",
        "version": "2.0.0",
        "endpoints": {
            "query": "POST /query",
            "faq": "GET /v1/projects/{project_id}/faqs/{faq_id}",
            "kb": "GET /v1/projects/{project_id}/kb/{kb_id}",
            "projects": "GET /projects"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/projects")
async def list_projects():
    """List available projects"""
    return {
        "projects": [
            {"id": pid, "name": name} 
            for pid, name in worker.projects.items()
        ]
    }


@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question with sources"""
    try:
        if request.project_id not in worker.projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        response = worker.answer_question(request.project_id, request.question)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/v1/projects/{project_id}/faqs/{faq_id}")
async def get_faq(
    project_id: str = FastAPIPath(..., description="Project ID"),
    faq_id: str = FastAPIPath(..., description="FAQ ID")
):
    """Get FAQ by ID, returns attachment file if available, otherwise JSON"""
    
    if project_id not in worker.projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check for attachment file first
    attachment_file = worker.base_dir / project_id / "attachments" / f"{faq_id}-faq.txt"
    if attachment_file.exists():
        return FileResponse(
            path=str(attachment_file),
            media_type="text/plain",
            filename=f"{faq_id}-faq.txt"
        )
    
    # Fall back to JSON
    faq = worker.get_faq_by_id(project_id, faq_id)
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    
    return faq.to_dict()


@app.get("/v1/projects/{project_id}/kb/{kb_id}")
async def get_kb(
    project_id: str = FastAPIPath(..., description="Project ID"),
    kb_id: str = FastAPIPath(..., description="KB ID")
):
    """Get KB entry by ID, returns attachment file if available, otherwise JSON"""
    
    if project_id not in worker.projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check for attachment files (multiple possible extensions)
    attachments_dir = worker.base_dir / project_id / "attachments"
    possible_files = [
        attachments_dir / f"{kb_id}-kb.txt",
        attachments_dir / f"{kb_id}-kb.docx", 
        attachments_dir / f"{kb_id}-kb.pdf"
    ]
    
    for attachment_file in possible_files:
        if attachment_file.exists():
            # Determine media type
            media_type, _ = mimetypes.guess_type(str(attachment_file))
            if not media_type:
                media_type = "application/octet-stream"
            
            return FileResponse(
                path=str(attachment_file),
                media_type=media_type,
                filename=attachment_file.name
            )
    
    # Fall back to JSON
    kb = worker.get_kb_by_id(project_id, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="KB entry not found")
    
    return kb.to_dict()


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting DARKBO AI Worker on {host}:{port}")
    print(f"📁 Base directory: {worker.base_dir}")
    print(f"📊 Projects loaded: {len(worker.projects)}")
    
    if not HAS_DEPS:
        print("⚠️  Some dependencies missing. Search will be limited.")
        print("💡 Install: pip install sentence-transformers faiss-cpu whoosh openai")
    
    uvicorn.run("ai_worker:app", host=host, port=port, reload=True)