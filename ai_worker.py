#!/usr/bin/env python3
"""
Simplified AI Worker for DARKBO
Provides minimal endpoints for querying knowledge bases with sources and external tools.
"""

import os
import json
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from fastapi import FastAPI, HTTPException, Path as FastAPIPath
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    from whoosh.index import open_dir
    from whoosh.qparser import QueryParser
    import openai
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from api.models import FAQEntry, KBEntry
from tools import ToolManager


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

class ToolUsage(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    execution_time: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    project_id: str
    timestamp: str
    tools_used: Optional[List[ToolUsage]] = None


class KnowledgeBaseRetriever:
    """Enhanced retriever with RRF, cross-encoder re-ranking, and chunking support"""
    
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
        self.cross_encoder = None
        
        self._load_indexes()
        self._load_cross_encoder()
    
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
            # Load dense index (check for HNSW first, then fallback to flat)
            if self.metadata.get('indexes', {}).get('dense', {}).get('available'):
                dense_dir = self.index_dir / "dense"
                
                # Try to load HNSW index first
                hnsw_index_file = dense_dir / "faiss_hnsw.index"
                flat_index_file = dense_dir / "faiss.index"
                metadata_file = dense_dir / "metadata.json"
                
                index_file = None
                if hnsw_index_file.exists():
                    index_file = hnsw_index_file
                    print(f"  Loading HNSW index for {self.project_id}")
                elif flat_index_file.exists():
                    index_file = flat_index_file
                    print(f"  Loading flat index for {self.project_id}")
                
                if index_file and metadata_file.exists():
                    self.dense_index = faiss.read_index(str(index_file))
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.dense_metadata = json.load(f)
                    
                    # Load appropriate embedding model
                    embedding_model_name = self.metadata.get('versions', {}).get('embedding_model')
                    if embedding_model_name and 'bge' in embedding_model_name.lower():
                        print(f"  Loading BGE embedding model...")
                        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
                    else:
                        print(f"  Loading MiniLM embedding model...")
                        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load sparse index
            if self.metadata.get('indexes', {}).get('sparse', {}).get('available'):
                sparse_dir = self.index_dir / "sparse"
                if sparse_dir.exists():
                    self.sparse_index = open_dir(str(sparse_dir))
                    
        except Exception as e:
            print(f"Warning: Could not load indexes for {self.project_id}: {e}")
    
    def _load_cross_encoder(self):
        """Load cross-encoder for re-ranking"""
        if not HAS_DEPS:
            return
            
        try:
            print(f"  Loading cross-encoder for re-ranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print(f"  ✅ Cross-encoder loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load cross-encoder: {e}")
            self.cross_encoder = None
    
    def search_dense(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using enhanced dense vector similarity with proper normalization"""
        if not self.dense_index or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Normalize query embedding consistently (L2 normalization)
            faiss.normalize_L2(query_embedding)
            
            # Search with more candidates for RRF
            scores, indices = self.dense_index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.dense_metadata):  # Check for valid indices
                    result = self.dense_metadata[idx].copy()
                    result['score'] = float(score)
                    result['rank'] = len(results) + 1  # Add rank for RRF
                    results.append(result)
            
            return results
        except Exception as e:
            print(f"Dense search error: {e}")
            return []
    
    def search_sparse(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using enhanced sparse text search"""
        if not self.sparse_index:
            return []
        
        try:
            with self.sparse_index.searcher() as searcher:
                parser = QueryParser("content", self.sparse_index.schema)
                parsed_query = parser.parse(query)
                
                results = []
                search_results = searcher.search(parsed_query, limit=top_k)
                
                for rank, hit in enumerate(search_results, 1):
                    results.append({
                        'id': hit['id'],
                        'original_id': hit.get('original_id', hit['id']),  # Support chunked results
                        'type': hit['type'],
                        'score': hit.score,
                        'rank': rank,  # Add rank for RRF
                        'question': hit.get('question', ''),
                        'answer': hit.get('answer', ''),
                        'article': hit.get('title', ''),
                        'content': hit.get('content', ''),
                        'chunk_index': hit.get('chunk_index', 0),
                        'total_chunks': hit.get('total_chunks', 1)
                    })
                
                return results
        except Exception as e:
            print(f"Sparse search error: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
        """Implement Reciprocal Rank Fusion (RRF) to combine dense and sparse results"""
        # RRF formula: score = 1 / (k + rank)
        # where k is a constant (typically 60) and rank is the position in the ranking
        
        rrf_scores = {}
        all_results = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result.get('original_id', result.get('id'))
            rank = result.get('rank', 1)
            rrf_score = 1.0 / (k + rank)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                all_results[doc_id] = result.copy()
                all_results[doc_id]['fusion_sources'] = []
            
            rrf_scores[doc_id] += rrf_score
            all_results[doc_id]['fusion_sources'].append('dense')
            all_results[doc_id]['dense_score'] = result.get('score', 0.0)
            all_results[doc_id]['dense_rank'] = rank
        
        # Process sparse results
        for result in sparse_results:
            doc_id = result.get('original_id', result.get('id'))
            rank = result.get('rank', 1)
            rrf_score = 1.0 / (k + rank)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                all_results[doc_id] = result.copy()
                all_results[doc_id]['fusion_sources'] = []
            
            rrf_scores[doc_id] += rrf_score
            all_results[doc_id]['fusion_sources'].append('sparse')
            all_results[doc_id]['sparse_score'] = result.get('score', 0.0)
            all_results[doc_id]['sparse_rank'] = rank
        
        # Sort by RRF score and prepare final results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            result = all_results[doc_id]
            result['rrf_score'] = rrf_score
            result['search_type'] = 'rrf'
            fused_results.append(result)
        
        return fused_results
    
    def _cross_encoder_rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Re-rank results using cross-encoder for better relevance"""
        if not self.cross_encoder or not results:
            return results[:top_k]
        
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for result in results:
                # Create document text from available content
                if result.get('type') == 'faq':
                    doc_text = f"Q: {result.get('question', '')} A: {result.get('answer', '')}"
                else:
                    article = result.get('article', '')
                    content = result.get('content', '')
                    doc_text = f"Title: {article} Content: {content}" if article else content
                
                query_doc_pairs.append([query, doc_text[:512]])  # Limit text length for cross-encoder
            
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Update results with cross-encoder scores
            for i, result in enumerate(results):
                result['ce_score'] = float(ce_scores[i])
                result['search_type'] = 'cross_encoder'
            
            # Sort by cross-encoder score
            results.sort(key=lambda x: x.get('ce_score', 0), reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Cross-encoder re-ranking error: {e}")
            # Fallback to RRF ranking
            return results[:top_k]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Enhanced hybrid search with RRF and cross-encoder re-ranking"""
        # Get more candidates for fusion (2x the final target)
        candidate_count = max(top_k * 2, 10)
        
        dense_results = self.search_dense(query, candidate_count)
        sparse_results = self.search_sparse(query, candidate_count)
        
        # If no ML-based search available, use basic text matching
        if not dense_results and not sparse_results:
            return self.search_basic(query, top_k)
        
        # Apply Reciprocal Rank Fusion
        if dense_results and sparse_results:
            print(f"  Applying RRF to {len(dense_results)} dense + {len(sparse_results)} sparse results")
            fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif dense_results:
            fused_results = dense_results
            for result in fused_results:
                result['search_type'] = 'dense_only'
        else:
            fused_results = sparse_results
            for result in fused_results:
                result['search_type'] = 'sparse_only'
        
        # Apply cross-encoder re-ranking to top candidates
        rerank_candidates = min(candidate_count, len(fused_results))
        final_results = self._cross_encoder_rerank(query, fused_results[:rerank_candidates], top_k)
        
        return final_results
    
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
                        'original_id': faq.id,
                        'type': 'faq',
                        'score': score,
                        'question': faq.question,
                        'answer': faq.answer,
                        'search_type': 'basic',
                        'chunk_index': 0,
                        'total_chunks': 1
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
                        'original_id': kb.id,
                        'type': 'kb',
                        'score': score,
                        'article': kb.article,
                        'content': kb.content,
                        'search_type': 'basic',
                        'chunk_index': 0,
                        'total_chunks': 1
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
        self.tool_manager = ToolManager()  # Initialize tool manager
    
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
    
    async def answer_question(self, project_id: str, question: str, use_tools: bool = True) -> QueryResponse:
        """Generate answer with sources and optional tool assistance"""
        # Get retriever
        retriever = self.get_retriever(project_id)
        
        # Search for relevant content in KB
        search_results = retriever.search(question, top_k=5)
        
        # Generate sources with better chunk handling
        sources = []
        seen_original_ids = set()  # Track original documents to avoid duplicates
        
        for result in search_results:
            original_id = result.get('original_id', result.get('id'))
            source_type = result.get('type', 'unknown')
            
            # Skip if we've already included this original document
            if original_id in seen_original_ids:
                continue
            seen_original_ids.add(original_id)
            
            if source_type == 'faq':
                title = f"FAQ: {result.get('question', 'Unknown Question')}"
                url = f"/v1/projects/{project_id}/faqs/{original_id}"
            else:
                title = result.get('article', 'Unknown Article')
                url = f"/v1/projects/{project_id}/kb/{original_id}"
            
            # Add chunking info to title if applicable
            chunk_info = ""
            if result.get('total_chunks', 1) > 1:
                chunk_idx = result.get('chunk_index', 0)
                total_chunks = result.get('total_chunks', 1)
                chunk_info = f" (chunk {chunk_idx + 1}/{total_chunks})"
            
            # Include search method info in title
            search_info = ""
            if result.get('search_type') == 'cross_encoder':
                search_info = f" [CE: {result.get('ce_score', 0):.3f}]"
            elif result.get('search_type') == 'rrf':
                search_info = f" [RRF: {result.get('rrf_score', 0):.3f}]"
            
            sources.append(Source(
                id=original_id,
                type=source_type,
                title=title + chunk_info + search_info,
                url=url,
                relevance_score=result.get('ce_score', result.get('rrf_score', result.get('score', 0.0)))
            ))
        
        # Use tools if enabled and appropriate
        tools_used = []
        tool_enhanced_answer = None
        
        if use_tools:
            suggested_tools = self.tool_manager.should_use_tool(question)
            
            for tool_name in suggested_tools:
                try:
                    # Prepare tool parameters based on the question
                    tool_params = self._prepare_tool_parameters(tool_name, question)
                    
                    # Execute tool
                    tool_result = await self.tool_manager.execute_tool(tool_name, **tool_params)
                    
                    # Record tool usage
                    tools_used.append(ToolUsage(
                        tool_name=tool_name,
                        parameters=tool_params,
                        result=tool_result.to_dict(),
                        success=tool_result.success,
                        execution_time=tool_result.execution_time
                    ))
                    
                    # If tool was successful, try to incorporate its result
                    # For datetime questions, prioritize datetime tool over web search
                    if tool_result.success and tool_result.data:
                        potential_answer = self._incorporate_tool_result(
                            question, tool_name, tool_result.data, search_results
                        )
                        
                        # Use the answer if we don't have one yet, or if this is a datetime tool for a time question
                        if potential_answer and (not tool_enhanced_answer or 
                            (tool_name == "datetime" and any(word in question.lower() for word in ['time', 'date', 'when', 'today', 'now']))):
                            tool_enhanced_answer = potential_answer
                        
                except Exception as e:
                    print(f"Error using tool {tool_name}: {e}")
                    # Continue without this tool
        
        # Generate final answer
        if tool_enhanced_answer:
            answer = tool_enhanced_answer
        elif search_results:
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
            timestamp=datetime.utcnow().isoformat(),
            tools_used=tools_used if tools_used else None
        )
    
    def _prepare_tool_parameters(self, tool_name: str, question: str) -> Dict[str, Any]:
        """Prepare parameters for tool execution based on the question"""
        if tool_name == "datetime":
            # For datetime tool, check if user wants specific format
            if "format" in question.lower() or "yyyy" in question.lower() or "mm/dd" in question.lower():
                return {"format": "%Y-%m-%d %H:%M:%S"}
            return {}
        
        elif tool_name == "web_search":
            # For web search, use the question as the query
            return {"query": question, "max_results": 3}
        
        return {}
    
    def _incorporate_tool_result(self, question: str, tool_name: str, tool_data: Any, kb_results: List[Dict]) -> str:
        """Incorporate tool results into the answer"""
        
        if tool_name == "datetime":
            if isinstance(tool_data, dict):
                current_time = tool_data.get('current_datetime', '')
                weekday = tool_data.get('weekday', '')
                
                # Create a natural language response
                if any(word in question.lower() for word in ['time', 'clock']):
                    return f"The current time is {current_time}."
                elif any(word in question.lower() for word in ['date', 'today']):
                    return f"Today is {weekday}, {current_time.split('T')[0]}."
                else:
                    return f"The current date and time is {current_time} ({weekday})."
        
        elif tool_name == "web_search":
            if isinstance(tool_data, dict) and tool_data.get('results'):
                results = tool_data['results']
                
                # Only use web search if it has valid results and no datetime tool was used for time questions
                if results and not any(r.get('source') == 'error' for r in results):
                    # If we have KB results, combine them with web results
                    if kb_results:
                        kb_answer = kb_results[0].get('answer') or kb_results[0].get('content', '')[:200]
                        web_info = results[0].get('snippet', '') if results else ''
                        
                        if kb_answer and web_info:
                            return f"Based on our knowledge base: {kb_answer}\n\nAdditional current information: {web_info}"
                        elif kb_answer:
                            return kb_answer
                    
                    # If no KB results, use web results
                    if results:
                        top_result = results[0]
                        return f"According to current web sources: {top_result.get('snippet', 'No information available.')}"
        
        return None


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
        "description": "Simplified AI worker for querying knowledge bases with external tools support",
        "version": "2.1.0",
        "endpoints": {
            "query": "POST /query",
            "faq": "GET /v1/projects/{project_id}/faqs/{faq_id}",
            "kb": "GET /v1/projects/{project_id}/kb/{kb_id}",
            "projects": "GET /projects",
            "tools": "GET /tools",
            "execute_tool": "POST /tools/{tool_name}"
        },
        "tools_available": len(worker.tool_manager.get_enabled_tools())
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
    """Answer a question with sources and optional tool assistance"""
    try:
        if request.project_id not in worker.projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        response = await worker.answer_question(request.project_id, request.question)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": worker.tool_manager.list_tools()
    }


@app.post("/tools/{tool_name}")
async def execute_tool(
    tool_name: str = FastAPIPath(..., description="Tool name"),
    parameters: Dict[str, Any] = {}
):
    """Execute a specific tool with given parameters"""
    try:
        result = await worker.tool_manager.execute_tool(tool_name, **parameters)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


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