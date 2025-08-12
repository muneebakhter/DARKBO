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
    from sentence_transformers import SentenceTransformer
    import faiss
    from whoosh.index import open_dir
    from whoosh.qparser import QueryParser
    import openai
    from dotenv import load_dotenv
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# Load environment variables from .env file
load_dotenv()

from api.models import FAQEntry, KBEntry
from api.storage import FileStorageManager
try:
    from api.document_processor import process_document_for_kb
except ImportError:
    from api.simple_processor import process_document_for_kb
from api.index_versioning import IndexBuilder, IndexVersionManager
from tools import ToolManager

# Additional imports for file handling - will be enabled when python-multipart is available
# from fastapi import File, UploadFile, Form, BackgroundTasks
from fastapi import BackgroundTasks
import uuid
import asyncio


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

class FAQCreateRequest(BaseModel):
    question: str
    answer: str

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    kb_entries_created: Optional[List[str]] = None
    index_build_started: bool = False
    
class IndexBuildResponse(BaseModel):
    success: bool
    message: str
    version: Optional[str] = None
    build_status: Optional[Dict[str, Any]] = None


class KnowledgeBaseRetriever:
    """Retrieves information from prebuilt indexes with versioning support"""
    
    def __init__(self, project_id: str, base_dir: str = "."):
        self.project_id = project_id
        self.base_dir = Path(base_dir)
        self.project_dir = self.base_dir / project_id
        self.version_manager = IndexVersionManager(project_id, base_dir)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Initialize indexes
        self.dense_index = None
        self.dense_metadata = None
        self.sparse_index = None
        self.embedding_model = None
        
        self._load_indexes()
    
    def _load_metadata(self) -> Dict:
        """Load index metadata from current version"""
        try:
            index_paths = self.version_manager.get_current_index_paths()
            meta_file = index_paths['meta']
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata for {self.project_id}: {e}")
        
        return {"error": "No index metadata found"}
    
    def _load_indexes(self):
        """Load prebuilt indexes from current version"""
        if not HAS_DEPS:
            return
        
        try:
            index_paths = self.version_manager.get_current_index_paths()
            
            # Load dense index
            if self.metadata.get('indexes', {}).get('dense', {}).get('available'):
                dense_dir = index_paths['dense']
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
                sparse_dir = index_paths['sparse']
                if sparse_dir.exists():
                    self.sparse_index = open_dir(str(sparse_dir))
                    
        except Exception as e:
            print(f"Warning: Could not load indexes for {self.project_id}: {e}")
    
    def reload_indexes(self):
        """Reload indexes after a rebuild"""
        self.metadata = self._load_metadata()
        self.dense_index = None
        self.dense_metadata = None
        self.sparse_index = None
        self.embedding_model = None
        self._load_indexes()
    
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
    """Main AI worker class with document ingestion support"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.projects = self._load_projects()
        self.retrievers = {}  # Cache retrievers
        self.tool_manager = ToolManager()  # Initialize tool manager
        self.storage = FileStorageManager(str(self.base_dir))  # Storage manager
        
        # Initialize OpenAI client
        self.openai_client = None
        self.openai_model = "gpt-4o-mini"  # Default model for cost efficiency
        self._setup_openai()
    
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
    
    def _setup_openai(self):
        """Setup OpenAI client with API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if api_key and api_key != "your_openai_api_key_here":
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.openai_model = model
                print(f"✅ OpenAI client initialized with model: {model}")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            print("⚠️ No valid OpenAI API key found. AI responses will be limited to knowledge base content.")
            self.openai_client = None
    
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
    
    async def _generate_ai_response(self, question: str, search_results: List[Dict], tools_used: List[ToolUsage], project_name: str = None) -> Optional[str]:
        """Generate AI response using OpenAI API with RAG context"""
        if not self.openai_client:
            return None
        
        try:
            # Prepare system prompt
            system_prompt = f"""You are ACD Direct's Knowledge Base AI System, a helpful and knowledgeable assistant. 
You have access to a comprehensive knowledge base and various tools to provide accurate information.

Your primary role is to:
1. Answer questions using the provided knowledge base context when relevant
2. Use tool results when they provide current or specific information
3. Introduce yourself appropriately when asked about your identity
4. Be helpful, accurate, and conversational

Project context: {project_name or 'Knowledge Base System'}"""

            # Prepare context from search results
            context_parts = []
            if search_results:
                context_parts.append("=== KNOWLEDGE BASE CONTEXT ===")
                for i, result in enumerate(search_results[:3], 1):  # Limit to top 3 for context window
                    if result.get('type') == 'faq':
                        context_parts.append(f"{i}. FAQ - {result.get('question', 'N/A')}")
                        context_parts.append(f"   Answer: {result.get('answer', 'N/A')}")
                    else:
                        context_parts.append(f"{i}. Article - {result.get('article', 'N/A')}")
                        content = result.get('content', '')
                        # Truncate long content
                        if len(content) > 300:
                            content = content[:300] + "..."
                        context_parts.append(f"   Content: {content}")
                    context_parts.append("")
            
            # Add tool results context
            if tools_used:
                context_parts.append("=== TOOL RESULTS ===")
                for tool in tools_used:
                    if tool.success and tool.result.get('data'):
                        context_parts.append(f"Tool: {tool.tool_name}")
                        context_parts.append(f"Result: {tool.result['data']}")
                        context_parts.append("")
            
            # Prepare user message
            context_text = "\n".join(context_parts) if context_parts else "No specific context available."
            user_message = f"""Question: {question}

Context:
{context_text}

Please provide a helpful response based on the question and available context. If the question is about your identity, introduce yourself as "ACD Direct's Knowledge Base AI System"."""

            # Make OpenAI API call
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_completion_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return None
    
    async def answer_question(self, project_id: str, question: str, use_tools: bool = True) -> QueryResponse:
        """Generate answer with sources and optional tool assistance"""
        # Get retriever
        retriever = self.get_retriever(project_id)
        
        # Search for relevant content in KB
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
        
        # Generate final answer using AI agent - no fallbacks allowed
        project_name = self.projects.get(project_id, "Knowledge Base")
        ai_response = await self._generate_ai_response(question, search_results, tools_used, project_name)
        
        if not ai_response:
            # If AI agent fails, return an error instead of falling back
            raise ValueError("AI agent is unavailable. Please ensure OpenAI API is properly configured and accessible.")
        
        return QueryResponse(
            answer=ai_response,
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

    async def ingest_document(self, project_id: str, file, article_title: str = None) -> DocumentUploadResponse:
        """Ingest a document into the knowledge base"""
        try:
            # Validate project
            if project_id not in self.projects:
                return DocumentUploadResponse(
                    success=False,
                    message=f"Project {project_id} not found"
                )
            
            # Validate file type
            if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
                return DocumentUploadResponse(
                    success=False,
                    message="Only PDF and DOCX files are supported"
                )
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Save uploaded file temporarily
            temp_file = f"/tmp/{doc_id}_{file.filename}"
            with open(temp_file, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            try:
                # Process document
                full_text, chunks, metadata = process_document_for_kb(
                    temp_file, 
                    article_title or Path(file.filename).stem
                )
                
                # Save attachment file
                attachment_filename = f"{doc_id}-kb{Path(file.filename).suffix}"
                self.storage.save_attachment(project_id, attachment_filename, content)
                
                # Create KB entries from chunks
                kb_entries = []
                created_ids = []
                
                for i, chunk in enumerate(chunks):
                    kb_entry = KBEntry.from_content(
                        project_id=project_id,
                        article=metadata['article_title'],
                        content=chunk,
                        source="upload",
                        source_file=attachment_filename,
                        chunk_index=i if len(chunks) > 1 else None
                    )
                    kb_entries.append(kb_entry)
                    created_ids.append(kb_entry.id)
                
                # Save KB entries
                if kb_entries:
                    self.storage.upsert_kb_entries(project_id, kb_entries)
                
                # Start index rebuild in background
                index_build_started = False
                try:
                    builder = IndexBuilder(project_id, str(self.base_dir))
                    if builder.version_manager.needs_rebuild():
                        # Start background task for index rebuild
                        asyncio.create_task(self._rebuild_indexes_async(project_id))
                        index_build_started = True
                except Exception as e:
                    print(f"Warning: Could not start index rebuild: {e}")
                
                return DocumentUploadResponse(
                    success=True,
                    message=f"Document processed successfully. Created {len(kb_entries)} KB entries.",
                    document_id=doc_id,
                    kb_entries_created=created_ids,
                    index_build_started=index_build_started
                )
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            return DocumentUploadResponse(
                success=False,
                message=f"Document processing failed: {str(e)}"
            )
    
    async def add_faq(self, project_id: str, question: str, answer: str) -> DocumentUploadResponse:
        """Add a new FAQ entry"""
        try:
            # Validate project
            if project_id not in self.projects:
                return DocumentUploadResponse(
                    success=False,
                    message=f"Project {project_id} not found"
                )
            
            # Create FAQ entry
            faq = FAQEntry.from_qa(
                project_id=project_id,
                question=question.strip(),
                answer=answer.strip(),
                source="manual"
            )
            
            # Save FAQ
            created_ids, updated_ids = self.storage.upsert_faqs(project_id, [faq])
            
            # Start index rebuild in background
            index_build_started = False
            try:
                builder = IndexBuilder(project_id, str(self.base_dir))
                if builder.version_manager.needs_rebuild():
                    asyncio.create_task(self._rebuild_indexes_async(project_id))
                    index_build_started = True
            except Exception as e:
                print(f"Warning: Could not start index rebuild: {e}")
            
            action = "updated" if updated_ids else "created"
            return DocumentUploadResponse(
                success=True,
                message=f"FAQ {action} successfully",
                document_id=faq.id,
                kb_entries_created=[faq.id],
                index_build_started=index_build_started
            )
            
        except Exception as e:
            return DocumentUploadResponse(
                success=False,
                message=f"FAQ creation failed: {str(e)}"
            )
    
    async def rebuild_indexes(self, project_id: str) -> IndexBuildResponse:
        """Manually trigger index rebuild"""
        try:
            if project_id not in self.projects:
                return IndexBuildResponse(
                    success=False,
                    message=f"Project {project_id} not found"
                )
            
            builder = IndexBuilder(project_id, str(self.base_dir))
            
            # Check if build is already in progress
            if builder.version_manager.is_building():
                return IndexBuildResponse(
                    success=False,
                    message="Index build already in progress",
                    build_status=builder.version_manager.get_build_status()
                )
            
            # Start build
            new_version = builder.build_new_version()
            
            # Reload retrievers to use new index
            if project_id in self.retrievers:
                self.retrievers[project_id].reload_indexes()
            
            return IndexBuildResponse(
                success=True,
                message=f"Index rebuild completed",
                version=new_version,
                build_status=builder.version_manager.get_build_status()
            )
            
        except Exception as e:
            return IndexBuildResponse(
                success=False,
                message=f"Index rebuild failed: {str(e)}"
            )
    
    async def get_build_status(self, project_id: str) -> IndexBuildResponse:
        """Get index build status"""
        try:
            if project_id not in self.projects:
                return IndexBuildResponse(
                    success=False,
                    message=f"Project {project_id} not found"
                )
            
            version_manager = IndexVersionManager(project_id, str(self.base_dir))
            build_status = version_manager.get_build_status()
            
            return IndexBuildResponse(
                success=True,
                message="Build status retrieved",
                build_status=build_status
            )
            
        except Exception as e:
            return IndexBuildResponse(
                success=False,
                message=f"Failed to get build status: {str(e)}"
            )
    
    async def _rebuild_indexes_async(self, project_id: str):
        """Background task to rebuild indexes"""
        try:
            builder = IndexBuilder(project_id, str(self.base_dir))
            new_version = builder.build_new_version()
            
            # Reload retrievers to use new index
            if project_id in self.retrievers:
                self.retrievers[project_id].reload_indexes()
                
            print(f"Background index rebuild completed for project {project_id}, version {new_version}")
            
        except Exception as e:
            print(f"Background index rebuild failed for project {project_id}: {e}")


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
            "execute_tool": "POST /tools/{tool_name}",
            "upload_document": "POST /projects/{project_id}/documents",
            "add_faq": "POST /projects/{project_id}/faqs",
            "rebuild_indexes": "POST /projects/{project_id}/rebuild-indexes",
            "build_status": "GET /projects/{project_id}/build-status"
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


# @app.post("/projects/{project_id}/documents")
# async def upload_document(
#     project_id: str = FastAPIPath(..., description="Project ID"),
#     file: UploadFile = File(..., description="Document file (PDF or DOCX)")
# ) -> DocumentUploadResponse:
#     """Upload and process a document for ingestion into the knowledge base"""
#     return await worker.ingest_document(project_id, file, None)

# Document upload endpoint disabled until python-multipart is installed
@app.post("/projects/{project_id}/documents-disabled")  
async def upload_document_disabled():
    """Document upload requires python-multipart to be installed"""
    raise HTTPException(
        status_code=501, 
        detail="Document upload requires: pip install python-multipart python-docx PyPDF2"
    )


@app.post("/projects/{project_id}/faqs")
async def add_faq(
    faq_request: FAQCreateRequest,
    project_id: str = FastAPIPath(..., description="Project ID")
) -> DocumentUploadResponse:
    """Add a new FAQ entry to the project"""
    return await worker.add_faq(project_id, faq_request.question, faq_request.answer)


@app.post("/projects/{project_id}/rebuild-indexes")
async def rebuild_indexes(
    project_id: str = FastAPIPath(..., description="Project ID")
) -> IndexBuildResponse:
    """Manually trigger index rebuild for a project"""
    return await worker.rebuild_indexes(project_id)


@app.get("/projects/{project_id}/build-status")
async def get_build_status(
    project_id: str = FastAPIPath(..., description="Project ID")
) -> IndexBuildResponse:
    """Get current index build status for a project"""
    return await worker.get_build_status(project_id)


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