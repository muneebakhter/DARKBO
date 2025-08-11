import uuid
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
import json

# Namespace for UUID5 generation
NAMESPACE_URL = uuid.NAMESPACE_URL

class FAQEntry(BaseModel):
    """FAQ entry with stable UUID5 ID"""
    id: str = Field(..., description="Stable UUID5 ID")
    question: str = Field(..., description="FAQ question")
    answer: str = Field(..., description="FAQ answer")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="manual", description="Source of the FAQ")
    source_file: Optional[str] = Field(None, description="Original file name")
    
    @classmethod
    def create_id(cls, project_id: str, question: str, answer: str) -> str:
        """Generate stable UUID5 ID for FAQ entry"""
        content = f"faq:{project_id}:{question.strip()}:{answer.strip()}"
        return str(uuid.uuid5(NAMESPACE_URL, content))
    
    @classmethod
    def from_qa(cls, project_id: str, question: str, answer: str, **kwargs) -> "FAQEntry":
        """Create FAQ entry from question and answer"""
        faq_id = cls.create_id(project_id, question, answer)
        return cls(
            id=faq_id,
            question=question,
            answer=answer,
            **kwargs
        )

class KBEntry(BaseModel):
    """Knowledge base entry with stable UUID5 ID"""
    id: str = Field(..., description="Stable UUID5 ID")
    article: str = Field(..., description="Article title/name")
    content: str = Field(..., description="Article content")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="upload", description="Source type")
    source_file: Optional[str] = Field(None, description="Original file name")
    chunk_index: Optional[int] = Field(None, description="Chunk index if split")
    
    @classmethod
    def create_id(cls, project_id: str, article: str, content: str) -> str:
        """Generate stable UUID5 ID for KB entry"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        content_str = f"kb:{project_id}:{article}:{content_hash}"
        return str(uuid.uuid5(NAMESPACE_URL, content_str))
    
    @classmethod
    def from_content(cls, project_id: str, article: str, content: str, **kwargs) -> "KBEntry":
        """Create KB entry from article and content"""
        kb_id = cls.create_id(project_id, article, content)
        return cls(
            id=kb_id,
            article=article,
            content=content,
            **kwargs
        )

class ProjectMetadata(BaseModel):
    """Project metadata"""
    project_id: str
    project_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    faq_count: int = 0
    kb_count: int = 0

class IndexMetadata(BaseModel):
    """Index metadata for tracking versions and checksums"""
    faq_checksum: Optional[str] = None
    kb_checksum: Optional[str] = None
    index_version: str = "1.0"
    embedding_model: str = "all-MiniLM-L6-v2"
    faq_count: int = 0
    kb_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class QueryRequest(BaseModel):
    """Query request model"""
    project_id: str
    question: str
    mode: str = Field(default="auto", description="Query mode: auto, faq, kb")
    strict_citations: bool = Field(default=True, description="Require strict citations")

class Citation(BaseModel):
    """Citation model for query responses"""
    type: str = Field(..., description="Citation type: faq or kb")
    id: str = Field(..., description="Entry ID")
    article: Optional[str] = Field(None, description="Article name for KB entries")
    lines: Optional[List[int]] = Field(None, description="Line numbers referenced")
    score: float = Field(..., description="Relevance score")

class QueryResponse(BaseModel):
    """Structured query response with citations"""
    answer: str
    mode: str = Field(..., description="Mode used: faq or kb")
    confidence: float = Field(..., description="Answer confidence")
    citations: List[Citation] = Field(default_factory=list)
    used_chunks: List[str] = Field(default_factory=list, description="Chunk IDs used")

class ProjectRequest(BaseModel):
    """Project creation/update request"""
    project_id: str
    project_name: str

class FAQBatchUpsertRequest(BaseModel):
    """FAQ batch upsert request"""
    items: List[Dict[str, str]] = Field(..., description="List of FAQ items with question/answer")
    replace: bool = Field(default=False, description="Replace all existing FAQs")

class IngestionResponse(BaseModel):
    """Response for document ingestion"""
    created: List[str] = Field(default_factory=list, description="Created entry IDs")
    updated: List[str] = Field(default_factory=list, description="Updated entry IDs")
    job_id: str = Field(..., description="Processing job ID")