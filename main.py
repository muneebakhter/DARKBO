from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import uuid
import os
from pathlib import Path

from api.models import (
    ProjectRequest, QueryRequest, QueryResponse,
    FAQBatchUpsertRequest, IngestionResponse,
    FAQEntry
)
from api.storage import FileStorageManager
from api.ingestion import IngestionPipeline
from api.retrieval import QueryProcessor, EmbeddingManager

# Initialize FastAPI app
app = FastAPI(
    title="DARKBO API",
    description="Document Augmented Retrieval Knowledge Base Operator - Hybrid Knowledge Graph & Vector Database",
    version="1.0.0"
)

# Global components
storage_manager = FileStorageManager(base_dir=os.getenv("DARKBO_HOME", str(Path.home())))
embedding_manager = EmbeddingManager(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
ingestion_pipeline = IngestionPipeline(storage_manager)
query_processor = QueryProcessor(storage_manager, embedding_manager)


# Dependency to get storage manager
def get_storage() -> FileStorageManager:
    return storage_manager


# Dependency to get ingestion pipeline
def get_ingestion() -> IngestionPipeline:
    return ingestion_pipeline


# Dependency to get query processor
def get_query_processor() -> QueryProcessor:
    return query_processor


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "DARKBO API",
        "description": "Document Augmented Retrieval Knowledge Base Operator",
        "version": "1.0.0",
        "endpoints": {
            "projects": "POST /v1/projects",
            "ingest": "POST /v1/projects/{project_id}/ingest",
            "faqs": "POST /v1/projects/{project_id}/faqs:batch_upsert",
            "query": "POST /v1/query"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": str(uuid.uuid4())}


@app.post("/v1/projects")
async def create_or_update_project(
    request: ProjectRequest,
    storage: FileStorageManager = Depends(get_storage)
):
    """Create or update a project"""
    try:
        is_new = storage.create_or_update_project(request.project_id, request.project_name)
        
        return {
            "project_id": request.project_id,
            "project_name": request.project_name,
            "status": "created" if is_new else "updated",
            "message": f"Project {'created' if is_new else 'updated'} successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create/update project: {str(e)}")


@app.get("/v1/projects")
async def list_projects(storage: FileStorageManager = Depends(get_storage)):
    """List all projects"""
    try:
        projects = storage.load_project_mapping()
        return {
            "projects": [
                {"project_id": pid, "project_name": name}
                for pid, name in projects.items()
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@app.get("/v1/projects/{project_id}")
async def get_project(
    project_id: str,
    storage: FileStorageManager = Depends(get_storage)
):
    """Get project details"""
    try:
        projects = storage.load_project_mapping()
        
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get FAQ and KB counts
        faqs = storage.load_faqs(project_id)
        kb_entries = storage.load_kb_entries(project_id)
        
        return {
            "project_id": project_id,
            "project_name": projects[project_id],
            "faq_count": len(faqs),
            "kb_count": len(kb_entries),
            "index_metadata": storage.get_index_metadata(project_id).model_dump()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@app.post("/v1/projects/{project_id}/ingest")
async def ingest_documents(
    project_id: str,
    files: List[UploadFile] = File(...),
    type: str = Form("kb"),
    split: str = Form("tokens"),
    max_chunk_chars: int = Form(1200),
    storage: FileStorageManager = Depends(get_storage),
    ingestion: IngestionPipeline = Depends(get_ingestion)
) -> IngestionResponse:
    """Ingest documents into a project"""
    try:
        # Verify project exists
        projects = storage.load_project_mapping()
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Validate parameters
        if type not in ["kb", "faq"]:
            raise HTTPException(status_code=400, detail="Type must be 'kb' or 'faq'")
        
        if split not in ["tokens", "headings", "none"]:
            raise HTTPException(status_code=400, detail="Split must be 'tokens', 'headings', or 'none'")
        
        if max_chunk_chars < 100 or max_chunk_chars > 10000:
            raise HTTPException(status_code=400, detail="max_chunk_chars must be between 100 and 10000")
        
        # Process uploaded files
        file_data = []
        for file in files:
            if file.size == 0:
                continue
            
            content = await file.read()
            file_data.append((file.filename, content))
        
        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files provided")
        
        # Ingest documents
        created_ids, updated_ids, job_id = ingestion.ingest_documents(
            project_id=project_id,
            files=file_data,
            ingest_type=type,
            split_mode=split,
            max_chunk_chars=max_chunk_chars
        )
        
        # Format created IDs with prefixes
        formatted_created = [f"{type}:{project_id}:{id}" for id in created_ids]
        formatted_updated = [f"{type}:{project_id}:{id}" for id in updated_ids]
        
        return IngestionResponse(
            created=formatted_created,
            updated=formatted_updated,
            job_id=job_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest documents: {str(e)}")


@app.post("/v1/projects/{project_id}/faqs:batch_upsert")
async def batch_upsert_faqs(
    project_id: str,
    request: FAQBatchUpsertRequest,
    storage: FileStorageManager = Depends(get_storage)
):
    """Batch upsert FAQ entries"""
    try:
        # Verify project exists
        projects = storage.load_project_mapping()
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Validate FAQ items
        if not request.items:
            raise HTTPException(status_code=400, detail="No FAQ items provided")
        
        faqs = []
        for item in request.items:
            if "question" not in item or "answer" not in item:
                raise HTTPException(status_code=400, detail="Each FAQ item must have 'question' and 'answer'")
            
            faq = FAQEntry.from_qa(
                project_id=project_id,
                question=item["question"],
                answer=item["answer"],
                source="manual"
            )
            faqs.append(faq)
        
        # Upsert FAQs
        created_ids, updated_ids = storage.upsert_faqs(project_id, faqs, replace=request.replace)
        
        return {
            "project_id": project_id,
            "created": len(created_ids),
            "updated": len(updated_ids),
            "total": len(faqs),
            "replace": request.replace,
            "created_ids": created_ids,
            "updated_ids": updated_ids
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert FAQs: {str(e)}")


@app.get("/v1/projects/{project_id}/faqs")
async def get_faqs(
    project_id: str,
    storage: FileStorageManager = Depends(get_storage)
):
    """Get all FAQ entries for a project"""
    try:
        # Verify project exists
        projects = storage.load_project_mapping()
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        faqs = storage.load_faqs(project_id)
        
        return {
            "project_id": project_id,
            "count": len(faqs),
            "faqs": [faq.model_dump() for faq in faqs]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get FAQs: {str(e)}")


@app.get("/v1/projects/{project_id}/kb")
async def get_kb_entries(
    project_id: str,
    storage: FileStorageManager = Depends(get_storage)
):
    """Get all KB entries for a project"""
    try:
        # Verify project exists
        projects = storage.load_project_mapping()
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        kb_entries = storage.load_kb_entries(project_id)
        
        return {
            "project_id": project_id,
            "count": len(kb_entries),
            "kb_entries": [entry.model_dump() for entry in kb_entries]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KB entries: {str(e)}")


@app.post("/v1/query")
async def query(
    request: QueryRequest,
    query_processor: QueryProcessor = Depends(get_query_processor),
    storage: FileStorageManager = Depends(get_storage)
) -> QueryResponse:
    """Ask a question and get an answer with citations"""
    try:
        # Verify project exists
        projects = storage.load_project_mapping()
        if request.project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Process query
        response = query_processor.process_query(
            project_id=request.project_id,
            question=request.question,
            mode=request.mode,
            strict_citations=request.strict_citations
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/v1/projects/{project_id}/rebuild_index")
async def rebuild_index(
    project_id: str,
    query_processor: QueryProcessor = Depends(get_query_processor),
    storage: FileStorageManager = Depends(get_storage)
):
    """Rebuild search indexes for a project"""
    try:
        # Verify project exists
        projects = storage.load_project_mapping()
        if project_id not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get retriever and rebuild indexes
        retriever = query_processor._get_retriever(project_id)
        retriever.rebuild_indexes()
        
        return {
            "project_id": project_id,
            "status": "success",
            "message": "Indexes rebuilt successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": "The requested resource was not found"}
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"Starting DARKBO API on {host}:{port}")
    print(f"Base directory: {storage_manager.base_dir}")
    print(f"Debug mode: {reload}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )