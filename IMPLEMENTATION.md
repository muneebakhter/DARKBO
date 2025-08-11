# DARKBO - Hybrid Knowledge Graph & Vector Database Implementation

## Overview

This implementation transforms DARKBO from a Flask-based frontend to a FastAPI-based hybrid knowledge graph and vector database system as specified in the requirements.

## Key Features Implemented

### ✅ 1. New File Structure
- `$HOME/proj_mapping.txt` - Tab-separated project ID to name mapping
- `$HOME/<project_id>/` - Individual project directories
- `$HOME/<project_id>/<project_id>.faq.json` - FAQ data with stable UUID5 IDs
- `$HOME/<project_id>/<project_id>.kb.json` - Knowledge base data with stable UUID5 IDs
- `$HOME/<project_id>/attachments/` - Raw uploaded documents
- `$HOME/<project_id>/index/dense/` - Dense vector indexes (FAISS)
- `$HOME/<project_id>/index/sparse/` - Sparse BM25 indexes (Whoosh)
- `$HOME/<project_id>/index/meta.json` - Index metadata and checksums

### ✅ 2. Stable UUID5 IDs
- **FAQ IDs**: `uuid5(NAMESPACE_URL, f"faq:{project_id}:{question.strip()}:{answer.strip()}")`
- **KB IDs**: `uuid5(NAMESPACE_URL, f"kb:{project_id}:{article}:{sha256(content)}")`
- Ensures deterministic IDs for reliable upsert/delete operations

### ✅ 3. Data Models
- **FAQEntry**: Question/answer pairs with metadata
- **KBEntry**: Knowledge base articles with content and metadata
- **Timestamps**: `created_at`, `updated_at` for all entries
- **Source tracking**: `source`, `source_file`, `chunk_index`

### ✅ 4. FastAPI Endpoints

#### Project Management
- `POST /v1/projects` - Create/update projects
- `GET /v1/projects` - List all projects
- `GET /v1/projects/{project_id}` - Get project details

#### Document Ingestion
- `POST /v1/projects/{project_id}/ingest?type=kb&split=tokens&max_chunk_chars=1200`
- Supports PDF, DOCX, TXT, Markdown files
- Multiple splitting modes: tokens, headings, none
- Automatic FAQ extraction from documents

#### FAQ Management
- `POST /v1/projects/{project_id}/faqs:batch_upsert` - Batch upsert FAQs
- `GET /v1/projects/{project_id}/faqs` - Get all FAQs
- Support for replace mode

#### Knowledge Base
- `GET /v1/projects/{project_id}/kb` - Get all KB entries

#### Query System
- `POST /v1/query` - Ask questions with structured responses
- FAQ-first routing with configurable threshold
- Hybrid retrieval (dense + sparse)
- Structured output with citations and confidence

### ✅ 5. Document Processing Pipeline
- **PDF Parser**: Using pdfminer.six with LAParams for better extraction
- **DOCX Parser**: Using python-docx for Word documents
- **Text Parser**: Plain text and Markdown support
- **Text Splitter**: Token-based and heading-based splitting
- **FAQ Extraction**: Heuristic-based Q&A detection

### ✅ 6. Hybrid Retrieval System
- **Dense Search**: Sentence transformer embeddings with FAISS
- **Sparse Search**: BM25 using Whoosh
- **Combined Scoring**: Weighted combination of sparse and dense results
- **FAQ-First Routing**: Configurable threshold for direct FAQ responses
- **Reranking**: Score-based result ordering

### ✅ 7. Atomic Writes & File Locking
- File-based locking for concurrent access
- Atomic writes using temporary files + rename
- Checksum-based change detection
- Graceful error handling and cleanup

### ✅ 8. Structured Query Responses
```json
{
  "answer": "...",
  "mode": "faq|kb",
  "confidence": 0.0-1.0,
  "citations": [
    {
      "type": "faq|kb",
      "id": "uuid5-id",
      "article": "...",
      "lines": [12, 28],
      "score": 0.93
    }
  ],
  "used_chunks": ["uuid5-id-1", "uuid5-id-2"]
}
```

## File Structure Example

```
$HOME/
├── proj_mapping.txt                    # "175\tACLU\n95\tASPCA"
├── 175/                                # ACLU project
│   ├── attachments/
│   │   └── aclu_rights_guide.txt
│   ├── index/
│   │   ├── dense/                      # FAISS indexes
│   │   ├── sparse/                     # Whoosh indexes
│   │   └── meta.json                   # Index metadata
│   ├── 175.faq.json                    # FAQ entries with UUID5 IDs
│   └── 175.kb.json                     # KB entries with UUID5 IDs
└── 95/                                 # ASPCA project
    ├── attachments/
    ├── index/
    ├── 95.faq.json
    └── 95.kb.json
```

## API Usage Examples

### 1. Create Project
```bash
curl -X POST "http://localhost:8000/v1/projects" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "175", "project_name": "ACLU"}'
```

### 2. Ingest Documents
```bash
curl -X POST "http://localhost:8000/v1/projects/175/ingest?type=kb&split=tokens&max_chunk_chars=1200" \
  -F "files=@handbook.pdf" \
  -F "files=@guide.docx"
```

### 3. Add FAQs
```bash
curl -X POST "http://localhost:8000/v1/projects/175/faqs:batch_upsert" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"question": "When are phones staffed?", "answer": "Mon–Fri 9–5 MT"}
    ],
    "replace": false
  }'
```

### 4. Query System
```bash
curl -X POST "http://localhost:8000/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "175",
    "question": "What are your phone hours on weekdays?",
    "mode": "auto",
    "strict_citations": true
  }'
```

## Current Implementation Status

### ✅ Completed
- Core data models with stable UUID5 IDs
- File storage system with atomic writes
- Document ingestion pipeline
- Basic hybrid retrieval framework
- FastAPI application structure
- Comprehensive test suite
- Sample data creation

### 🚧 Requires Dependencies for Full Functionality
- FastAPI/Uvicorn for web server
- Sentence transformers for embeddings
- FAISS for dense vector search
- Whoosh for sparse BM25 search
- Document processing libraries (pdfminer, python-docx)

### 🔄 Optional Enhancements
- LLM integration for answer generation
- Cross-encoder reranking
- Advanced answerability checking
- Elasticsearch integration option
- pgvector/Qdrant integration option

## Running the System

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python main.py
   # or
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Test the implementation**:
   ```bash
   python test_core.py        # Test core components
   python create_sample_data.py  # Create sample data
   python test_api.py         # Test API endpoints (requires server)
   ```

## Configuration

Environment variables:
- `DARKBO_HOME`: Base directory for data storage
- `EMBEDDING_MODEL`: Sentence transformer model name
- `FAQ_THRESHOLD`: Threshold for FAQ-first routing
- `HOST`, `PORT`: Server configuration
- `DEBUG`: Debug mode

## Architecture Benefits

1. **Single Source of Truth**: File-based storage ensures data persistence
2. **Scalable**: Easy to add more projects and scale horizontally
3. **Hybrid Search**: Combines semantic and lexical search
4. **Atomic Operations**: Ensures data consistency
5. **Extensible**: Easy to add new document types and search methods
6. **API-First**: Clean REST API for integration