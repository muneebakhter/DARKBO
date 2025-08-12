# DARKBO - Simplified Knowledge Base System Implementation

## Overview

DARKBO has been simplified to a two-script architecture that focuses on the core use case: building knowledge base indexes and serving queries with source citations.

## Simplified Architecture

### Core Scripts
1. **prebuild_kb.py** - Builds vector stores and search indexes from FAQ/KB data
2. **ai_worker.py** - FastAPI server that answers questions with source citations

### Key Features Implemented

### ✅ 1. File Structure (Optimized)
- `proj_mapping.txt` - Tab-separated project ID to name mapping
- `<project_id>/` - Individual project directories
- `<project_id>/<project_id>.faq.json` - FAQ data with stable UUID5 IDs
- `<project_id>/<project_id>.kb.json` - Knowledge base data with stable UUID5 IDs
- `<project_id>/attachments/` - Optional raw uploaded documents
- `<project_id>/index/dense/` - Dense vector indexes (FAISS) when dependencies available
- `<project_id>/index/sparse/` - Sparse text indexes (Whoosh) when dependencies available  
- `<project_id>/index/meta.json` - Index metadata and checksums

### ✅ 2. Stable UUID5 IDs
- **FAQ IDs**: `uuid5(NAMESPACE_URL, f"faq:{project_id}:{question.strip()}:{answer.strip()}")`
- **KB IDs**: `uuid5(NAMESPACE_URL, f"kb:{project_id}:{article}:{sha256(content)}")`
- Ensures deterministic IDs for reliable operations

### ✅ 3. Simple Data Models  
- **FAQEntry**: Question/answer pairs with metadata
- **KBEntry**: Knowledge base articles with content and metadata
- **Timestamps**: `created_at`, `updated_at` for all entries
- **Source tracking**: `source`, `source_file`, `chunk_index`

### ✅ 4. Simplified API Endpoints

#### Core Query System
- `POST /query` - Ask questions and get answers with source citations
- `GET /projects` - List available projects  
- `GET /health` - Health check endpoint

#### Source Retrieval
- `GET /v1/projects/{project_id}/faqs/{faq_id}` - Get FAQ by ID (returns attachment file if available, otherwise JSON)
- `GET /v1/projects/{project_id}/kb/{kb_id}` - Get KB entry by ID (returns attachment file if available, otherwise JSON)

### ✅ 5. Hybrid Search System (Confirmed)
- **Dense Search**: FAISS vector similarity with sentence-transformers (semantic search)
- **Sparse Search**: Whoosh text search (keyword/BM25-style search)
- **Basic Search**: Simple keyword matching fallback (no dependencies required)
- **Smart Fallback**: Gracefully degrades when ML dependencies not available
- **Hybrid Results**: Combines dense and sparse results, removes duplicates, ranks by relevance

### ✅ 7. External Tools Framework (New)
- **Tools Architecture**: Modular framework in `tools/` directory with abstract base classes
- **DateTime Tool**: Provides current date, time, timezone information with customizable formatting
- **Web Search Tool**: DuckDuckGo API integration for web search capabilities
- **Tool Manager**: Automatic tool selection based on query keywords
- **Smart Integration**: Tools results incorporated into knowledge base answers
- **API Endpoints**: Direct tool access via `/tools` and `/tools/{tool_name}` endpoints
- **Graceful Handling**: Tools failures don't break the main query processing

### Tools Usage Patterns
- **Date/Time Queries**: `"What time is it?"`, `"What date is today?"` → Uses DateTime tool
- **General Questions**: `"What is X?"`, `"How to Y?"` → Uses Web search + KB knowledge
- **KB-first Approach**: Knowledge base results are primary, tools provide supplementary information
- **Error Resilience**: Network failures handled gracefully with informative error messages

## Simplified Implementation Details

### Two-Script Workflow

#### 1. prebuild_kb.py
- Loads FAQ and KB data from project directories
- Builds dense vector indexes using sentence-transformers (optional)
- Builds sparse text indexes using Whoosh (optional) 
- Creates metadata files with checksums for change detection
- Works with minimal dependencies (creates metadata-only indexes)

#### 2. ai_worker.py  
- FastAPI server that loads prebuilt indexes
- Provides query endpoint with hybrid search
- Returns answers with source citations
- Serves attachment files when available
- Includes basic keyword search fallback

### Query Response Format
```json
{
  "answer": "American Society for the Prevention of Cruelty to Animals",
  "sources": [
    {
      "id": "uuid5-id",
      "type": "faq",
      "title": "FAQ: What does ASPCA stand for?",
      "url": "/v1/projects/95/faqs/uuid5-id",
      "relevance_score": 0.95
    }
  ],
  "project_id": "95",
  "timestamp": "2025-08-11T22:04:11.265860"
}
```

## Simplified File Structure

```
./
├── proj_mapping.txt                     # "175\tACLU\n95\tASPCA"
├── 175/                                 # ACLU project
│   ├── attachments/                     # Optional files
│   ├── index/                           # Generated during prebuild
│   │   ├── dense/                       # FAISS indexes (optional)
│   │   ├── sparse/                      # Whoosh indexes (optional)
│   │   └── meta.json                    # Index metadata
│   ├── 175.faq.json                     # FAQ entries with UUID5 IDs
│   └── 175.kb.json                      # KB entries with UUID5 IDs
└── 95/                                  # ASPCA project
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

## Usage Examples

### 1. Basic Setup
```bash
# Create project mapping
echo -e "95\tASPCA\n175\tACLU" > proj_mapping.txt

# Copy sample data (optional)
cp -r sample_data/* .

# Build indexes
python prebuild_kb.py
```

### 2. Start Server
```bash
python ai_worker.py
```

### 3. Query the System  
```bash
# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What does ASPCA stand for?"}'

# List projects
curl "http://localhost:8000/projects"

# Get source by ID
curl "http://localhost:8000/v1/projects/95/faqs/1766291f-f2f5-5f01-b1bb-fc95501ab163"
```

## Implementation Status

### ✅ Completed - Core Functionality
- Simplified two-script architecture
- Core data models with stable UUID5 IDs
- File storage system with metadata tracking
- Hybrid search with graceful fallback
- FastAPI server with essential endpoints
- Basic keyword search (no dependencies required)
- Source citations and attachment serving

### ✅ Completed - Enhanced Functionality (Optional Dependencies)
- Dense vector search with sentence-transformers + FAISS
- Sparse text search with Whoosh
- Hybrid search combining both approaches

### ⚠️ Removed - Complex Features
- Document ingestion pipeline
- Project management endpoints
- Batch operations
- Complex FastAPI application

## Running the Simplified System

### Quick Start
```bash
# 1. Generate sample data
python3 create_sample_data.py

# 2. Change to sample data directory
cd sample_data

# 3. Build indexes
python3 ../prebuild_kb.py

# 4. Start server  
python3 ../ai_worker.py

# 5. Test queries
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What does ASPCA stand for?"}'
```

### Minimal Installation
```bash
pip install fastapi uvicorn pydantic
python prebuild_kb.py  # Creates metadata-only indexes
python ai_worker.py    # Start server with basic search
```

### Full Installation  
```bash
pip install fastapi uvicorn pydantic sentence-transformers faiss-cpu whoosh numpy
python3 create_sample_data.py  # Generate sample data
cd sample_data
python3 ../prebuild_kb.py      # Creates full indexes with hybrid search
python3 ../ai_worker.py        # Start server with hybrid search
```

### Alternative Demo
```bash
./demo.sh  # Runs the complete workflow automatically
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