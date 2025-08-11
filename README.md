# DARKBO - Document Augmented Retrieval Knowledge Base Operator

A simplified AI knowledge base system with two main scripts:

1. **prebuild_kb.py** - Builds vector stores and search indexes from FAQ/KB data
2. **ai_worker.py** - FastAPI server that answers questions with source citations

## 📁 File Structure

```
proj_mapping.txt                             # "<id>\t<name>\n"
<project_id>/
  attachments/                               # raw docs uploaded (optional keep)
  <project_id>.faq.json                      # [{"id","question","answer",...}]
  <project_id>.kb.json                       # [{"id","article","content",...}]
  index/                                     # derived: embeddings, bm25, caches
    dense/                                   # FAISS/Qdrant/pgvector
    sparse/                                  # Whoosh/Elastic/Tantivy index
    meta.json                                # checksums, versions, counts
```

## 🚀 Quick Start

### Prerequisites

Install Python 3.8+ and required dependencies:

```bash
# Minimal installation (metadata-only indexes)
pip install fastapi uvicorn pydantic

# Full installation (with vector search capabilities)
pip install fastapi uvicorn pydantic sentence-transformers faiss-cpu whoosh numpy

# Optional: Enhanced answer generation
pip install openai
```

### 1. Prepare Your Data

Create a project mapping file:
```bash
echo -e "95\tASPCA\n175\tACLU" > proj_mapping.txt
```

Create project directories with FAQ and KB data:
```bash
mkdir -p 95 175
# Add your 95.faq.json, 95.kb.json, 175.faq.json, 175.kb.json files
```

### 2. Build Knowledge Base Indexes

```bash
python prebuild_kb.py
```

This will:
- Load FAQ and KB data from project directories
- Build dense vector indexes (if dependencies available)
- Build sparse text indexes (if dependencies available)  
- Create metadata for change detection
- Output progress for each project

### 3. Start the AI Worker Server

```bash
python ai_worker.py
```

The server will start on `http://localhost:8000` with the following endpoints:
- `POST /query` - Ask questions and get answers with sources
- `GET /projects` - List available projects
- `GET /v1/projects/{project_id}/faqs/{faq_id}` - Get FAQ by ID
- `GET /v1/projects/{project_id}/kb/{kb_id}` - Get KB entry by ID

### 4. Test the System

```bash
# Test with the included sample data
cp -r sample_data/* .
python prebuild_kb.py
python ai_worker.py

# In another terminal, test queries:
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What does ASPCA stand for?"}'
```

## 📡 API Endpoints

### Core Query Endpoint
```bash
# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "95",
    "question": "What does ASPCA stand for?"
  }'
```

Response includes answer and sources:
```json
{
  "answer": "American Society for the Prevention of Cruelty to Animals",
  "sources": [
    {
      "id": "1766291f-f2f5-5f01-b1bb-fc95501ab163",
      "type": "faq",
      "title": "FAQ: What does ASPCA stand for?",
      "url": "/v1/projects/95/faqs/1766291f-f2f5-5f01-b1bb-fc95501ab163",
      "relevance_score": 0.95
    }
  ],
  "project_id": "95",
  "timestamp": "2025-08-11T22:04:11.265860"
}
```

### Source Retrieval Endpoints

```bash
# Get FAQ by ID (returns attachment file if exists, otherwise JSON)
curl "http://localhost:8000/v1/projects/95/faqs/1766291f-f2f5-5f01-b1bb-fc95501ab163"

# Get KB entry by ID (returns attachment file if exists, otherwise JSON)  
curl "http://localhost:8000/v1/projects/95/kb/b91e6501-9bc9-5f31-b0e3-368eb49d8e12"

# List available projects
curl "http://localhost:8000/projects"
```

## 🧪 Testing

### Test Core Functionality
```bash
# Test basic models and data processing
python test_core.py

# Test AI worker functionality  
python test_ai_worker.py
```

### Run Complete Integration Test
```bash
# Copy sample data and run full test
cp -r sample_data/* .
python prebuild_kb.py
python test_ai_worker.py
```

## 📦 Dependencies

### Required (core functionality)
```bash
pip install fastapi uvicorn pydantic
```

### Optional (enhanced search)
```bash
pip install sentence-transformers faiss-cpu whoosh numpy
```

### Optional (AI-powered answers)
```bash
pip install openai
export OPENAI_API_KEY=your_key_here
```

## 🔧 Configuration

### Environment Variables
```bash
# Server configuration
export HOST=0.0.0.0
export PORT=8000

# Optional: OpenAI API for enhanced answer generation
export OPENAI_API_KEY=your_key_here
```

## 🎯 Key Features

- **Simple Two-Script Architecture**: Just prebuild_kb.py and ai_worker.py
- **Hybrid Search**: Combines dense (semantic) and sparse (keyword) search when dependencies available
- **Source Citations**: All answers include clickable source links
- **File Attachments**: Serves original files when available
- **Graceful Degradation**: Works with minimal dependencies, enhanced with full dependencies
- **Fast Setup**: File-based storage, no external databases required

## 📋 Scripts Overview

### prebuild_kb.py
- Processes FAQ and KB JSON files
- Builds FAISS dense vector indexes (semantic search)
- Builds Whoosh sparse text indexes (keyword search) 
- Creates metadata for change detection
- Works with or without ML dependencies

### ai_worker.py
- FastAPI server with query endpoints
- Loads prebuilt indexes for fast search
- Returns answers with source citations
- Serves attachment files when available
- Handles multiple projects

## 📋 Sample Data Format

### FAQ Format (`<project_id>.faq.json`)
```json
[
  {
    "id": "uuid-here",
    "question": "What does ASPCA stand for?",
    "answer": "American Society for the Prevention of Cruelty to Animals",
    "created_at": "2025-08-11T21:20:31.773424",
    "updated_at": "2025-08-11T21:20:31.773426",
    "source": "manual",
    "source_file": null
  }
]
```

### Knowledge Base Format (`<project_id>.kb.json`)
```json
[
  {
    "id": "uuid-here", 
    "article": "Mission and History",
    "content": "The ASPCA was founded in 1866...",
    "created_at": "2025-08-11T21:20:31.774353",
    "updated_at": "2025-08-11T21:20:31.774355",
    "source": "upload",
    "source_file": "aspca_mission.txt",
    "chunk_index": null
  }
]
```

## 🎯 Key Features

- **Simplified Architecture**: Just two scripts - prebuild and worker
- **Hybrid Search**: Combines dense (semantic) and sparse (keyword) search
- **Source Citations**: All answers include clickable source links
- **File Attachments**: Serves original files when available
- **Metadata Tracking**: Checksums and versions for change detection
- **Graceful Degradation**: Works with or without ML dependencies

## 🔄 Migration from Complex Setup

The new simplified structure replaces the previous complex FastAPI application with multiple endpoints. Key changes:

- **Removed**: Complex ingestion, project management, batch operations
- **Kept**: Core query functionality with source citations  
- **Added**: Prebuild step for better performance
- **Simplified**: File-based storage, minimal dependencies

## 📚 Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   prebuild_kb   │    │   ai_worker     │
│                 │    │                 │
│ • Load FAQ/KB   │    │ • Load indexes  │
│ • Build indexes │    │ • Serve queries │
│ • Save metadata │    │ • Return sources│
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│             File System                 │
│                                         │
│ proj_mapping.txt                        │
│ 95/                                     │
│   ├── 95.faq.json                      │
│   ├── 95.kb.json                       │
│   ├── attachments/                     │
│   └── index/                           │
│       ├── dense/                       │
│       ├── sparse/                      │
│       └── meta.json                    │
└─────────────────────────────────────────┘
```

This optimized structure focuses on the core use case: fast, accurate answers with proper source attribution.
