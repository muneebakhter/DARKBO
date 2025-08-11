# DARKBO - Document Augmented Retrieval Knowledge Base Operator

A simplified AI knowledge base system with two main components:

1. **Knowledge Base Prebuild Script** - Prepares vector stores and indexes
2. **AI Worker** - Serves queries with intelligent answers and source citations

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

### 1. Prepare Your Data

Create a project mapping file:
```bash
echo -e "95\tASPCA\n175\tACLU" > proj_mapping.txt
```

Create project directories with FAQ and KB data:
```bash
mkdir -p 95 175
# Add 95.faq.json, 95.kb.json, etc.
```

### 2. Prebuild Knowledge Base Indexes

```bash
# Basic prebuild (metadata only)
python prebuild_kb.py

# Full prebuild with vector search (requires dependencies)
pip install sentence-transformers faiss-cpu whoosh
python prebuild_kb.py
```

### 3. Start the AI Worker

```bash
# Install FastAPI
pip install fastapi uvicorn

# Start the server
python ai_worker.py
```

The AI worker will be available at `http://localhost:8000`

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
# Test without dependencies
python test_core.py

# Test AI worker functionality
python test_ai_worker.py
```

### Test with Sample Data
```bash
# Copy sample data
cp -r sample_data/* .

# Prebuild indexes
python prebuild_kb.py

# Test queries
python test_ai_worker.py
```

## 📦 Dependencies

### Minimal (metadata-only indexes)
```bash
pip install fastapi uvicorn
```

### Full functionality (vector search)
```bash
pip install sentence-transformers faiss-cpu whoosh openai
```

## 🔧 Configuration

### Environment Variables
```bash
# Server configuration
export HOST=0.0.0.0
export PORT=8000

# OpenAI API (optional, for advanced answer generation)
export OPENAI_API_KEY=your_key_here
```

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
