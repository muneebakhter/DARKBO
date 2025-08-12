# DARKBO - Document Augmented Retrieval Knowledge Base Operator

A simplified AI knowledge base system with external tools support and **enhanced embedding & indexing**:

1. **prebuild_kb.py** - Builds vector stores and search indexes from FAQ/KB data with BGE embeddings, HNSW indexing, and document chunking
2. **ai_worker.py** - FastAPI server that answers questions with source citations, external tools, RRF fusion, and cross-encoder re-ranking

## 🚀 Enhanced Features (NEW)

DARKBO now includes significant improvements to embedding and indexing:

- **🧠 BGE Embeddings**: Upgraded from MiniLM to BAAI/bge-small-en for better semantic accuracy
- **📊 HNSW Indexing**: Scalable similarity search with logarithmic complexity (vs flat indexes)  
- **📄 Document Chunking**: Intelligent 400±50 token chunks with overlap for better granularity
- **🔀 Rank Fusion**: Reciprocal Rank Fusion (RRF) combines semantic + keyword search optimally
- **🎯 Cross-Encoder Re-ranking**: Final relevance scoring with cross-encoder models
- **⚙️ Consistent Normalization**: L2 normalization throughout for efficient cosine similarity

See [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) for complete technical details.

## 🛠️ New Tools Framework

DARKBO now supports external tools that can enhance query responses with real-time data:

- **DateTime Tool**: Get current date, time, and timezone information
- **Web Search Tool**: Search the web for current information using DuckDuckGo
- **Extensible Framework**: Easy to add new tools for additional capabilities

### Tools Usage Examples

```bash
# Get current time
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What time is it now?"}'

# Get current date  
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What date is today?"}'

# Web search (combined with KB knowledge)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What is machine learning?"}'
```

### Direct Tool Access

```bash
# List available tools
curl "http://localhost:8000/tools"

# Execute datetime tool directly
curl -X POST "http://localhost:8000/tools/datetime" \
  -H "Content-Type: application/json" \
  -d '{"format": "%Y-%m-%d %H:%M:%S"}'

# Execute web search tool directly  
curl -X POST "http://localhost:8000/tools/web_search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Python programming", "max_results": 3}'
```

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

# Full installation (with enhanced vector search capabilities)
pip install fastapi uvicorn pydantic sentence-transformers faiss-cpu whoosh numpy transformers

# Optional: Enhanced answer generation
pip install openai
```

### 1. Generate Sample Data

Use the included script to create sample data:

```bash
# Generate sample ACLU and ASPCA projects with FAQs, KB entries, and attachments
python3 create_sample_data.py
```

This creates a `sample_data/` directory with:
- Project directories (175/ for ACLU, 95/ for ASPCA)  
- FAQ and KB JSON files
- Sample attachments
- Project mapping file

### 1a. Alternative: Prepare Your Own Data

If you want to use your own data instead of samples:

```bash
# Create project mapping file
echo -e "95\tASPCA\n175\tACLU" > proj_mapping.txt

# Create project directories with FAQ and KB data
mkdir -p 95 175
# Add your 95.faq.json, 95.kb.json, 175.faq.json, 175.kb.json files
```

### 2. Build Knowledge Base Indexes

```bash
# Change to the sample_data directory (or your data directory)
cd sample_data

# Build indexes for all projects
python3 ../prebuild_kb.py
```

This will:
- Load FAQ and KB data from project directories
- Build dense vector indexes (if dependencies available)
- Build sparse text indexes (if dependencies available)  
- Create metadata for change detection
- Output progress for each project

### 3. Start the AI Worker Server

```bash
# Start server from the data directory
python3 ../ai_worker.py
```

The server will start on `http://localhost:8000` with the following endpoints:
- `POST /query` - Ask questions and get answers with sources
- `GET /projects` - List available projects
- `GET /v1/projects/{project_id}/faqs/{faq_id}` - Get FAQ by ID
- `GET /v1/projects/{project_id}/kb/{kb_id}` - Get KB entry by ID

### 4. Test the System

```bash
# Generate sample data and test the complete system
python3 create_sample_data.py
cd sample_data
python3 ../prebuild_kb.py  
python3 ../ai_worker.py

# In another terminal, test queries:
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What does ASPCA stand for?"}'
```

### 5. Run Complete Demo

```bash
# Run the included demo script that shows the full workflow
./demo.sh
```

## 📡 API Endpoints

### Core Query Endpoint (Enhanced with Tools)
```bash
# Ask a question (automatically uses tools when appropriate)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "95",
    "question": "What time is it now?"
  }'
```

Response includes answer, sources, and tools used:
```json
{
  "answer": "The current time is 2025-08-11T23:30:35.898854+00:00.",
  "sources": [...],
  "project_id": "95", 
  "timestamp": "2025-08-11T23:30:35.898938",
  "tools_used": [
    {
      "tool_name": "datetime",
      "parameters": {},
      "result": {
        "success": true,
        "data": {
          "current_datetime": "2025-08-11T23:30:35.898854+00:00",
          "weekday": "Monday",
          ...
        }
      },
      "success": true,
      "execution_time": 4.076957702636719e-05
    }
  ]
}
```

### Tools Management Endpoints

```bash
# List available tools
curl "http://localhost:8000/tools"

# Execute a specific tool
curl -X POST "http://localhost:8000/tools/datetime" \
  -H "Content-Type: application/json" \
  -d '{"format": "%B %d, %Y"}'

# Web search
curl -X POST "http://localhost:8000/tools/web_search" \
  -H "Content-Type: application/json" \  
  -d '{"query": "latest AI news", "max_results": 5}'
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
# Generate sample data and test complete workflow
python3 create_sample_data.py
cd sample_data
python3 ../prebuild_kb.py
python3 ../ai_worker.py

# In another terminal, test queries:
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What does ASPCA stand for?"}'
```

### Run Complete Integration Test
```bash
# Run automated demo that tests the full system
./demo.sh
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
- **External Tools Support**: DateTime and web search tools with extensible framework
- **Enhanced Hybrid Vector Store**: BGE embeddings + HNSW indexing + document chunking + RRF fusion
- **Intelligent Search**: Combines dense (semantic) and sparse (keyword) search with cross-encoder re-ranking
- **Source Citations**: All answers include clickable source links
- **Tool-Enhanced Responses**: Automatically uses tools to provide current information
- **File Attachments**: Serves original files when available
- **Graceful Degradation**: Works with minimal dependencies, enhanced with full dependencies
- **Fast Setup**: File-based storage, no external databases required

## 📋 Scripts Overview

### prebuild_kb.py
- Processes FAQ and KB JSON files with intelligent chunking
- Builds enhanced FAISS HNSW vector indexes with BGE embeddings  
- Builds enhanced Whoosh sparse text indexes with chunked content
- Creates comprehensive metadata for change detection and performance tracking
- Works with or without ML dependencies (graceful degradation)

### ai_worker.py
- FastAPI server with query endpoints
- Loads prebuilt indexes for fast hybrid search
- Returns answers with source citations using dense + sparse + basic search
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
- **Hybrid Vector Store**: Combines dense (FAISS semantic) and sparse (Whoosh keyword) search
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
