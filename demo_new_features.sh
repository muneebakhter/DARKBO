#!/bin/bash
"""
DARKBO Document Ingestion and FAQ Management Demo
Demonstrates the new versioned index system with document upload and FAQ management
"""

echo "🚀 DARKBO Document Ingestion and FAQ Management Demo"
echo "============================================================"
echo ""

# Change to sample_data directory
cd sample_data

echo "📋 1. Checking current project status..."
curl -s "http://localhost:8000/projects" | python -m json.tool
echo ""

echo "📊 2. Checking build status for ASPCA project (95)..."
curl -s "http://localhost:8000/projects/95/build-status" | python -m json.tool
echo ""

echo "❓ 3. Adding a new FAQ about document ingestion..."
curl -X POST "http://localhost:8000/projects/95/faqs" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I upload documents to the knowledge base?",
    "answer": "You can upload PDF and DOCX documents using the document ingestion endpoint. The system will automatically extract text, create chunks, and rebuild the search indexes."
  }' | python -m json.tool
echo ""

echo "⏳ 4. Waiting for index rebuild to complete..."
sleep 3

echo "📊 5. Checking build status after FAQ addition..."
curl -s "http://localhost:8000/projects/95/build-status" | python -m json.tool  
echo ""

echo "🔍 6. Testing query for the new FAQ..."
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "95",
    "question": "How do I upload documents?"
  }' | python -m json.tool
echo ""

echo "❓ 7. Adding another FAQ about index versioning..."
curl -X POST "http://localhost:8000/projects/95/faqs" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the index versioning system work?",
    "answer": "The system uses versioned indexes to allow atomic updates. When new content is added, a new index version is built in the background while queries continue using the current version. Once complete, the system switches to the new version atomically."
  }' | python -m json.tool
echo ""

echo "🔄 8. Manually triggering index rebuild..."
curl -X POST "http://localhost:8000/projects/95/rebuild-indexes" | python -m json.tool
echo ""

echo "🔍 9. Testing complex query that should match multiple sources..."
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "95", 
    "question": "Tell me about ASPCA and how the system works"
  }' | python -m json.tool
echo ""

echo "📈 10. Final build status check..."
curl -s "http://localhost:8000/projects/95/build-status" | python -m json.tool
echo ""

echo "✅ Demo completed successfully!"
echo ""
echo "📋 Summary of features demonstrated:"
echo "  ✅ FAQ creation via API"
echo "  ✅ Automatic index rebuilding after content changes"
echo "  ✅ Versioned index system with atomic updates"
echo "  ✅ Build status monitoring"
echo "  ✅ Manual index rebuild triggers"
echo "  ✅ Query system integration with new content"
echo "  ✅ Source citation and relevance scoring"
echo ""
echo "🚀 To enable full document upload functionality:"
echo "   pip install python-multipart python-docx PyPDF2"
echo ""
echo "📄 Document ingestion endpoint will be available at:"
echo "   POST /projects/{project_id}/documents"
echo ""