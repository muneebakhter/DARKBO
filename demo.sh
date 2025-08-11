#!/bin/bash
# DARKBO Quick Start Demo Script
# This script demonstrates the complete workflow of the optimized DARKBO system

echo "🚀 DARKBO Quick Start Demo"
echo "========================="

# Step 1: Install minimal dependencies
echo "📦 Step 1: Install minimal dependencies..."
pip install fastapi uvicorn pydantic

# Step 2: Generate sample data
echo "📁 Step 2: Generate sample data..."
python3 create_sample_data.py

# Step 3: Build knowledge base indexes  
echo "🔧 Step 3: Building knowledge base indexes..."
cd sample_data
python3 ../prebuild_kb.py

# Step 4: Start the AI worker in background
echo "🚀 Step 4: Starting AI worker server..."
python3 ../ai_worker.py &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 5

# Step 5: Test the system
echo "🧪 Step 5: Testing the system..."

echo ""
echo "1. List available projects:"
curl -s http://localhost:8000/projects | python -m json.tool

echo ""
echo "2. Ask a question about ASPCA:"
curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What does ASPCA stand for?"}' | python -m json.tool

echo ""
echo "3. Ask a question about ACLU:"
curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "175", "question": "What is the ACLU?"}' | python -m json.tool

echo ""
echo "4. Get FAQ source directly:"
curl -s "http://localhost:8000/v1/projects/95/faqs/1766291f-f2f5-5f01-b1bb-fc95501ab163" | python -m json.tool

# Cleanup
echo ""
echo "🛑 Stopping server..."
kill $SERVER_PID 2>/dev/null || true
cd ..

echo ""
echo "✅ Demo completed!"
echo ""
echo "🎯 Key takeaways:"
echo "  - Generate sample data with: python3 create_sample_data.py"
echo "  - Two scripts only: prebuild_kb.py + ai_worker.py"
echo "  - Works with minimal dependencies (no ML libraries needed for basic functionality)"
echo "  - Provides answers with source citations"
echo "  - Serves attachment files when available"
echo "  - Graceful fallback search when advanced dependencies unavailable"
echo ""
echo "📚 Next steps:"
echo "  - Install enhanced dependencies: pip install sentence-transformers faiss-cpu whoosh"
echo "  - Re-run prebuild_kb.py for vector search capabilities"
echo "  - Add your own FAQ/KB data in project directories"