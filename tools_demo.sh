#!/bin/bash
# 
# DARKBO Tools Demo Script
# Demonstrates the tools functionality added to DARKBO
#

echo "🚀 DARKBO Tools Demo"
echo "===================="
echo ""

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ DARKBO server is not running!"
    echo "Please start it first with: cd sample_data && python3 ../ai_worker.py"
    exit 1
fi

echo "✅ DARKBO server is running"
echo ""

# Test 1: List available tools
echo "📋 Test 1: List available tools"
echo "curl -X GET 'http://localhost:8000/tools'"
curl -s -X GET "http://localhost:8000/tools" | python3 -m json.tool
echo ""

# Test 2: DateTime tool direct usage
echo "🕐 Test 2: DateTime tool - direct usage"
echo "curl -X POST 'http://localhost:8000/tools/datetime' -d '{}'"
curl -s -X POST "http://localhost:8000/tools/datetime" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool
echo ""

# Test 3: DateTime tool with custom format
echo "📅 Test 3: DateTime tool - custom format"
echo "curl -X POST 'http://localhost:8000/tools/datetime' -d '{\"format\": \"%B %d, %Y at %I:%M %p\"}'"
curl -s -X POST "http://localhost:8000/tools/datetime" \
  -H "Content-Type: application/json" \
  -d '{"format": "%B %d, %Y at %I:%M %p"}' | python3 -m json.tool
echo ""

# Test 4: Web search tool direct usage  
echo "🔍 Test 4: Web search tool - direct usage"
echo "curl -X POST 'http://localhost:8000/tools/web_search' -d '{\"query\": \"Python programming\", \"max_results\": 2}'"
curl -s -X POST "http://localhost:8000/tools/web_search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Python programming", "max_results": 2}' | python3 -m json.tool
echo ""

# Test 5: Query with datetime tool integration
echo "⏰ Test 5: Query - 'What time is it now?' (uses datetime tool)"
echo "curl -X POST 'http://localhost:8000/query' -d '{\"project_id\": \"95\", \"question\": \"What time is it now?\"}'"
curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What time is it now?"}' | python3 -m json.tool
echo ""

# Test 6: Query with date tool integration
echo "📆 Test 6: Query - 'What date is today?' (uses datetime tool)"
echo "curl -X POST 'http://localhost:8000/query' -d '{\"project_id\": \"95\", \"question\": \"What date is today?\"}'"
curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What date is today?"}' | python3 -m json.tool
echo ""

# Test 7: Regular KB query (no tools)
echo "📚 Test 7: Regular KB query - 'What does ASPCA stand for?' (KB + web search)"
echo "curl -X POST 'http://localhost:8000/query' -d '{\"project_id\": \"95\", \"question\": \"What does ASPCA stand for?\"}'"
curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project_id": "95", "question": "What does ASPCA stand for?"}' | python3 -m json.tool
echo ""

# Test 8: Check main API endpoints
echo "🏠 Test 8: Main API info"
echo "curl -X GET 'http://localhost:8000/'"
curl -s -X GET "http://localhost:8000/" | python3 -m json.tool
echo ""

echo "🎉 DARKBO Tools Demo Complete!"
echo ""
echo "Key Features Demonstrated:"
echo "- ✅ DateTime tool for current time/date"
echo "- ✅ Web search tool for external information" 
echo "- ✅ Automatic tool selection in queries"
echo "- ✅ Tools integrated with knowledge base answers"
echo "- ✅ Direct tool execution endpoints"
echo "- ✅ Graceful error handling for network issues"
echo ""
echo "The system now supports external tools while maintaining full backward compatibility!"