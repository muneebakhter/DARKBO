#!/usr/bin/env python3
"""
Simple test script for the new DARKBO AI Worker endpoints
"""

import json
import requests
import time
import subprocess
import os
import signal
from pathlib import Path

BASE_URL = "http://localhost:8000"

def start_server():
    """Start the AI worker server"""
    print("🚀 Starting AI Worker server...")
    
    # Check if FastAPI is available
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
        return None
    
    # Start server in background
    process = subprocess.Popen(
        ["python", "ai_worker.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for server to start
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server started successfully")
                return process
        except:
            time.sleep(1)
    
    print("❌ Server failed to start")
    return None

def test_endpoints():
    """Test all the AI worker endpoints"""
    print("\n🧪 Testing AI Worker Endpoints")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            data = response.json()
            print(f"   Status: {data.get('status')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test 2: List projects
    print("\n2. Testing projects endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/projects")
        if response.status_code == 200:
            data = response.json()
            projects = data.get('projects', [])
            print(f"✅ Found {len(projects)} projects")
            for project in projects:
                print(f"   - {project['id']}: {project['name']}")
        else:
            print(f"❌ Projects endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Projects endpoint error: {e}")
    
    # Test 3: Query endpoint
    print("\n3. Testing query endpoint...")
    test_queries = [
        ("95", "What does ASPCA stand for?"),
        ("95", "How can I report animal cruelty?"),
        ("175", "What is the ACLU?")
    ]
    
    for project_id, question in test_queries:
        try:
            payload = {
                "project_id": project_id,
                "question": question
            }
            response = requests.post(
                f"{BASE_URL}/query", 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Query successful for: {question}")
                print(f"   Answer: {data.get('answer', 'No answer')[:100]}...")
                sources = data.get('sources', [])
                print(f"   Sources: {len(sources)} found")
                for source in sources[:2]:  # Show first 2 sources
                    print(f"     - {source.get('title', 'Unknown')[:50]}...")
                    print(f"       URL: {source.get('url', 'No URL')}")
            else:
                print(f"❌ Query failed for '{question}': {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Query error for '{question}': {e}")
    
    # Test 4: FAQ endpoint
    print("\n4. Testing FAQ endpoint...")
    try:
        # Get a FAQ ID from sample data
        with open("95/95.faq.json", 'r') as f:
            faqs = json.load(f)
            if faqs:
                faq_id = faqs[0]['id']
                
                response = requests.get(f"{BASE_URL}/v1/projects/95/faqs/{faq_id}")
                if response.status_code == 200:
                    print("✅ FAQ endpoint successful")
                    # Check if it's JSON or file
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        data = response.json()
                        print(f"   Question: {data.get('question', 'Unknown')[:60]}...")
                        print(f"   Answer: {data.get('answer', 'Unknown')[:60]}...")
                    else:
                        print(f"   File content ({content_type}): {len(response.content)} bytes")
                else:
                    print(f"❌ FAQ endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ FAQ endpoint error: {e}")
    
    # Test 5: KB endpoint
    print("\n5. Testing KB endpoint...")
    try:
        # Get a KB ID from sample data
        with open("95/95.kb.json", 'r') as f:
            kb_entries = json.load(f)
            if kb_entries:
                kb_id = kb_entries[0]['id']
                
                response = requests.get(f"{BASE_URL}/v1/projects/95/kb/{kb_id}")
                if response.status_code == 200:
                    print("✅ KB endpoint successful")
                    # Check if it's JSON or file
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        data = response.json()
                        print(f"   Article: {data.get('article', 'Unknown')}")
                        print(f"   Content: {data.get('content', 'Unknown')[:60]}...")
                    else:
                        print(f"   File content ({content_type}): {len(response.content)} bytes")
                else:
                    print(f"❌ KB endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ KB endpoint error: {e}")

def main():
    """Main test function"""
    print("🧪 DARKBO AI Worker Integration Test")
    print("=" * 50)
    
    # Check if we have sample data
    if not Path("proj_mapping.txt").exists():
        print("❌ No proj_mapping.txt found. Run: cp -r sample_data/* .")
        return
    
    if not Path("95").exists() or not Path("175").exists():
        print("❌ Project directories not found. Run: cp -r sample_data/* .")
        return
    
    # Check if indexes are built
    if not Path("95/index/meta.json").exists():
        print("📦 Building indexes first...")
        os.system("python prebuild_kb.py")
    
    # Try to start server and test
    server_process = start_server()
    if server_process:
        try:
            time.sleep(2)  # Give server time to fully start
            test_endpoints()
        finally:
            print("\n🛑 Stopping server...")
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait()
            print("✅ Server stopped")
    else:
        print("\n💡 To test manually:")
        print("1. Install dependencies: pip install fastapi uvicorn")
        print("2. Start server: python ai_worker.py") 
        print("3. Test endpoints with curl or browser")

if __name__ == "__main__":
    main()