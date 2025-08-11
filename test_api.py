#!/usr/bin/env python3
"""
Simple test script for DARKBO API
"""
import requests
import json
import tempfile
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the DARKBO API endpoints"""
    
    print("🚀 Testing DARKBO API...")
    
    # Test 1: Health check
    print("\n1. Health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        return
    
    # Test 2: Create project
    print("\n2. Creating test project...")
    project_data = {
        "project_id": "175",
        "project_name": "ACLU"
    }
    
    response = requests.post(f"{BASE_URL}/v1/projects", json=project_data)
    if response.status_code == 200:
        print("✅ Project created successfully")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ Project creation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return
    
    # Test 3: List projects
    print("\n3. Listing projects...")
    response = requests.get(f"{BASE_URL}/v1/projects")
    if response.status_code == 200:
        projects = response.json()
        print("✅ Projects listed successfully")
        print(f"   Found {len(projects['projects'])} projects")
    else:
        print(f"❌ Failed to list projects: {response.status_code}")
    
    # Test 4: Add FAQ
    print("\n4. Adding FAQ entries...")
    faq_data = {
        "items": [
            {
                "question": "When are phones staffed?",
                "answer": "Mon–Fri 9–5 MT"
            },
            {
                "question": "What is the ACLU?", 
                "answer": "The American Civil Liberties Union is a nonprofit organization that defends civil rights and liberties."
            }
        ],
        "replace": False
    }
    
    response = requests.post(f"{BASE_URL}/v1/projects/175/faqs:batch_upsert", json=faq_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ FAQs added successfully")
        print(f"   Created: {result['created']}, Updated: {result['updated']}")
    else:
        print(f"❌ Failed to add FAQs: {response.status_code}")
        print(f"   Error: {response.text}")
    
    # Test 5: Create and upload document
    print("\n5. Ingesting a test document...")
    
    # Create a test document
    test_content = """
    ACLU Information Guide
    
    About the ACLU
    The American Civil Liberties Union (ACLU) is a non-profit organization founded in 1920. 
    Our mission is to defend and preserve the individual rights and liberties guaranteed by the 
    Constitution and laws of the United States.
    
    Our Work
    We work through litigation, advocacy, and public education to protect civil liberties.
    Key areas include:
    - Freedom of speech and expression
    - Religious liberty
    - Privacy rights
    - Equal protection under law
    
    Contact Information
    Phone: Available Monday through Friday, 9 AM to 5 PM Mountain Time
    Website: www.aclu.org
    """
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = f.name
    
    try:
        with open(temp_file_path, 'rb') as f:
            files = {'files': ('aclu_guide.txt', f, 'text/plain')}
            data = {
                'type': 'kb',
                'split': 'headings',
                'max_chunk_chars': 1200
            }
            
            response = requests.post(f"{BASE_URL}/v1/projects/175/ingest", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Document ingested successfully")
                print(f"   Created: {len(result['created'])} entries")
                print(f"   Job ID: {result['job_id']}")
            else:
                print(f"❌ Failed to ingest document: {response.status_code}")
                print(f"   Error: {response.text}")
    
    finally:
        # Clean up temp file
        Path(temp_file_path).unlink()
    
    # Test 6: Query the system
    print("\n6. Testing queries...")
    
    test_queries = [
        {
            "project_id": "175",
            "question": "What are your phone hours on weekdays?",
            "mode": "auto",
            "strict_citations": True
        },
        {
            "project_id": "175", 
            "question": "What does the ACLU do?",
            "mode": "auto",
            "strict_citations": True
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query['question']}")
        response = requests.post(f"{BASE_URL}/v1/query", json=query)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Answer: {result['answer'][:100]}...")
            print(f"   Mode: {result['mode']}, Confidence: {result['confidence']:.2f}")
            print(f"   Citations: {len(result['citations'])}")
        else:
            print(f"   ❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    print("\n🎉 API testing completed!")

if __name__ == "__main__":
    test_api()