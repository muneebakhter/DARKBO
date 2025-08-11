#!/usr/bin/env python3
"""
Simple test for the AI worker functionality
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from api.models_simple import FAQEntry, KBEntry


def test_ai_worker_core():
    """Test the core AI worker functionality without FastAPI"""
    print("🧪 Testing AI Worker Core Functionality")
    print("=" * 50)
    
    # Test 1: Load project mapping
    print("\n1. Testing project mapping...")
    projects = {}
    mapping_file = Path("proj_mapping.txt")
    
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    project_id, name = line.split('\t', 1)
                    projects[project_id.strip()] = name.strip()
        print(f"✅ Loaded {len(projects)} projects: {list(projects.keys())}")
    else:
        print("❌ No proj_mapping.txt found")
        return
    
    # Test 2: Load project data
    for project_id, project_name in projects.items():
        print(f"\n2. Testing project {project_id} ({project_name})...")
        
        project_dir = Path(project_id)
        if not project_dir.exists():
            print(f"❌ Project directory {project_id} not found")
            continue
            
        # Load FAQs
        faq_file = project_dir / f"{project_id}.faq.json"
        faqs = []
        if faq_file.exists():
            with open(faq_file, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                faqs = [FAQEntry.from_dict(item) for item in faq_data]
        
        # Load KB entries  
        kb_file = project_dir / f"{project_id}.kb.json"
        kb_entries = []
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
                kb_entries = [KBEntry.from_dict(item) for item in kb_data]
        
        print(f"  📊 Found {len(faqs)} FAQs and {len(kb_entries)} KB entries")
        
        # Test FAQ retrieval by ID
        if faqs:
            test_faq = faqs[0]
            print(f"  📝 Sample FAQ: {test_faq.question[:50]}...")
            
            # Test attachment file lookup
            attachment_file = project_dir / "attachments" / f"{test_faq.id}-faq.txt"
            if attachment_file.exists():
                print(f"  📎 Attachment exists: {attachment_file.name}")
            else:
                print(f"  📋 No attachment, would return JSON")
        
        # Test KB retrieval by ID
        if kb_entries:
            test_kb = kb_entries[0]
            print(f"  📚 Sample KB: {test_kb.article}")
            
            # Test attachment file lookup
            attachments_dir = project_dir / "attachments"
            possible_files = [
                attachments_dir / f"{test_kb.id}-kb.txt",
                attachments_dir / f"{test_kb.id}-kb.docx", 
                attachments_dir / f"{test_kb.id}-kb.pdf"
            ]
            
            found_attachment = False
            for attachment_file in possible_files:
                if attachment_file.exists():
                    print(f"  📎 Attachment exists: {attachment_file.name}")
                    found_attachment = True
                    break
            
            if not found_attachment:
                print(f"  📋 No attachment, would return JSON")
        
        # Test index metadata
        index_meta_file = project_dir / "index" / "meta.json"
        if index_meta_file.exists():
            with open(index_meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"  🗂️  Index metadata: {metadata['counts']['total']} total items")
        else:
            print(f"  ❌ No index metadata found")
    
    # Test 3: Simulate query processing
    print("\n3. Testing query simulation...")
    test_questions = [
        "What does ASPCA stand for?",
        "How can I report animal cruelty?", 
        "What is the mission of ACLU?"
    ]
    
    for question in test_questions:
        print(f"\n  Q: {question}")
        
        # Simple keyword matching simulation
        best_match = None
        best_score = 0
        best_source = None
        
        for project_id in projects.keys():
            project_dir = Path(project_id)
            
            # Check FAQs
            faq_file = project_dir / f"{project_id}.faq.json"
            if faq_file.exists():
                with open(faq_file, 'r', encoding='utf-8') as f:
                    faq_data = json.load(f)
                    for faq_item in faq_data:
                        # Simple keyword matching
                        question_lower = question.lower()
                        faq_question_lower = faq_item['question'].lower()
                        
                        # Count matching words
                        question_words = set(question_lower.split())
                        faq_words = set(faq_question_lower.split())
                        matches = len(question_words.intersection(faq_words))
                        
                        if matches > best_score:
                            best_score = matches
                            best_match = faq_item['answer']
                            best_source = {
                                'id': faq_item['id'],
                                'type': 'faq',
                                'title': f"FAQ: {faq_item['question']}",
                                'url': f"/v1/projects/{project_id}/faqs/{faq_item['id']}"
                            }
        
        if best_match:
            print(f"  A: {best_match}")
            print(f"  📄 Source: {best_source['title'][:60]}...")
            print(f"  🔗 URL: {best_source['url']}")
        else:
            print(f"  A: No relevant information found.")
    
    print("\n" + "=" * 50)
    print("🎉 Core AI Worker tests completed!")
    print("\nNext steps:")
    print("1. Install FastAPI: pip install fastapi uvicorn")
    print("2. Run: python ai_worker.py")
    print("3. Test endpoints:")
    print("   - GET /projects")
    print("   - POST /query")
    print("   - GET /v1/projects/{project_id}/faqs/{faq_id}")
    print("   - GET /v1/projects/{project_id}/kb/{kb_id}")


if __name__ == "__main__":
    test_ai_worker_core()