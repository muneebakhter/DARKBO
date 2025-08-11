#!/usr/bin/env python3
"""
Simplified prebuild script for knowledge graphs and vector stores.
Creates indexes per project folder based on FAQ and KB data.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    from whoosh.index import create_index
    from whoosh.fields import Schema, TEXT, ID, STORED
    from whoosh.writing import AsyncWriter
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("⚠️  Heavy dependencies not available. Creating basic indexes only.")

from api.models_simple import FAQEntry, KBEntry


class IndexBuilder:
    """Builds dense and sparse indexes for a project"""
    
    def __init__(self, project_id: str, base_dir: str = "."):
        self.project_id = project_id
        self.base_dir = Path(base_dir)
        self.project_dir = self.base_dir / project_id
        self.index_dir = self.project_dir / "index"
        self.dense_dir = self.index_dir / "dense"
        self.sparse_dir = self.index_dir / "sparse"
        
        # Create directories
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models if available
        self.embedding_model = None
        if HAS_DEPS:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
    
    def load_project_data(self) -> tuple[List[FAQEntry], List[KBEntry]]:
        """Load FAQ and KB data for the project"""
        faqs = []
        kb_entries = []
        
        # Load FAQ data
        faq_file = self.project_dir / f"{self.project_id}.faq.json"
        if faq_file.exists():
            with open(faq_file, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                faqs = [FAQEntry.from_dict(item) for item in faq_data]
        
        # Load KB data  
        kb_file = self.project_dir / f"{self.project_id}.kb.json"
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
                kb_entries = [KBEntry.from_dict(item) for item in kb_data]
        
        return faqs, kb_entries
    
    def build_dense_index(self, faqs: List[FAQEntry], kb_entries: List[KBEntry]) -> Optional[str]:
        """Build FAISS dense vector index"""
        if not HAS_DEPS or not self.embedding_model:
            print(f"  Skipping dense index for {self.project_id} (dependencies not available)")
            return None
            
        try:
            # Collect all text content
            texts = []
            metadata = []
            
            # Add FAQ content
            for faq in faqs:
                text = f"Q: {faq.question}\nA: {faq.answer}"
                texts.append(text)
                metadata.append({
                    'id': faq.id,
                    'type': 'faq',
                    'question': faq.question,
                    'answer': faq.answer
                })
            
            # Add KB content
            for kb in kb_entries:
                text = f"Title: {kb.article}\nContent: {kb.content}"
                texts.append(text)
                metadata.append({
                    'id': kb.id,
                    'type': 'kb',
                    'article': kb.article,
                    'content': kb.content
                })
            
            if not texts:
                print(f"  No content to index for {self.project_id}")
                return None
                
            # Generate embeddings
            print(f"  Generating embeddings for {len(texts)} items...")
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            # Save index and metadata
            index_file = self.dense_dir / "faiss.index"
            metadata_file = self.dense_dir / "metadata.json"
            
            faiss.write_index(index, str(index_file))
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Dense index built: {len(texts)} vectors, dimension {dimension}")
            return str(index_file)
            
        except Exception as e:
            print(f"  ❌ Error building dense index: {e}")
            return None
    
    def build_sparse_index(self, faqs: List[FAQEntry], kb_entries: List[KBEntry]) -> Optional[str]:
        """Build Whoosh sparse text index"""
        if not HAS_DEPS:
            print(f"  Skipping sparse index for {self.project_id} (dependencies not available)")
            return None
            
        try:
            # Define schema
            schema = Schema(
                id=ID(stored=True),
                type=STORED(),
                content=TEXT(stored=True),
                title=TEXT(stored=True),
                question=TEXT(stored=True),
                answer=TEXT(stored=True)
            )
            
            # Create index
            index = create_index(schema, str(self.sparse_dir))
            
            # Add documents
            writer = index.writer()
            
            # Add FAQ entries
            for faq in faqs:
                writer.add_document(
                    id=faq.id,
                    type="faq",
                    content=f"{faq.question} {faq.answer}",
                    title="",
                    question=faq.question,
                    answer=faq.answer
                )
            
            # Add KB entries
            for kb in kb_entries:
                writer.add_document(
                    id=kb.id,
                    type="kb", 
                    content=f"{kb.article} {kb.content}",
                    title=kb.article,
                    question="",
                    answer=""
                )
            
            writer.commit()
            
            total_docs = len(faqs) + len(kb_entries)
            print(f"  ✅ Sparse index built: {total_docs} documents")
            return str(self.sparse_dir)
            
        except Exception as e:
            print(f"  ❌ Error building sparse index: {e}")
            return None
    
    def create_metadata(self, faqs: List[FAQEntry], kb_entries: List[KBEntry], 
                       dense_path: Optional[str], sparse_path: Optional[str]) -> Dict:
        """Create index metadata"""
        
        # Calculate content checksums for change detection
        faq_content = json.dumps([faq.to_dict() for faq in faqs], sort_keys=True)
        kb_content = json.dumps([kb.to_dict() for kb in kb_entries], sort_keys=True)
        
        faq_checksum = hashlib.md5(faq_content.encode()).hexdigest()
        kb_checksum = hashlib.md5(kb_content.encode()).hexdigest()
        
        metadata = {
            "project_id": self.project_id,
            "created_at": datetime.utcnow().isoformat(),
            "counts": {
                "faqs": len(faqs),
                "kb_entries": len(kb_entries),
                "total": len(faqs) + len(kb_entries)
            },
            "checksums": {
                "faqs": faq_checksum,
                "kb_entries": kb_checksum
            },
            "indexes": {
                "dense": {
                    "type": "faiss",
                    "path": dense_path,
                    "available": dense_path is not None
                },
                "sparse": {
                    "type": "whoosh", 
                    "path": sparse_path,
                    "available": sparse_path is not None
                }
            },
            "versions": {
                "schema": "1.0",
                "embedding_model": "all-MiniLM-L6-v2" if HAS_DEPS else None
            }
        }
        
        # Save metadata
        meta_file = self.index_dir / "meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def build_all_indexes(self) -> Dict:
        """Build all indexes for the project"""
        print(f"\n📁 Building indexes for project {self.project_id}...")
        
        # Load data
        faqs, kb_entries = self.load_project_data()
        total_items = len(faqs) + len(kb_entries)
        
        if total_items == 0:
            print(f"  ⚠️  No data found for project {self.project_id}")
            return {"error": "No data found"}
        
        print(f"  📊 Found {len(faqs)} FAQs and {len(kb_entries)} KB entries")
        
        # Build indexes
        dense_path = self.build_dense_index(faqs, kb_entries)
        sparse_path = self.build_sparse_index(faqs, kb_entries)
        
        # Create metadata
        metadata = self.create_metadata(faqs, kb_entries, dense_path, sparse_path)
        
        print(f"  ✅ Index building complete for {self.project_id}")
        return metadata


def load_project_mapping(mapping_file: str = "proj_mapping.txt") -> Dict[str, str]:
    """Load project ID to name mapping"""
    projects = {}
    
    if not os.path.exists(mapping_file):
        print(f"❌ Project mapping file not found: {mapping_file}")
        return projects
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                project_id, name = line.split('\t', 1)
                projects[project_id.strip()] = name.strip()
    
    return projects


def main():
    """Main prebuild function"""
    print("🚀 DARKBO Knowledge Base Prebuild")
    print("=" * 50)
    
    # Load project mapping
    projects = load_project_mapping()
    if not projects:
        print("❌ No projects found in proj_mapping.txt")
        return
    
    print(f"📋 Found {len(projects)} projects to process")
    
    # Build indexes for each project
    results = {}
    for project_id, project_name in projects.items():
        try:
            builder = IndexBuilder(project_id)
            metadata = builder.build_all_indexes()
            results[project_id] = metadata
        except Exception as e:
            print(f"❌ Error processing project {project_id}: {e}")
            results[project_id] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Build Summary")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    for project_id, result in results.items():
        project_name = projects.get(project_id, "Unknown")
        if "error" in result:
            print(f"❌ {project_id} ({project_name}): {result['error']}")
            failed += 1
        else:
            counts = result.get('counts', {})
            total = counts.get('total', 0)
            dense_ok = result.get('indexes', {}).get('dense', {}).get('available', False)
            sparse_ok = result.get('indexes', {}).get('sparse', {}).get('available', False)
            
            status = []
            if dense_ok:
                status.append("dense")
            if sparse_ok:
                status.append("sparse")
            status_str = "+".join(status) if status else "metadata-only"
            
            print(f"✅ {project_id} ({project_name}): {total} items, {status_str}")
            successful += 1
    
    print(f"\n🎉 Completed: {successful} successful, {failed} failed")
    
    if not HAS_DEPS:
        print("\n💡 To enable full indexing, install dependencies:")
        print("   pip install sentence-transformers faiss-cpu whoosh")


if __name__ == "__main__":
    main()