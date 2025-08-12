#!/usr/bin/env python3
"""
Enhanced prebuild script for knowledge graphs and vector stores.
Creates indexes per project folder based on FAQ and KB data with improved embedding models,
document chunking, and HNSW indexing for better retrieval performance.
"""

import os
import json
import hashlib
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    from whoosh.index import create_in
    from whoosh.fields import Schema, TEXT, ID, STORED
    from whoosh.writing import AsyncWriter
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("⚠️  Heavy dependencies not available. Creating basic indexes only.")

from api.models import FAQEntry, KBEntry


class DocumentChunker:
    """Handles document chunking with overlap for better retrieval granularity"""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def estimate_tokens(self, text: str) -> int:
        """Simple token estimation (roughly 4 characters per token)"""
        return len(text) // 4
    
    def chunk_text(self, text: str, chunk_id_prefix: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_id_prefix: Prefix for chunk IDs
            
        Returns:
            List of chunks with metadata
        """
        # Estimate tokens and see if chunking is needed
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= self.chunk_size:
            # Text is small enough, return as single chunk
            return [{
                'chunk_id': f"{chunk_id_prefix}_chunk_0",
                'text': text,
                'chunk_index': 0,
                'total_chunks': 1,
                'start_pos': 0,
                'end_pos': len(text)
            }]
        
        # Split text into sentences for better chunk boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'chunk_id': f"{chunk_id_prefix}_chunk_{chunk_index}",
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'total_chunks': -1,  # Will be updated later
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(current_chunk)
                })
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    # Keep last part of current chunk for overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self.estimate_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                
                chunk_index += 1
                start_pos = chunks[-1]['end_pos'] - len(overlap_text) if self.overlap > 0 else chunks[-1]['end_pos']
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"{chunk_id_prefix}_chunk_{chunk_index}",
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'total_chunks': -1,  # Will be updated later
                'start_pos': start_pos,
                'end_pos': len(text)
            })
        
        # Update total_chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last part of text for overlap"""
        overlap_chars = overlap_tokens * 4  # Rough estimation
        if len(text) <= overlap_chars:
            return text
        
        # Try to find a good break point (sentence boundary)
        overlap_text = text[-overlap_chars:]
        sentence_break = overlap_text.find('. ')
        if sentence_break > 0:
            return overlap_text[sentence_break + 2:]
        
        return overlap_text


class IndexBuilder:
    """Builds enhanced dense and sparse indexes for a project"""
    
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
        
        # Initialize chunker
        self.chunker = DocumentChunker(chunk_size=400, overlap=50)
        
        # Initialize models if available
        self.embedding_model = None
        if HAS_DEPS:
            try:
                # Use enhanced BGE model instead of MiniLM
                print(f"  Loading BGE embedding model...")
                self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
                print(f"  ✅ BGE model loaded successfully")
            except Exception as e:
                print(f"  Warning: Could not load BGE model, falling back to MiniLM: {e}")
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    print(f"  ✅ MiniLM model loaded as fallback")
                except Exception as e2:
                    print(f"  Warning: Could not load any embedding model: {e2}")
    
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
    
    def _prepare_chunks_for_indexing(self, faqs: List[FAQEntry], kb_entries: List[KBEntry]) -> Tuple[List[str], List[Dict]]:
        """Prepare text chunks and metadata for indexing"""
        texts = []
        metadata = []
        
        # Process FAQ content
        for faq in faqs:
            text = f"Q: {faq.question}\nA: {faq.answer}"
            
            # Chunk the FAQ content
            chunks = self.chunker.chunk_text(text, f"faq_{faq.id}")
            
            for chunk in chunks:
                texts.append(chunk['text'])
                metadata.append({
                    'id': chunk['chunk_id'],
                    'original_id': faq.id,
                    'type': 'faq',
                    'question': faq.question,
                    'answer': faq.answer,
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos']
                })
        
        # Process KB content
        for kb in kb_entries:
            text = f"Title: {kb.article}\nContent: {kb.content}"
            
            # Chunk the KB content
            chunks = self.chunker.chunk_text(text, f"kb_{kb.id}")
            
            for chunk in chunks:
                texts.append(chunk['text'])
                metadata.append({
                    'id': chunk['chunk_id'],
                    'original_id': kb.id,
                    'type': 'kb',
                    'article': kb.article,
                    'content': chunk['text'],  # Use chunk text instead of full content
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos']
                })
        
        return texts, metadata
    
    def build_dense_index(self, faqs: List[FAQEntry], kb_entries: List[KBEntry]) -> Optional[str]:
        """Build enhanced FAISS HNSW vector index with chunking"""
        if not HAS_DEPS or not self.embedding_model:
            print(f"  Skipping dense index for {self.project_id} (dependencies not available)")
            return None
            
        try:
            # Prepare chunked content
            texts, metadata = self._prepare_chunks_for_indexing(faqs, kb_entries)
            
            if not texts:
                print(f"  No content to index for {self.project_id}")
                return None
                
            # Generate embeddings
            print(f"  Generating embeddings for {len(texts)} text chunks...")
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize embeddings consistently (L2 normalization)
            print(f"  Normalizing embeddings...")
            faiss.normalize_L2(embeddings)
            
            # Create enhanced FAISS index (HNSW for better scalability)
            dimension = embeddings.shape[1]
            
            # Use HNSW index for better performance
            # M=16: number of bi-directional links for each node (trade-off between speed and accuracy)
            # efConstruction=200: size of the dynamic candidate list (higher = better quality, slower build)
            print(f"  Creating HNSW index with dimension {dimension}...")
            index = faiss.IndexHNSWFlat(dimension, 16)
            index.hnsw.efConstruction = 200
            
            # Add embeddings to index
            print(f"  Adding {len(embeddings)} vectors to HNSW index...")
            index.add(embeddings.astype('float32'))
            
            # Set search time parameters
            index.hnsw.efSearch = 64  # Higher = more accurate search, slower
            
            # Save index and metadata
            index_file = self.dense_dir / "faiss_hnsw.index"
            metadata_file = self.dense_dir / "metadata.json"
            
            faiss.write_index(index, str(index_file))
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Enhanced HNSW index built: {len(texts)} chunks, dimension {dimension}")
            return str(index_file)
            
        except Exception as e:
            print(f"  ❌ Error building enhanced dense index: {e}")
            return None
    
    def build_sparse_index(self, faqs: List[FAQEntry], kb_entries: List[KBEntry]) -> Optional[str]:
        """Build enhanced Whoosh sparse text index with chunking"""
        if not HAS_DEPS:
            print(f"  Skipping sparse index for {self.project_id} (dependencies not available)")
            return None
            
        try:
            # Define enhanced schema
            schema = Schema(
                id=ID(stored=True),
                original_id=ID(stored=True),
                type=STORED(),
                content=TEXT(stored=True),
                title=TEXT(stored=True),
                question=TEXT(stored=True),
                answer=TEXT(stored=True),
                chunk_index=STORED(),
                total_chunks=STORED()
            )
            
            # Create index
            index = create_in(str(self.sparse_dir), schema)
            
            # Prepare chunked content
            texts, metadata = self._prepare_chunks_for_indexing(faqs, kb_entries)
            
            # Add documents with chunks
            writer = index.writer()
            
            for text, meta in zip(texts, metadata):
                if meta['type'] == 'faq':
                    writer.add_document(
                        id=meta['id'],
                        original_id=meta['original_id'],
                        type="faq",
                        content=text,
                        title="",
                        question=meta['question'],
                        answer=meta['answer'],
                        chunk_index=meta['chunk_index'],
                        total_chunks=meta['total_chunks']
                    )
                else:  # kb
                    writer.add_document(
                        id=meta['id'],
                        original_id=meta['original_id'],
                        type="kb", 
                        content=text,
                        title=meta['article'],
                        question="",
                        answer="",
                        chunk_index=meta['chunk_index'],
                        total_chunks=meta['total_chunks']
                    )
            
            writer.commit()
            
            total_chunks = len(texts)
            original_docs = len(faqs) + len(kb_entries)
            print(f"  ✅ Enhanced sparse index built: {total_chunks} chunks from {original_docs} documents")
            return str(self.sparse_dir)
            
        except Exception as e:
            print(f"  ❌ Error building enhanced sparse index: {e}")
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
            "created_at": datetime.now(datetime.timezone.utc).isoformat(),
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
                "schema": "2.0",  # Updated schema version
                "embedding_model": self.embedding_model.model_name if (HAS_DEPS and self.embedding_model) else None,
                "chunking": {
                    "enabled": True,
                    "chunk_size": self.chunker.chunk_size,
                    "overlap": self.chunker.overlap
                },
                "index_type": "hnsw" if dense_path and "hnsw" in str(dense_path) else "flat"
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
        
        # Update metadata
        metadata = self.create_metadata(faqs, kb_entries, dense_path, sparse_path)
        
        print(f"  ✅ Enhanced index building complete for {self.project_id}")
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
    print("🚀 DARKBO Enhanced Knowledge Base Prebuild")
    print("=" * 50)
    
    # Load project mapping
    projects = load_project_mapping()
    if not projects:
        print("❌ No projects found in proj_mapping.txt")
        return
    
    print(f"📋 Found {len(projects)} projects to process")
    print(f"💡 Enhancements: BGE embeddings, HNSW indexing, document chunking, consistent normalization")
    
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
    print("📊 Enhanced Build Summary")
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
                index_type = result.get('versions', {}).get('index_type', 'flat')
                status.append(f"dense-{index_type}")
            if sparse_ok:
                status.append("sparse-chunked")
            status_str = "+".join(status) if status else "metadata-only"
            
            # Show chunking info
            chunking_info = result.get('versions', {}).get('chunking', {})
            if chunking_info.get('enabled'):
                chunk_size = chunking_info.get('chunk_size', 400)
                overlap = chunking_info.get('overlap', 50)
                status_str += f" (chunks: {chunk_size}±{overlap})"
            
            embedding_model = result.get('versions', {}).get('embedding_model', 'unknown')
            if embedding_model and 'bge' in embedding_model.lower():
                status_str += " [BGE]"
            
            print(f"✅ {project_id} ({project_name}): {total} items, {status_str}")
            successful += 1
    
    print(f"\n🎉 Completed: {successful} successful, {failed} failed")
    
    if not HAS_DEPS:
        print("\n💡 To enable full enhanced indexing, install dependencies:")
        print("   pip install sentence-transformers faiss-cpu whoosh transformers")
    else:
        print(f"\n🚀 Enhanced features enabled:")
        print(f"   ✓ BGE embedding model for better semantic accuracy")
        print(f"   ✓ HNSW indexing for scalable similarity search")
        print(f"   ✓ Document chunking (400 tokens ±50 overlap)")
        print(f"   ✓ Consistent L2 normalization")


if __name__ == "__main__":
    main()