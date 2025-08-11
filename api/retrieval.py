import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

# Vector search imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from whoosh.index import create_index, open_dir
    from whoosh.fields import Schema, TEXT, ID, STORED
    from whoosh.qparser import QueryParser
    from whoosh.writing import AsyncWriter
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from api.models import FAQEntry, KBEntry, Citation, QueryResponse
from api.storage import FileStorageManager


@dataclass
class SearchResult:
    """Search result with score and metadata"""
    entry_id: str
    entry_type: str  # 'faq' or 'kb'
    content: str
    score: float
    metadata: Dict
    article: Optional[str] = None


class EmbeddingManager:
    """Manages dense vector embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(self.model_name)
        else:
            raise RuntimeError("sentence-transformers not available")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        return self.model.encode(texts, convert_to_tensor=False)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.encode([text])[0]


class SparseIndex:
    """Manages sparse BM25 search using Whoosh"""
    
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.schema = Schema(
            id=ID(stored=True),
            type=TEXT(stored=True),
            content=TEXT(stored=True),
            article=TEXT(stored=True),
            metadata=STORED()
        )
        self.index = None
        self._init_index()
    
    def _init_index(self):
        """Initialize or open Whoosh index"""
        if not WHOOSH_AVAILABLE:
            return
        
        try:
            self.index = open_dir(str(self.index_dir))
        except:
            self.index = create_index(str(self.index_dir), self.schema)
    
    def index_entries(self, entries: List[Tuple[str, str, str, str, Dict]]):
        """Index entries: (id, type, content, article, metadata)"""
        if not self.index:
            return
        
        writer = self.index.writer()
        try:
            for entry_id, entry_type, content, article, metadata in entries:
                writer.add_document(
                    id=entry_id,
                    type=entry_type,
                    content=content,
                    article=article or "",
                    metadata=json.dumps(metadata)
                )
            writer.commit()
        except Exception as e:
            writer.cancel()
            raise e
    
    def search(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Search using BM25"""
        if not self.index:
            return []
        
        with self.index.searcher() as searcher:
            parser = QueryParser("content", self.index.schema)
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=limit)
            
            search_results = []
            for hit in results:
                try:
                    metadata = json.loads(hit['metadata'])
                except:
                    metadata = {}
                
                result = SearchResult(
                    entry_id=hit['id'],
                    entry_type=hit['type'],
                    content=hit['content'],
                    score=hit.score,
                    metadata=metadata,
                    article=hit.get('article')
                )
                search_results.append(result)
            
            return search_results


class DenseIndex:
    """Manages dense vector search using FAISS"""
    
    def __init__(self, index_dir: Path, embedding_manager: EmbeddingManager):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_manager = embedding_manager
        self.index = None
        self.id_map = {}  # Maps FAISS index -> entry_id
        self.metadata_map = {}  # Maps entry_id -> metadata
        self._load_index()
    
    def _load_index(self):
        """Load or create FAISS index"""
        if not FAISS_AVAILABLE:
            return
        
        index_file = self.index_dir / "faiss.index"
        metadata_file = self.index_dir / "metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.id_map = data.get('id_map', {})
                    self.metadata_map = data.get('metadata_map', {})
            except Exception:
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        if not FAISS_AVAILABLE:
            return
        
        # Create flat index for now (can upgrade to IVF later)
        dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim)
        self.id_map = {}
        self.metadata_map = {}
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        if not self.index:
            return
        
        index_file = self.index_dir / "faiss.index"
        metadata_file = self.index_dir / "metadata.json"
        
        faiss.write_index(self.index, str(index_file))
        
        with open(metadata_file, 'w') as f:
            json.dump({
                'id_map': self.id_map,
                'metadata_map': self.metadata_map
            }, f, indent=2)
    
    def index_entries(self, entries: List[Tuple[str, str, str, str, Dict]]):
        """Index entries: (id, type, content, article, metadata)"""
        if not self.index:
            return
        
        texts = [content for _, _, content, _, _ in entries]
        embeddings = self.embedding_manager.encode(texts)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Update mappings
        for i, (entry_id, entry_type, content, article, metadata) in enumerate(entries):
            faiss_idx = start_idx + i
            self.id_map[str(faiss_idx)] = entry_id
            self.metadata_map[entry_id] = {
                'type': entry_type,
                'content': content,
                'article': article,
                **metadata
            }
        
        self._save_index()
    
    def search(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Search using dense vectors"""
        if not self.index or self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_manager.encode_single(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(limit, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            entry_id = self.id_map.get(str(idx))
            if not entry_id:
                continue
            
            metadata = self.metadata_map.get(entry_id, {})
            
            result = SearchResult(
                entry_id=entry_id,
                entry_type=metadata.get('type', 'unknown'),
                content=metadata.get('content', ''),
                score=float(score),
                metadata=metadata,
                article=metadata.get('article')
            )
            results.append(result)
        
        return results


class HybridRetriever:
    """Hybrid retrieval system combining sparse and dense search"""
    
    def __init__(
        self,
        project_id: str,
        storage_manager: FileStorageManager,
        embedding_manager: EmbeddingManager
    ):
        self.project_id = project_id
        self.storage = storage_manager
        self.embedding_manager = embedding_manager
        
        # Initialize indexes
        project_dir = storage_manager.base_dir / project_id
        self.sparse_index = SparseIndex(project_dir / "index" / "sparse")
        self.dense_index = DenseIndex(project_dir / "index" / "dense", embedding_manager)
        
        self.faq_threshold = 0.85  # Threshold for FAQ-first routing
    
    def rebuild_indexes(self):
        """Rebuild all indexes from current data"""
        # Load current data
        faqs = self.storage.load_faqs(self.project_id)
        kb_entries = self.storage.load_kb_entries(self.project_id)
        
        # Prepare entries for indexing
        entries = []
        
        # Add FAQ entries
        for faq in faqs:
            entries.append((
                faq.id,
                "faq",
                f"{faq.question} {faq.answer}",  # Index both question and answer
                None,
                {
                    "question": faq.question,
                    "answer": faq.answer,
                    "source": faq.source,
                    "source_file": faq.source_file
                }
            ))
        
        # Add KB entries
        for kb_entry in kb_entries:
            entries.append((
                kb_entry.id,
                "kb",
                kb_entry.content,
                kb_entry.article,
                {
                    "article": kb_entry.article,
                    "source": kb_entry.source,
                    "source_file": kb_entry.source_file,
                    "chunk_index": kb_entry.chunk_index
                }
            ))
        
        # Index in both sparse and dense
        if entries:
            self.sparse_index.index_entries(entries)
            self.dense_index.index_entries(entries)
    
    def _combine_results(
        self,
        sparse_results: List[SearchResult],
        dense_results: List[SearchResult],
        alpha: float = 0.6
    ) -> List[SearchResult]:
        """Combine sparse and dense results with weighted scoring"""
        result_map = {}
        
        # Add sparse results
        for result in sparse_results:
            result_map[result.entry_id] = SearchResult(
                entry_id=result.entry_id,
                entry_type=result.entry_type,
                content=result.content,
                score=alpha * result.score,
                metadata=result.metadata,
                article=result.article
            )
        
        # Add dense results
        for result in dense_results:
            if result.entry_id in result_map:
                # Combine scores
                existing = result_map[result.entry_id]
                existing.score += (1 - alpha) * result.score
            else:
                result_map[result.entry_id] = SearchResult(
                    entry_id=result.entry_id,
                    entry_type=result.entry_type,
                    content=result.content,
                    score=(1 - alpha) * result.score,
                    metadata=result.metadata,
                    article=result.article
                )
        
        # Sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    def search_faqs(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search FAQs specifically"""
        sparse_results = self.sparse_index.search(query, limit * 2)
        dense_results = self.dense_index.search(query, limit * 2)
        
        # Filter for FAQ entries only
        sparse_faqs = [r for r in sparse_results if r.entry_type == "faq"]
        dense_faqs = [r for r in dense_results if r.entry_type == "faq"]
        
        combined = self._combine_results(sparse_faqs, dense_faqs)
        return combined[:limit]
    
    def search_kb(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search knowledge base specifically"""
        sparse_results = self.sparse_index.search(query, limit * 2)
        dense_results = self.dense_index.search(query, limit * 2)
        
        # Filter for KB entries only
        sparse_kb = [r for r in sparse_results if r.entry_type == "kb"]
        dense_kb = [r for r in dense_results if r.entry_type == "kb"]
        
        combined = self._combine_results(sparse_kb, dense_kb)
        return combined[:limit]
    
    def hybrid_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Full hybrid search across all content"""
        sparse_results = self.sparse_index.search(query, limit * 2)
        dense_results = self.dense_index.search(query, limit * 2)
        
        combined = self._combine_results(sparse_results, dense_results)
        return combined[:limit]


class QueryProcessor:
    """Main query processing with FAQ-first routing"""
    
    def __init__(
        self,
        storage_manager: FileStorageManager,
        embedding_manager: EmbeddingManager
    ):
        self.storage = storage_manager
        self.embedding_manager = embedding_manager
        self.retrievers = {}  # Cache retrievers per project
    
    def _get_retriever(self, project_id: str) -> HybridRetriever:
        """Get or create retriever for project"""
        if project_id not in self.retrievers:
            retriever = HybridRetriever(project_id, self.storage, self.embedding_manager)
            # Ensure indexes are built
            retriever.rebuild_indexes()
            self.retrievers[project_id] = retriever
        
        return self.retrievers[project_id]
    
    def process_query(
        self,
        project_id: str,
        question: str,
        mode: str = "auto",
        strict_citations: bool = True
    ) -> QueryResponse:
        """Process query with FAQ-first routing"""
        retriever = self._get_retriever(project_id)
        
        if mode == "faq":
            # FAQ-only mode
            faq_results = retriever.search_faqs(question, limit=5)
            if faq_results and faq_results[0].score >= 0.7:
                # Return top FAQ
                top_result = faq_results[0]
                return self._create_faq_response(top_result)
            else:
                # No good FAQ match
                return QueryResponse(
                    answer="I couldn't find a relevant FAQ for your question.",
                    mode="faq",
                    confidence=0.0,
                    citations=[]
                )
        
        elif mode == "kb":
            # KB-only mode
            kb_results = retriever.search_kb(question, limit=6)
            return self._create_kb_response(question, kb_results, strict_citations)
        
        else:  # mode == "auto"
            # FAQ-first routing
            faq_results = retriever.search_faqs(question, limit=3)
            
            # Check if top FAQ is good enough
            if faq_results and faq_results[0].score >= retriever.faq_threshold:
                return self._create_faq_response(faq_results[0])
            
            # Fall back to KB search
            kb_results = retriever.search_kb(question, limit=6)
            return self._create_kb_response(question, kb_results, strict_citations)
    
    def _create_faq_response(self, faq_result: SearchResult) -> QueryResponse:
        """Create response from FAQ result"""
        citation = Citation(
            type="faq",
            id=faq_result.entry_id,
            article=None,
            lines=None,
            score=faq_result.score
        )
        
        return QueryResponse(
            answer=faq_result.metadata.get("answer", ""),
            mode="faq",
            confidence=faq_result.score,
            citations=[citation],
            used_chunks=[faq_result.entry_id]
        )
    
    def _create_kb_response(
        self,
        question: str,
        kb_results: List[SearchResult],
        strict_citations: bool
    ) -> QueryResponse:
        """Create response from KB results"""
        if not kb_results:
            return QueryResponse(
                answer="I couldn't find relevant information in the knowledge base.",
                mode="kb",
                confidence=0.0,
                citations=[]
            )
        
        # Use the top results to generate answer
        citations = []
        used_chunks = []
        context_pieces = []
        
        for result in kb_results[:3]:  # Use top 3 results
            citations.append(Citation(
                type="kb",
                id=result.entry_id,
                article=result.article,
                lines=None,  # Could add line number detection here
                score=result.score
            ))
            
            used_chunks.append(result.entry_id)
            context_pieces.append(result.content)
        
        # For now, return extractive answer (verbatim from top result)
        # TODO: Add LLM-based answer generation here
        answer = context_pieces[0] if context_pieces else "No relevant information found."
        
        # Simple confidence calculation
        confidence = kb_results[0].score if kb_results else 0.0
        
        return QueryResponse(
            answer=answer,
            mode="kb",
            confidence=confidence,
            citations=citations,
            used_chunks=used_chunks
        )