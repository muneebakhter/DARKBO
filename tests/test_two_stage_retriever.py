import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.schema import Document
from rag.retriever import TwoStageRetriever


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k=4, filter=None):
        def match(doc, cond):
            if not cond:
                return True
            if "$and" in cond:
                return all(match(doc, c) for c in cond["$and"])
            if "$or" in cond:
                return any(match(doc, c) for c in cond["$or"])
            for key, op in cond.items():
                val = doc.metadata.get(key)
                if "$eq" in op:
                    if val != op["$eq"]:
                        return False
                if "$in" in op:
                    if val not in op["$in"]:
                        return False
            return True

        results = [doc for doc in self.docs if match(doc, filter)]
        return results[:k]


class DummySettings:
    MAX_ARTICLES = 5
    MAX_SECTIONS_PER_ARTICLE = 5


def build_docs():
    docs = [
        Document(
            page_content="A1 chunk0",
            metadata={"project_id": "1", "source": "kb", "article_id": "a1", "chunk_id": 0}
        ),
        Document(
            page_content="A1 chunk1",
            metadata={"project_id": "1", "source": "kb", "article_id": "a1", "chunk_id": 1}
        ),
        Document(
            page_content="FAQ answer",
            metadata={"project_id": "1", "source": "faq", "question": "What?"}
        ),
    ]
    return docs


def test_article_level_includes_faq():
    docs = build_docs()
    store = FakeVectorStore(docs)
    retriever = TwoStageRetriever(store, "1", DummySettings())
    results = retriever._get_article_level_results("query")
    assert any(d.metadata.get("source") == "faq" for d in results)
    assert any(d.metadata.get("chunk_id") == 0 for d in results)


def test_get_relevant_documents_with_faq():
    docs = build_docs()
    store = FakeVectorStore(docs)
    retriever = TwoStageRetriever(store, "1", DummySettings())
    results = retriever.get_relevant_documents("query")
    # Expect FAQ doc present
    faq_results = [d for d in results if d.metadata.get("source") == "faq"]
    assert len(faq_results) == 1
    # Should also include section-level chunks
    kb_results = [d for d in results if d.metadata.get("source") == "kb"]
    assert any(d.metadata.get("chunk_id") == 1 for d in kb_results)
