"""
Tests for RAG engine module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_engine import RAGEngine, RetrievedContext
from src.contract_parser import ContractSection, ParsedContract


class TestRetrievedContext:
    """Test RetrievedContext class"""

    def test_context_creation(self):
        """Test creating retrieved context"""
        chunks = [
            {
                "text": "نص القانون",
                "metadata": {"article": "1", "law": "قانون العمل"},
                "score": 0.95
            }
        ]

        context = RetrievedContext(
            query_text="استعلام تجريبي",
            retrieved_chunks=chunks,
            retrieval_stage="test"
        )

        assert context.query_text == "استعلام تجريبي"
        assert len(context.retrieved_chunks) == 1
        assert context.retrieval_stage == "test"

    def test_get_context_text(self):
        """Test getting formatted context text"""
        chunks = [
            {
                "text": "نص المادة الأولى",
                "metadata": {"article": "1", "law": "قانون العمل"},
                "score": 0.95
            }
        ]

        context = RetrievedContext("query", chunks)
        text = context.get_context_text()

        assert "المادة 1" in text
        assert "نص المادة الأولى" in text

    def test_get_top_articles(self):
        """Test getting top article numbers"""
        chunks = [
            {
                "text": "نص 1",
                "metadata": {"article": "1", "law": "قانون"},
                "score": 0.95
            },
            {
                "text": "نص 2",
                "metadata": {"article": "2", "law": "قانون"},
                "score": 0.90
            }
        ]

        context = RetrievedContext("query", chunks)
        articles = context.get_top_articles(2)

        assert "1" in articles
        assert "2" in articles


class TestRAGEngine:
    """Test RAG engine functionality"""

    def test_engine_creation(self):
        """Test creating RAG engine"""
        engine = RAGEngine()

        assert engine.config is not None
        assert engine.vector_store is not None

    def test_retrieve_for_section(self):
        """Test retrieving for contract section"""
        # This test requires ChromaDB to be initialized
        # Skip if not available
        pytest.skip("Requires initialized ChromaDB")

        engine = RAGEngine()
        section = ContractSection(
            text="يعمل الموظف 60 ساعة في الأسبوع",
            section_number="1"
        )

        context = engine.retrieve_for_section(section)

        assert context is not None
        assert context.retrieval_stage == "section"

    def test_retrieve_for_contract(self):
        """Test retrieving for full contract"""
        pytest.skip("Requires initialized ChromaDB")

        engine = RAGEngine()
        contract = ParsedContract("test.txt", "نص العقد")
        contract.add_section(ContractSection("البند الأول", "1"))

        results = engine.retrieve_for_contract(contract)

        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
