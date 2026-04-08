"""
Tests for data ingestion module
"""

import pytest
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestion


class TestDataIngestion:
    """Test data ingestion functionality"""

    def test_ingestion_creation(self):
        """Test creating DataIngestion instance"""
        ingestion = DataIngestion()

        assert ingestion.config is not None
        assert ingestion.vector_store is not None

    def test_load_jsonl(self):
        """Test loading JSONL file"""
        ingestion = DataIngestion()

        # Load sample data
        file_path = "data/labor14_2025_chunks.cleaned.jsonl"

        if Path(file_path).exists():
            documents = ingestion.load_jsonl(file_path)

            assert len(documents) > 0
            assert "id" in documents[0]
            assert "law" in documents[0]
            assert "article" in documents[0]
            assert "text" in documents[0]

    def test_validate_document(self):
        """Test document validation"""
        ingestion = DataIngestion()

        # Valid document
        valid_doc = {
            "id": "test_1",
            "law": "قانون العمل",
            "article": "1",
            "chunk_index": 1,
            "text": "نص القانون",
            "source": "test"
        }

        assert ingestion.validate_document(valid_doc) == True

        # Invalid document (missing field)
        invalid_doc = {
            "id": "test_2",
            "law": "قانون العمل"
            # Missing required fields
        }

        assert ingestion.validate_document(invalid_doc) == False

    def test_prepare_documents(self):
        """Test preparing documents for ingestion"""
        ingestion = DataIngestion()

        documents = [
            {
                "id": "test_1",
                "law": "قانون العمل رقم 14",
                "article": "1",
                "chunk_index": 1,
                "text": "نص المادة الأولى",
                "normalized_text": "نص المادة الأولى",
                "source": "labor14"
            }
        ]

        texts, metadatas, ids = ingestion.prepare_documents_for_ingestion(documents)

        assert len(texts) == 1
        assert len(metadatas) == 1
        assert len(ids) == 1
        assert texts[0] == "نص المادة الأولى"
        assert metadatas[0]["article"] == "1"
        assert ids[0] == "test_1"


class TestJSONLFormat:
    """Test JSONL file format"""

    def test_jsonl_structure(self):
        """Test that JSONL file has correct structure"""
        file_path = "data/labor14_2025_chunks.cleaned.jsonl"

        if not Path(file_path).exists():
            pytest.skip("JSONL file not found")

        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            doc = json.loads(first_line)

            # Check required fields
            required_fields = ['id', 'law', 'article', 'chunk_index', 'text', 'source']
            for field in required_fields:
                assert field in doc, f"Missing required field: {field}"

            # Check data types
            assert isinstance(doc['article'], str)
            assert isinstance(doc['chunk_index'], int)
            assert isinstance(doc['text'], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
