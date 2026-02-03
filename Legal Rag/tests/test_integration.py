"""
Integration tests for end-to-end functionality
"""

import pytest
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEndToEnd:
    """End-to-end integration tests"""

    @pytest.mark.skip(reason="Requires full environment setup")
    def test_full_analysis_workflow(self):
        """Test complete contract analysis workflow"""
        from src.contract_analyzer import ContractAnalyzer
        from src.report_generator import ReportGenerator

        # Initialize components
        analyzer = ContractAnalyzer()
        report_gen = ReportGenerator()

        # Analyze sample contract
        contract_path = "data/sample_contracts/sample_contract.txt"
        result = analyzer.analyze_contract(contract_path)

        # Verify result
        assert result is not None
        assert result.contract_id is not None
        assert result.overall_status in ["compliant", "issues_found", "major_violations"]

        # Generate reports
        json_report = report_gen.generate_json(result)
        assert json_report is not None

        html_report = report_gen.generate_html(result)
        assert html_report is not None
        assert len(html_report) > 0

    @pytest.mark.skip(reason="Requires full environment setup")
    def test_data_ingestion_to_retrieval(self):
        """Test data ingestion and retrieval pipeline"""
        from src.data_ingestion import DataIngestion
        from src.rag_engine import RAGEngine

        # Load data
        ingestion = DataIngestion()
        success = ingestion.ingest_labor_law_data()
        assert success == True

        # Test retrieval
        engine = RAGEngine()
        context = engine.retrieve_for_text("ساعات العمل", top_k=3)

        assert context is not None
        assert len(context.retrieved_chunks) > 0

    def test_sample_contract_structure(self):
        """Test that sample contract exists and has violations"""
        contract_path = Path("data/sample_contracts/sample_contract.txt")

        assert contract_path.exists(), "Sample contract file missing"

        content = contract_path.read_text(encoding='utf-8')

        # Check for intentional violations
        assert "60 ساعة" in content  # Excessive hours
        assert "10 أيام" in content  # Insufficient leave

    def test_labor_law_data_exists(self):
        """Test that labor law data file exists"""
        data_path = Path("data/labor14_2025_chunks.cleaned.jsonl")

        assert data_path.exists(), "Labor law data file missing"

        # Verify it's valid JSONL
        with open(data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            data = json.loads(first_line)

            assert "id" in data
            assert "text" in data
            assert "article" in data


class TestModuleImports:
    """Test that all modules can be imported"""

    def test_import_config(self):
        """Test importing config module"""
        from src import config
        assert hasattr(config, 'Config')
        assert hasattr(config, 'get_config')

    def test_import_vector_store(self):
        """Test importing vector store module"""
        from src import vector_store
        assert hasattr(vector_store, 'VectorStore')

    def test_import_data_ingestion(self):
        """Test importing data ingestion module"""
        from src import data_ingestion
        assert hasattr(data_ingestion, 'DataIngestion')

    def test_import_contract_parser(self):
        """Test importing contract parser module"""
        from src import contract_parser
        assert hasattr(contract_parser, 'ContractParser')

    def test_import_rag_engine(self):
        """Test importing RAG engine module"""
        from src import rag_engine
        assert hasattr(rag_engine, 'RAGEngine')

    def test_import_llm_client(self):
        """Test importing LLM client module"""
        from src import llm_client
        assert hasattr(llm_client, 'LLMClient')

    def test_import_contract_analyzer(self):
        """Test importing contract analyzer module"""
        from src import contract_analyzer
        assert hasattr(contract_analyzer, 'ContractAnalyzer')

    def test_import_report_generator(self):
        """Test importing report generator module"""
        from src import report_generator
        assert hasattr(report_generator, 'ReportGenerator')

    def test_import_main(self):
        """Test importing main module"""
        from src import main
        assert hasattr(main, 'main')

    def test_import_api(self):
        """Test importing API module"""
        from src import api
        assert hasattr(api, 'create_app')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
