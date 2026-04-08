"""
Tests for contract parser module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.contract_parser import ContractParser, ContractSection, ParsedContract
from src.config import Config


class TestContractSection:
    """Test ContractSection class"""

    def test_section_creation(self):
        """Test creating a contract section"""
        section = ContractSection(
            text="هذا نص تجريبي للبند",
            section_number="1",
            title="البند الأول",
            level=1
        )

        assert section.text == "هذا نص تجريبي للبند"
        assert section.section_number == "1"
        assert section.title == "البند الأول"
        assert section.level == 1

    def test_section_to_dict(self):
        """Test converting section to dictionary"""
        section = ContractSection(
            text="نص تجريبي",
            section_number="1"
        )

        result = section.to_dict()

        assert "text" in result
        assert "section_number" in result
        assert "word_count" in result


class TestParsedContract:
    """Test ParsedContract class"""

    def test_contract_creation(self):
        """Test creating a parsed contract"""
        contract = ParsedContract("test.pdf", "نص العقد الكامل")

        assert contract.file_path == "test.pdf"
        assert contract.raw_text == "نص العقد الكامل"
        assert len(contract.sections) == 0

    def test_add_section(self):
        """Test adding sections to contract"""
        contract = ParsedContract("test.pdf", "نص العقد")

        section1 = ContractSection("البند الأول", "1")
        section2 = ContractSection("البند الثاني", "2")

        contract.add_section(section1)
        contract.add_section(section2)

        assert len(contract.sections) == 2
        assert contract.sections[0].section_number == "1"

    def test_get_sections_text(self):
        """Test getting list of section texts"""
        contract = ParsedContract("test.pdf", "نص")

        contract.add_section(ContractSection("نص 1", "1"))
        contract.add_section(ContractSection("نص 2", "2"))

        texts = contract.get_sections_text()

        assert len(texts) == 2
        assert "نص 1" in texts
        assert "نص 2" in texts


class TestContractParser:
    """Test ContractParser class"""

    def test_parser_creation(self):
        """Test creating parser"""
        parser = ContractParser()

        assert parser.config is not None

    def test_parse_txt_file(self):
        """Test parsing TXT file"""
        parser = ContractParser()

        # Parse sample contract
        contract_path = "data/sample_contracts/sample_contract.txt"

        if Path(contract_path).exists():
            contract = parser.parse_file(contract_path)

            assert contract is not None
            assert len(contract.raw_text) > 0
            assert len(contract.sections) > 0

    def test_section_detection(self):
        """Test section detection in contract"""
        parser = ContractParser()

        # Create contract with numbered sections
        text = """1. البند الأول
        هذا هو نص البند الأول

        2. البند الثاني
        هذا هو نص البند الثاني"""

        contract = ParsedContract("test.txt", text)
        parser._detect_sections(contract)

        assert len(contract.sections) >= 1

    def test_unsupported_format(self):
        """Test handling unsupported file format"""
        parser = ContractParser()

        result = parser.parse_file("test.xyz")

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
