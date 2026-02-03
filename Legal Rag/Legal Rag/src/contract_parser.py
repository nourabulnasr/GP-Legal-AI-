"""
Contract Parser module for multi-format contract parsing (PDF, DOCX, TXT)
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
from docx import Document
import docx2txt

from .config import get_config


logger = logging.getLogger(__name__)


class ContractSection:
    """Represents a section or clause in a contract"""

    def __init__(
        self,
        text: str,
        section_number: Optional[str] = None,
        title: Optional[str] = None,
        level: int = 0
    ):
        """
        Initialize a contract section

        Args:
            text: Section text content
            section_number: Section number (e.g., "1.1", "2.3.1")
            title: Section title
            level: Hierarchical level (0=root, 1=section, 2=subsection, etc.)
        """
        self.text = text
        self.section_number = section_number
        self.title = title
        self.level = level

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "section_number": self.section_number,
            "title": self.title,
            "level": self.level,
            "word_count": len(self.text.split())
        }


class ParsedContract:
    """Represents a parsed contract with structured content"""

    def __init__(self, file_path: str, raw_text: str):
        """
        Initialize parsed contract

        Args:
            file_path: Path to original contract file
            raw_text: Raw contract text
        """
        self.file_path = file_path
        self.raw_text = raw_text
        self.sections: List[ContractSection] = []
        self.metadata: Dict[str, Any] = {}

    def add_section(self, section: ContractSection):
        """Add a section to the contract"""
        self.sections.append(section)

    def get_full_text(self) -> str:
        """Get the full contract text"""
        return self.raw_text

    def get_sections_text(self) -> List[str]:
        """Get list of section texts"""
        return [section.text for section in self.sections]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "file_path": self.file_path,
            "raw_text": self.raw_text,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
            "total_sections": len(self.sections),
            "word_count": len(self.raw_text.split())
        }


class ContractParser:
    """Multi-format contract parser supporting PDF, DOCX, and TXT"""

    def __init__(self, config=None):
        """
        Initialize contract parser

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()

    def parse_file(self, file_path: str) -> Optional[ParsedContract]:
        """
        Parse contract file (auto-detects format)

        Args:
            file_path: Path to contract file

        Returns:
            ParsedContract object or None if parsing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.parsing.max_file_size_mb:
            logger.error(
                f"File too large: {file_size_mb:.2f}MB "
                f"(max: {self.config.parsing.max_file_size_mb}MB)"
            )
            return None

        # Detect format and parse
        suffix = file_path.suffix.lower()

        logger.info(f"Parsing {suffix} file: {file_path}")

        if suffix == '.pdf':
            return self.parse_pdf(str(file_path))
        elif suffix in ['.docx', '.doc']:
            return self.parse_docx(str(file_path))
        elif suffix == '.txt':
            return self.parse_txt(str(file_path))
        else:
            logger.error(f"Unsupported file format: {suffix}")
            return None

    def parse_pdf(self, file_path: str) -> Optional[ParsedContract]:
        """
        Parse PDF contract

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedContract object or None
        """
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

            if not text.strip():
                logger.error("No text extracted from PDF")
                return None

            # Create parsed contract
            contract = ParsedContract(file_path, text.strip())

            # Detect and parse sections
            if self.config.parsing.detect_sections:
                self._detect_sections(contract)

            return contract

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            return None

    def parse_docx(self, file_path: str) -> Optional[ParsedContract]:
        """
        Parse DOCX contract

        Args:
            file_path: Path to DOCX file

        Returns:
            ParsedContract object or None
        """
        try:
            # Extract text using docx2txt (handles Arabic better)
            text = docx2txt.process(file_path)

            if not text.strip():
                # Fallback to python-docx
                logger.warning("docx2txt returned empty, trying python-docx...")
                doc = Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs]
                text = "\n".join(paragraphs)

            if not text.strip():
                logger.error("No text extracted from DOCX")
                return None

            # Create parsed contract
            contract = ParsedContract(file_path, text.strip())

            # Detect and parse sections
            if self.config.parsing.detect_sections:
                self._detect_sections(contract)

            return contract

        except Exception as e:
            logger.error(f"Failed to parse DOCX: {e}")
            return None

    def parse_txt(self, file_path: str) -> Optional[ParsedContract]:
        """
        Parse plain text contract

        Args:
            file_path: Path to TXT file

        Returns:
            ParsedContract object or None
        """
        try:
            # Try UTF-8 first, then fallback to other encodings
            encodings = ['utf-8', 'utf-16', 'cp1256', 'iso-8859-1']

            text = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    logger.info(f"Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                logger.error("Failed to decode text file with any encoding")
                return None

            if not text.strip():
                logger.error("Text file is empty")
                return None

            # Create parsed contract
            contract = ParsedContract(file_path, text.strip())

            # Detect and parse sections
            if self.config.parsing.detect_sections:
                self._detect_sections(contract)

            return contract

        except Exception as e:
            logger.error(f"Failed to parse TXT: {e}")
            return None

    def _detect_sections(self, contract: ParsedContract):
        """
        Detect and extract sections from contract text

        Args:
            contract: ParsedContract object to populate with sections
        """
        text = contract.raw_text

        # Pattern for Arabic and English numbered sections
        # Matches: "1.", "1-", "المادة 1", "البند 1", "Article 1", etc.
        section_patterns = [
            r'(?:المادة|البند|الفقرة|Article|Section|Clause)\s+(\d+)',
            r'^(\d+)[\.\-\)]\s+',
            r'\n(\d+)[\.\-\)]\s+',
        ]

        # Try to split by section markers
        sections_found = False
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))

            if len(matches) >= 2:  # Found at least 2 sections
                sections_found = True
                logger.info(f"Found {len(matches)} sections using pattern: {pattern}")

                for i, match in enumerate(matches):
                    section_num = match.group(1) if match.groups() else str(i + 1)
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

                    section_text = text[start:end].strip()

                    # Skip very short sections
                    if len(section_text) < self.config.parsing.min_clause_length:
                        continue

                    contract.add_section(ContractSection(
                        text=section_text,
                        section_number=section_num,
                        level=1
                    ))

                break

        # If no sections found, split by paragraphs
        if not sections_found or len(contract.sections) == 0:
            logger.info("No section markers found, splitting by paragraphs")
            paragraphs = text.split('\n\n')

            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) >= self.config.parsing.min_clause_length:
                    contract.add_section(ContractSection(
                        text=para,
                        section_number=str(i + 1),
                        level=0
                    ))

        logger.info(f"Extracted {len(contract.sections)} sections from contract")


if __name__ == "__main__":
    # Test parser
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python contract_parser.py <contract_file>")
        sys.exit(1)

    parser = ContractParser()
    contract = parser.parse_file(sys.argv[1])

    if contract:
        print("\nContract parsed successfully!")
        print(f"File: {contract.file_path}")
        print(f"Total words: {len(contract.raw_text.split())}")
        print(f"Sections: {len(contract.sections)}")

        print("\nFirst 3 sections:")
        for i, section in enumerate(contract.sections[:3], 1):
            print(f"\n{i}. Section {section.section_number}")
            print(f"   {section.text[:200]}...")
    else:
        print("Failed to parse contract")
