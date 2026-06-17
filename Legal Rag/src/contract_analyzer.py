"""
Contract Analyzer - Main orchestration module for legal analysis
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid

from .config import get_config
from .contract_parser import ContractParser, ParsedContract
from .rag_engine import RAGEngine
from .llm_client import LLMClient


logger = logging.getLogger(__name__)


class AnalysisResult:
    """Represents the complete analysis result for a contract"""

    def __init__(self, contract_path: str):
        """
        Initialize analysis result

        Args:
            contract_path: Path to the analyzed contract
        """
        self.contract_id = str(uuid.uuid4())
        self.contract_path = contract_path
        self.analysis_date = datetime.now().isoformat()
        self.issues: List[Dict[str, Any]] = []
        self.overall_status = "pending"
        self.summary = ""
        self.metadata: Dict[str, Any] = {}

    def add_issue(self, issue: Dict[str, Any]):
        """Add an identified issue"""
        issue["issue_id"] = len(self.issues) + 1
        self.issues.append(issue)

    def calculate_overall_status(self):
        """Calculate overall compliance status based on issues"""
        if not self.issues:
            self.overall_status = "compliant"
            return

        # Check severity levels
        has_high = any(
            issue.get("severity") in ["عالية", "high"]
            for issue in self.issues
        )

        if has_high:
            self.overall_status = "major_violations"
        else:
            self.overall_status = "issues_found"

    def generate_summary(self):
        """Generate a summary of the analysis"""
        total_issues = len(self.issues)

        if total_issues == 0:
            self.summary = "تم فحص العقد ولم يتم العثور على مخالفات لقانون العمل المصري."
            return

        # Count by severity
        severity_counts = {}
        for issue in self.issues:
            severity = issue.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        summary_parts = [
            f"تم العثور على {total_issues} مشكلة محتملة في العقد:"
        ]

        for severity, count in severity_counts.items():
            summary_parts.append(f"  - {count} مشكلة بدرجة خطورة: {severity}")

        self.summary = "\n".join(summary_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "contract_id": self.contract_id,
            "contract_path": self.contract_path,
            "analysis_date": self.analysis_date,
            "overall_status": self.overall_status,
            "total_issues": len(self.issues),
            "issues": self.issues,
            "summary": self.summary,
            "metadata": self.metadata
        }


class ContractAnalyzer:
    """Main analyzer for contract legal compliance"""

    def __init__(self, config=None):
        """
        Initialize contract analyzer

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.parser = ContractParser(self.config)
        self.rag_engine = RAGEngine(self.config)
        self.llm_client = LLMClient(self.config)

    def analyze_contract(
        self,
        contract_path: str,
        detailed_analysis: bool = True
    ) -> AnalysisResult:
        """
        Analyze a contract for labor law compliance

        Args:
            contract_path: Path to contract file
            detailed_analysis: Perform detailed clause-by-clause analysis

        Returns:
            AnalysisResult object
        """
        logger.info(f"Starting analysis of contract: {contract_path}")

        # Initialize result
        result = AnalysisResult(contract_path)

        # Step 1: Parse contract
        logger.info("Step 1: Parsing contract...")
        contract = self.parser.parse_file(contract_path)

        if not contract:
            logger.error("Failed to parse contract")
            result.metadata["error"] = "Failed to parse contract"
            return result

        result.metadata["total_sections"] = len(contract.sections)
        result.metadata["word_count"] = len(contract.raw_text.split())

        # Step 2: Retrieve relevant laws
        logger.info("Step 2: Retrieving relevant laws...")
        retrieval_results = self.rag_engine.retrieve_for_contract(
            contract,
            use_two_stage=detailed_analysis
        )

        result.metadata["retrieval_results_count"] = len(retrieval_results)

        # Step 3: Analyze each section/clause with LLM
        logger.info("Step 3: Analyzing with LLM...")

        for i, retrieval_result in enumerate(retrieval_results, 1):
            section = retrieval_result.get("section")
            context = retrieval_result["context"]

            # Get clause text
            if section:
                clause_text = section.text
                section_number = section.section_number or str(i)
            else:
                # For full contract analysis
                clause_text = contract.get_full_text()[:2000]  # Truncate for LLM
                section_number = "Full Contract"

            logger.info(f"Analyzing section {section_number} ({i}/{len(retrieval_results)})")

            # Get context text
            laws_text = context.get_context_text(max_chunks=5)

            # Skip if no relevant laws found
            if not context.retrieved_chunks:
                logger.debug(f"No relevant laws found for section {section_number}, skipping")
                continue

            # Analyze with LLM
            try:
                analysis = self.llm_client.analyze_clause(clause_text, laws_text)

                # Check if violation was found
                has_violation = analysis.get("has_violation")

                if has_violation or has_violation is None:  # Include uncertain cases
                    # Prepare violated law info
                    violated_law = None
                    if context.retrieved_chunks:
                        top_chunk = context.retrieved_chunks[0]
                        violated_law = {
                            "article": top_chunk['metadata'].get('article'),
                            "text": top_chunk['text'][:500],  # Truncate
                            "law": top_chunk['metadata'].get('law')
                        }

                    # Create issue record
                    issue = {
                        "section_number": section_number,
                        "contract_clause": clause_text[:500],  # Truncate
                        "severity": analysis.get("severity", "unknown"),
                        "issue_description": analysis.get("violation_description", ""),
                        "violated_law": violated_law,
                        "recommendation": analysis.get("recommendation", ""),
                        "confidence": "uncertain" if has_violation is None else "confident"
                    }

                    result.add_issue(issue)
                    logger.info(f"Issue found in section {section_number}: {issue['severity']}")

            except Exception as e:
                logger.error(f"Error analyzing section {section_number}: {e}")
                continue

        # Step 4: Generate summary and calculate status
        logger.info("Step 4: Generating summary...")
        result.calculate_overall_status()
        result.generate_summary()

        # Add article summary
        article_summary = self.rag_engine.get_relevant_articles_summary(contract)
        result.metadata["relevant_articles"] = article_summary

        logger.info(
            f"Analysis complete: {len(result.issues)} issues found, "
            f"status: {result.overall_status}"
        )

        return result

    def analyze_text(
        self,
        text: str,
        contract_id: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze raw contract text (without parsing from file)

        Args:
            text: Contract text
            contract_id: Optional contract identifier

        Returns:
            AnalysisResult object
        """
        logger.info("Analyzing raw contract text...")

        # Create temporary parsed contract
        from .contract_parser import ParsedContract

        contract_id = contract_id or "text_input"
        contract = ParsedContract(contract_id, text)

        # Detect sections
        self.parser._detect_sections(contract)

        # Create result
        result = AnalysisResult(contract_id)

        # Perform analysis similar to analyze_contract
        retrieval_results = self.rag_engine.retrieve_for_contract(contract)

        for i, retrieval_result in enumerate(retrieval_results, 1):
            section = retrieval_result.get("section")
            context = retrieval_result["context"]

            clause_text = section.text if section else text[:2000]
            section_number = section.section_number if section else str(i)

            laws_text = context.get_context_text(max_chunks=5)

            if not context.retrieved_chunks:
                continue

            try:
                analysis = self.llm_client.analyze_clause(clause_text, laws_text)

                if analysis.get("has_violation"):
                    violated_law = None
                    if context.retrieved_chunks:
                        top_chunk = context.retrieved_chunks[0]
                        violated_law = {
                            "article": top_chunk['metadata'].get('article'),
                            "text": top_chunk['text'][:500],
                            "law": top_chunk['metadata'].get('law')
                        }

                    issue = {
                        "section_number": section_number,
                        "contract_clause": clause_text[:500],
                        "severity": analysis.get("severity", "unknown"),
                        "issue_description": analysis.get("violation_description", ""),
                        "violated_law": violated_law,
                        "recommendation": analysis.get("recommendation", "")
                    }

                    result.add_issue(issue)

            except Exception as e:
                logger.error(f"Error analyzing section: {e}")
                continue

        result.calculate_overall_status()
        result.generate_summary()

        return result


if __name__ == "__main__":
    # Test analyzer
    import sys
    import json
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python contract_analyzer.py <contract_file>")
        sys.exit(1)

    # Initialize analyzer
    analyzer = ContractAnalyzer()

    # Analyze contract
    result = analyzer.analyze_contract(sys.argv[1])

    # Print results
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\nContract: {result.contract_path}")
    print(f"Status: {result.overall_status}")
    print(f"Issues Found: {len(result.issues)}")

    print(f"\nSummary:\n{result.summary}")

    if result.issues:
        print("\nDetailed Issues:")
        for issue in result.issues:
            print(f"\n  Issue #{issue['issue_id']} - Severity: {issue['severity']}")
            print(f"  Section: {issue['section_number']}")
            print(f"  Description: {issue['issue_description']}")
            if issue.get('violated_law'):
                print(f"  Violated Article: {issue['violated_law']['article']}")
            if issue.get('recommendation'):
                print(f"  Recommendation: {issue['recommendation']}")

    # Save to JSON
    output_file = "analysis_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\n✓ Full results saved to: {output_file}")
