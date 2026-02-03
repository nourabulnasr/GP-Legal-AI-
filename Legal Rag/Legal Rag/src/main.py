"""
Main entry point for Legal RAG System
Provides both CLI and API interfaces
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import get_config
from .contract_analyzer import ContractAnalyzer
from .report_generator import ReportGenerator


# Configure logging
def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('legal_rag.log', encoding='utf-8')
        ]
    )


def cli_analyze(args):
    """Handle CLI analyze command"""
    config = get_config(args.config)
    setup_logging(config.logging.level)

    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing contract: {args.contract}")

    # Initialize components
    analyzer = ContractAnalyzer(config)
    report_generator = ReportGenerator(config)

    # Analyze contract
    result = analyzer.analyze_contract(args.contract)

    # Determine output format
    output_format = args.format or config.output.default_format

    # Generate report
    if output_format == "json":
        report_generator.generate_json(result, args.output)
        print(f"\n✓ JSON report saved to: {args.output}")

    elif output_format == "html":
        report_generator.generate_html(result, args.output)
        print(f"\n✓ HTML report saved to: {args.output}")

    elif output_format == "pdf":
        if report_generator.generate_pdf(result, args.output):
            print(f"\n✓ PDF report saved to: {args.output}")
        else:
            print("\n✗ Failed to generate PDF report")
            return 1

    elif output_format == "all":
        # Generate all formats
        base_filename = Path(args.output).stem
        output_paths = report_generator.generate_all_formats(result, base_filename)

        print("\n✓ Reports generated:")
        for fmt, path in output_paths.items():
            print(f"  - {fmt.upper()}: {path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nContract: {result.contract_path}")
    print(f"Status: {result.overall_status}")
    print(f"Issues Found: {len(result.issues)}")
    print(f"\n{result.summary}")

    if result.issues and args.verbose:
        print("\n" + "-" * 80)
        print("DETAILED ISSUES")
        print("-" * 80)
        for issue in result.issues:
            print(f"\nIssue #{issue['issue_id']} - Severity: {issue['severity']}")
            print(f"Section: {issue['section_number']}")
            print(f"Description: {issue['issue_description']}")

    return 0


def cli_api(args):
    """Handle CLI API command - start FastAPI server"""
    config = get_config(args.config)
    setup_logging(config.logging.level)

    logger = logging.getLogger(__name__)
    logger.info("Starting API server...")

    try:
        import uvicorn
        from .api import create_app

        # Create FastAPI app
        app = create_app(config)

        # Run server
        uvicorn.run(
            app,
            host=args.host or config.api.host,
            port=args.port or config.api.port,
            log_level=config.logging.level.lower()
        )

    except ImportError:
        print("Error: FastAPI or uvicorn not installed")
        print("Install with: pip install fastapi uvicorn[standard]")
        return 1

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Legal RAG System - Contract Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a contract and output JSON
  python -m src.main analyze --contract contract.pdf --output report.json

  # Analyze and generate all formats
  python -m src.main analyze --contract contract.pdf --output report --format all

  # Start API server
  python -m src.main api

  # Start API server on custom port
  python -m src.main api --port 8080
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a contract file'
    )
    analyze_parser.add_argument(
        '--contract',
        type=str,
        required=True,
        help='Path to contract file (PDF, DOCX, or TXT)'
    )
    analyze_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path for the report'
    )
    analyze_parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'html', 'pdf', 'all'],
        help='Output format (default: from config)'
    )
    analyze_parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print detailed issues to console'
    )

    # API command
    api_parser = subparsers.add_parser(
        'api',
        help='Start REST API server'
    )
    api_parser.add_argument(
        '--host',
        type=str,
        help='Host to bind to (default: from config)'
    )
    api_parser.add_argument(
        '--port',
        type=int,
        help='Port to bind to (default: from config)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'analyze':
        return cli_analyze(args)
    elif args.command == 'api':
        return cli_api(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
