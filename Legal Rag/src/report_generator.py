"""
Report Generator module for creating multi-format reports (JSON, HTML, PDF)
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import get_config


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports in multiple formats from analysis results"""

    def __init__(self, config=None):
        """
        Initialize report generator

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()

        # Setup Jinja2 environment for HTML templates
        template_dir = Path(__file__).parent.parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_json(
        self,
        analysis_result,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate JSON report

        Args:
            analysis_result: AnalysisResult object
            output_path: Output file path (optional)

        Returns:
            JSON string
        """
        logger.info("Generating JSON report...")

        # Convert to dictionary
        report_data = analysis_result.to_dict()

        # Generate JSON string
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

            logger.info(f"JSON report saved to: {output_path}")

        return json_str

    def generate_html(
        self,
        analysis_result,
        output_path: Optional[str] = None,
        template_name: str = "report_template_ar.html"
    ) -> str:
        """
        Generate HTML report

        Args:
            analysis_result: AnalysisResult object
            output_path: Output file path (optional)
            template_name: Template file name

        Returns:
            HTML string
        """
        logger.info("Generating HTML report...")

        try:
            # Load template
            template = self.jinja_env.get_template(template_name)

            # Prepare data for template
            report_data = analysis_result.to_dict()

            # Add formatting helpers
            report_data['generation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Map severity to colors
            severity_colors = {
                "عالية": "danger",
                "high": "danger",
                "متوسطة": "warning",
                "medium": "warning",
                "منخفضة": "info",
                "low": "info",
                "unknown": "secondary"
            }

            report_data['severity_colors'] = severity_colors

            # Map status to colors
            status_colors = {
                "compliant": "success",
                "issues_found": "warning",
                "major_violations": "danger"
            }

            report_data['status_color'] = status_colors.get(
                analysis_result.overall_status,
                "secondary"
            )

            # Render template
            html_content = template.render(**report_data)

            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                logger.info(f"HTML report saved to: {output_path}")

            return html_content

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"

    def generate_pdf(
        self,
        analysis_result,
        output_path: str
    ) -> bool:
        """
        Generate PDF report

        Args:
            analysis_result: AnalysisResult object
            output_path: Output file path

        Returns:
            True if successful
        """
        logger.info("Generating PDF report...")

        try:
            # First generate HTML
            html_content = self.generate_html(analysis_result)

            # Convert HTML to PDF using WeasyPrint
            try:
                from weasyprint import HTML, CSS
                from weasyprint.text.fonts import FontConfiguration

                # Configure fonts for Arabic support
                font_config = FontConfiguration()

                # Create CSS for better rendering
                css = CSS(string='''
                    @page {
                        size: A4;
                        margin: 2cm;
                    }
                    body {
                        font-family: 'Arial', 'Tahoma', sans-serif;
                        direction: rtl;
                    }
                ''', font_config=font_config)

                # Generate PDF
                HTML(string=html_content).write_pdf(
                    output_path,
                    stylesheets=[css],
                    font_config=font_config
                )

                logger.info(f"PDF report saved to: {output_path}")
                return True

            except ImportError:
                logger.error(
                    "WeasyPrint not available. Install with: pip install weasyprint\n"
                    "Note: WeasyPrint requires additional system dependencies."
                )
                return False

        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return False

    def generate_all_formats(
        self,
        analysis_result,
        base_filename: str
    ) -> Dict[str, str]:
        """
        Generate reports in all configured formats

        Args:
            analysis_result: AnalysisResult object
            base_filename: Base filename (without extension)

        Returns:
            Dictionary mapping format to output path
        """
        logger.info(f"Generating reports in all formats for: {base_filename}")

        # Create reports directory
        reports_dir = Path(self.config.output.reports_directory)
        reports_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}
        formats = self.config.output.formats

        # Generate JSON
        if "json" in formats:
            json_path = reports_dir / f"{base_filename}.json"
            self.generate_json(analysis_result, str(json_path))
            output_paths["json"] = str(json_path)

        # Generate HTML
        if "html" in formats:
            html_path = reports_dir / f"{base_filename}.html"
            self.generate_html(analysis_result, str(html_path))
            output_paths["html"] = str(html_path)

        # Generate PDF
        if "pdf" in formats:
            pdf_path = reports_dir / f"{base_filename}.pdf"
            if self.generate_pdf(analysis_result, str(pdf_path)):
                output_paths["pdf"] = str(pdf_path)

        logger.info(f"Generated {len(output_paths)} report(s)")
        return output_paths


if __name__ == "__main__":
    # Test report generator with mock data
    import sys
    logging.basicConfig(level=logging.INFO)

    # Create mock analysis result
    from .contract_analyzer import AnalysisResult

    result = AnalysisResult("test_contract.pdf")
    result.contract_id = "test-123"
    result.overall_status = "issues_found"

    # Add sample issues
    result.add_issue({
        "section_number": "1",
        "contract_clause": "يعمل الموظف 60 ساعة في الأسبوع...",
        "severity": "عالية",
        "issue_description": "تجاوز ساعات العمل القانونية",
        "violated_law": {
            "article": "100",
            "text": "لا يجوز تشغيل العامل أكثر من 48 ساعة في الأسبوع",
            "law": "قانون العمل رقم 14 لسنة 2025"
        },
        "recommendation": "تعديل ساعات العمل لتكون 48 ساعة أسبوعياً"
    })

    result.generate_summary()

    # Initialize generator
    generator = ReportGenerator()

    # Generate reports
    print("\nGenerating test reports...")
    output_paths = generator.generate_all_formats(result, "test_report")

    print("\nGenerated reports:")
    for format_type, path in output_paths.items():
        print(f"  {format_type.upper()}: {path}")
