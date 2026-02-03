"""
FastAPI REST API for Legal RAG System
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import Config
from .contract_analyzer import ContractAnalyzer
from .report_generator import ReportGenerator


logger = logging.getLogger(__name__)


# Pydantic models for API
class AnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str
    contract_id: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response model for analysis"""
    contract_id: str
    overall_status: str
    total_issues: int
    summary: str
    report_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


def create_app(config: Config) -> FastAPI:
    """
    Create and configure FastAPI application

    Args:
        config: Configuration object

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Legal RAG API",
        description="REST API for Egyptian Labor Law Contract Analysis",
        version="0.1.0"
    )

    # Configure CORS
    if config.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Initialize components
    analyzer = ContractAnalyzer(config)
    report_generator = ReportGenerator(config)

    # Store for analysis results (in production, use a database)
    analysis_cache = {}

    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint"""
        return {
            "message": "Legal RAG API",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "analyze_file": "/analyze (POST with file upload)",
                "analyze_text": "/analyze/text (POST with JSON)",
                "get_report": "/report/{contract_id}"
            }
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            version="0.1.0"
        )

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_contract_file(
        file: UploadFile = File(...),
        format: str = "json",
        background_tasks: BackgroundTasks = None
    ):
        """
        Analyze a contract file

        Args:
            file: Uploaded contract file (PDF, DOCX, or TXT)
            format: Output format (json, html, pdf)

        Returns:
            Analysis response with contract_id
        """
        logger.info(f"Received file for analysis: {file.filename}")

        # Validate file size
        max_size = config.api.max_upload_size_mb * 1024 * 1024
        content = await file.read()

        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.api.max_upload_size_mb}MB"
            )

        # Save to temporary file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Analyze contract
            logger.info(f"Analyzing contract: {temp_path}")
            result = analyzer.analyze_contract(temp_path)

            # Generate report
            report_filename = f"report_{result.contract_id}"
            output_paths = report_generator.generate_all_formats(result, report_filename)

            # Store result in cache
            analysis_cache[result.contract_id] = {
                "result": result,
                "output_paths": output_paths
            }

            # Return response
            return AnalysisResponse(
                contract_id=result.contract_id,
                overall_status=result.overall_status,
                total_issues=len(result.issues),
                summary=result.summary,
                report_url=f"/report/{result.contract_id}"
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

    @app.post("/analyze/text", response_model=AnalysisResponse)
    async def analyze_contract_text(request: AnalysisRequest):
        """
        Analyze contract text directly

        Args:
            request: AnalysisRequest with text content

        Returns:
            Analysis response with contract_id
        """
        logger.info("Analyzing contract text")

        try:
            # Analyze text
            result = analyzer.analyze_text(request.text, request.contract_id)

            # Generate reports
            report_filename = f"report_{result.contract_id}"
            output_paths = report_generator.generate_all_formats(result, report_filename)

            # Store result
            analysis_cache[result.contract_id] = {
                "result": result,
                "output_paths": output_paths
            }

            return AnalysisResponse(
                contract_id=result.contract_id,
                overall_status=result.overall_status,
                total_issues=len(result.issues),
                summary=result.summary,
                report_url=f"/report/{result.contract_id}"
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    @app.get("/report/{contract_id}")
    async def get_report(contract_id: str, format: str = "json"):
        """
        Get analysis report by contract ID

        Args:
            contract_id: Contract identifier
            format: Report format (json, html, pdf)

        Returns:
            Report file or JSON data
        """
        if contract_id not in analysis_cache:
            raise HTTPException(status_code=404, detail="Report not found")

        cached_data = analysis_cache[contract_id]
        result = cached_data["result"]
        output_paths = cached_data["output_paths"]

        # Return appropriate format
        if format == "json":
            return JSONResponse(content=result.to_dict())

        elif format == "html":
            if "html" in output_paths:
                return FileResponse(
                    output_paths["html"],
                    media_type="text/html",
                    filename=f"report_{contract_id}.html"
                )
            else:
                raise HTTPException(status_code=404, detail="HTML report not found")

        elif format == "pdf":
            if "pdf" in output_paths:
                return FileResponse(
                    output_paths["pdf"],
                    media_type="application/pdf",
                    filename=f"report_{contract_id}.pdf"
                )
            else:
                raise HTTPException(status_code=404, detail="PDF report not found")

        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use: json, html, or pdf")

    @app.delete("/report/{contract_id}")
    async def delete_report(contract_id: str):
        """
        Delete a report from cache

        Args:
            contract_id: Contract identifier

        Returns:
            Success message
        """
        if contract_id not in analysis_cache:
            raise HTTPException(status_code=404, detail="Report not found")

        # Delete files
        cached_data = analysis_cache[contract_id]
        for file_path in cached_data["output_paths"].values():
            Path(file_path).unlink(missing_ok=True)

        # Remove from cache
        del analysis_cache[contract_id]

        return {"message": "Report deleted successfully"}

    return app


if __name__ == "__main__":
    # Run API server directly
    import uvicorn
    from .config import get_config

    config = get_config()
    app = create_app(config)

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port
    )
