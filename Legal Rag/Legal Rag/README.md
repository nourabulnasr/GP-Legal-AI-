# Legal AI RAG System

A Retrieval-Augmented Generation (RAG) system for analyzing employment contracts against Egyptian labor laws.

## Overview

This system uses:
- **ChromaDB** for vector storage of labor law articles
- **LiquidAI/LFM2.5-1.2B-Instruct** LLM for legal analysis
- **Arabic-optimized embeddings** (aubmindlab/bert-base-arabertv2)
- **Multi-format support** for contracts (PDF, DOCX, TXT)

## Features

- Analyze employment contracts for labor law violations
- Support for Egyptian Labor Law No. 14/2025
- Multi-format contract parsing (PDF, DOCX, plain text)
- Structured output (JSON, HTML, PDF reports)
- Both CLI and REST API interfaces
- Arabic language support with RTL text handling

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

1. Clone the repository
```bash
cd "Legal Rag"
```

2. Create and activate virtual environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create environment file
```bash
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

5. Initialize the vector database (first-time setup)
```bash
python -m src.data_ingestion
```

## Usage

### Command Line Interface (CLI)

Analyze a contract:
```bash
python -m src.main analyze --contract path/to/contract.pdf --output report.json
```

Options:
- `--contract`: Path to contract file (PDF, DOCX, or TXT)
- `--output`: Output file path for the report
- `--format`: Output format (json, html, pdf) - default: json

### REST API

Start the API server:
```bash
python -m src.main api
```

The API will be available at `http://localhost:8000`

Endpoints:
- `POST /analyze`: Upload contract for analysis
- `GET /report/{contract_id}`: Retrieve analysis report
- `GET /health`: Health check

API Documentation: `http://localhost:8000/docs`

## Configuration

Edit `config.yaml` to customize:
- Model settings (LLM and embeddings)
- RAG parameters (top-k, similarity thresholds)
- Output preferences
- API settings

## Project Structure

```
legal-rag/
├── data/
│   ├── labor14_2025_chunks.cleaned.jsonl  # Egyptian labor law data
│   └── sample_contracts/                   # Test contracts
├── chroma_db/                              # Vector database (auto-created)
├── src/
│   ├── config.py                          # Configuration management
│   ├── data_ingestion.py                  # Load laws into ChromaDB
│   ├── vector_store.py                    # ChromaDB wrapper
│   ├── contract_parser.py                 # Multi-format contract parsing
│   ├── rag_engine.py                      # Retrieval engine
│   ├── llm_client.py                      # LLM interface
│   ├── contract_analyzer.py               # Analysis orchestration
│   ├── report_generator.py                # Report generation
│   └── main.py                            # CLI/API entry point
├── templates/                              # HTML report templates
├── tests/                                  # Test suite
├── config.yaml                            # Configuration file
└── requirements.txt                       # Python dependencies
```

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Laws

1. Add new JSONL files to `data/` directory
2. Run data ingestion:
```bash
python -m src.data_ingestion --file data/new_law.jsonl
```

## Output Format

### JSON Output
```json
{
  "contract_id": "...",
  "analysis_date": "...",
  "overall_status": "compliant|issues_found|major_violations",
  "issues": [
    {
      "issue_id": 1,
      "severity": "high|medium|low",
      "contract_clause": "...",
      "issue_description": "...",
      "violated_law": {
        "article": "1",
        "text": "...",
        "law": "قانون العمل رقم 14 لسنة 2025"
      },
      "recommendation": "..."
    }
  ],
  "summary": "..."
}
```

## License

This project is for educational and research purposes.

## Contributors

Legal RAG Team
