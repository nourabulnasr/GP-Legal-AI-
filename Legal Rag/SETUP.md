# Legal RAG System - Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd "c:\Users\ahmed\Desktop\Experts House\code\Legal Rag"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Initialize Vector Database

Load the labor law data into ChromaDB:

```bash
python -m src.data_ingestion
```

This will:
- Load 154 chunks from `data/labor14_2025_chunks.cleaned.jsonl`
- Generate Arabic embeddings using `aubmindlab/bert-base-arabertv2`
- Store in ChromaDB at `chroma_db/`

### 3. Test the System

#### Option A: CLI Mode

Analyze a sample contract:

```bash
python -m src.main analyze --contract "data/sample_contracts/sample_contract.txt" --output "reports/test_report.json" --verbose
```

Generate all report formats:

```bash
python -m src.main analyze --contract "data/sample_contracts/sample_contract.txt" --output "reports/test_report" --format all
```

#### Option B: API Mode

Start the API server:

```bash
python -m src.main api
```

Access API documentation at: `http://localhost:8000/docs`

Test with curl:

```bash
# Upload and analyze contract
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@data/sample_contracts/sample_contract.txt"

# Get report (replace {contract_id} with actual ID from previous response)
curl "http://localhost:8000/report/{contract_id}?format=json"
```

## Configuration

Edit `config.yaml` to customize:

### LLM Settings
```yaml
llm:
  model_name: "LiquidAI/LFM2.5-1.2B-Instruct"
  device: "cuda"  # or "cpu"
  load_in_8bit: false  # set true to reduce memory usage
  temperature: 0.1
```

### Embedding Settings
```yaml
embeddings:
  model_name: "aubmindlab/bert-base-arabertv2"
  device: "cuda"  # or "cpu"
```

### RAG Parameters
```yaml
rag:
  section_top_k: 10  # chunks to retrieve per section
  clause_top_k: 5    # chunks to retrieve per clause
  min_similarity_score: 0.5
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA memory errors:

1. Enable 8-bit quantization in `config.yaml`:
```yaml
llm:
  load_in_8bit: true
```

2. Or use CPU mode:
```yaml
llm:
  device: "cpu"
embeddings:
  device: "cpu"
```

### Missing Models

Models will auto-download on first use. If downloads fail:

1. Check internet connection
2. Manually download from HuggingFace:
   - `aubmindlab/bert-base-arabertv2`
   - `LiquidAI/LFM2.5-1.2B-Instruct`

### PDF Generation Issues

If PDF generation fails:

1. Install WeasyPrint dependencies (OS-specific)
2. Use HTML or JSON output instead
3. Set `formats: ["json", "html"]` in config.yaml

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB free disk space

### Recommended
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 20GB free disk space

## Project Structure

```
Legal Rag/
├── data/
│   ├── labor14_2025_chunks.cleaned.jsonl  # Labor law data (154 chunks)
│   └── sample_contracts/                   # Test contracts
├── chroma_db/                              # Vector database (auto-created)
├── reports/                                # Generated reports (auto-created)
├── src/                                    # Source code
│   ├── config.py                          # Configuration management
│   ├── data_ingestion.py                  # Load laws into ChromaDB
│   ├── vector_store.py                    # ChromaDB wrapper
│   ├── contract_parser.py                 # Multi-format parsing
│   ├── rag_engine.py                      # Retrieval engine
│   ├── llm_client.py                      # LLM interface
│   ├── contract_analyzer.py               # Analysis orchestration
│   ├── report_generator.py                # Report generation
│   ├── main.py                            # CLI/API entry point
│   └── api.py                             # FastAPI endpoints
├── templates/                              # HTML templates
│   └── report_template_ar.html            # Arabic report template
├── config.yaml                            # Configuration file
├── requirements.txt                       # Python dependencies
└── README.md                              # Documentation
```

## Next Steps

1. **Add More Laws**: Add additional JSONL files to `data/` and re-run data ingestion
2. **Fine-tune Parameters**: Adjust RAG parameters in `config.yaml` for better results
3. **Create Custom Templates**: Modify `templates/report_template_ar.html` for custom branding
4. **Deploy API**: Use Docker or cloud services to deploy the API

## Support

For issues or questions:
- Check logs in `legal_rag.log`
- Review error messages carefully
- Ensure all dependencies are installed
- Verify CUDA is available (if using GPU)
