# Legal RAG System - Complete Usage Guide

## Overview

This system analyzes employment contracts against Egyptian Labor Law No. 14/2025 using:
- **RAG (Retrieval-Augmented Generation)** for finding relevant laws
- **LiquidAI/LFM2.5-1.2B-Instruct** LLM for legal analysis
- **ChromaDB** with Arabic embeddings for vector search

## Installation

### Step 1: Setup Environment

```bash
# Navigate to project
cd "c:\Users\ahmed\Desktop\Experts House\code\Legal Rag"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Initialize Database

Load Egyptian labor law data:

```bash
python -m src.data_ingestion
```

Expected output:
```
Loading data from: ./data/labor14_2025_chunks.cleaned.jsonl
Loaded 154 documents
Generating embeddings for 154 documents...
✓ Data ingestion completed successfully!

Collection Statistics:
  Name: egyptian_labor_laws
  Documents: 154
```

## Usage

### CLI Mode

#### Basic Analysis

```bash
python -m src.main analyze \
  --contract "path/to/contract.pdf" \
  --output "report.json"
```

#### Generate All Formats (JSON, HTML, PDF)

```bash
python -m src.main analyze \
  --contract "data/sample_contracts/sample_contract.txt" \
  --output "reports/my_report" \
  --format all \
  --verbose
```

#### Supported Formats

- **PDF**: `.pdf` files (best for scanned documents)
- **DOCX**: `.docx`, `.doc` files (Microsoft Word)
- **TXT**: `.txt` files (plain text)

### API Mode

#### Start Server

```bash
python -m src.main api
```

Server runs at: `http://localhost:8000`

#### API Endpoints

##### 1. Health Check
```bash
curl http://localhost:8000/health
```

##### 2. Analyze File
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@contract.pdf" \
  -F "format=json"
```

Response:
```json
{
  "contract_id": "abc-123-def",
  "overall_status": "issues_found",
  "total_issues": 3,
  "summary": "تم العثور على 3 مشاكل محتملة...",
  "report_url": "/report/abc-123-def"
}
```

##### 3. Analyze Text
```bash
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "يعمل الموظف 60 ساعة في الأسبوع...",
    "contract_id": "my-contract-1"
  }'
```

##### 4. Get Report
```bash
# Get JSON report
curl "http://localhost:8000/report/abc-123-def?format=json"

# Get HTML report
curl "http://localhost:8000/report/abc-123-def?format=html" > report.html

# Get PDF report
curl "http://localhost:8000/report/abc-123-def?format=pdf" > report.pdf
```

##### 5. Delete Report
```bash
curl -X DELETE "http://localhost:8000/report/abc-123-def"
```

#### API Documentation

Interactive API docs: `http://localhost:8000/docs`

## Output Format

### JSON Structure

```json
{
  "contract_id": "abc-123",
  "contract_path": "contract.pdf",
  "analysis_date": "2025-01-31T01:00:00",
  "overall_status": "issues_found",
  "total_issues": 3,
  "summary": "تم العثور على 3 مشاكل محتملة في العقد...",
  "issues": [
    {
      "issue_id": 1,
      "section_number": "2",
      "contract_clause": "يعمل الموظف 60 ساعة في الأسبوع...",
      "severity": "عالية",
      "issue_description": "تجاوز ساعات العمل القانونية",
      "violated_law": {
        "article": "100",
        "text": "لا يجوز تشغيل العامل أكثر من 48 ساعة...",
        "law": "قانون العمل رقم 14 لسنة 2025"
      },
      "recommendation": "تعديل ساعات العمل لتكون 48 ساعة"
    }
  ],
  "metadata": {
    "total_sections": 7,
    "word_count": 245
  }
}
```

### Status Codes

- `compliant`: No violations found
- `issues_found`: Minor/medium issues detected
- `major_violations`: High severity issues found

### Severity Levels

- `عالية` (high): Serious labor law violation
- `متوسطة` (medium): Moderate concern
- `منخفضة` (low): Minor issue

## Configuration

Edit `config.yaml` to customize behavior:

### LLM Settings

```yaml
llm:
  model_name: "LiquidAI/LFM2.5-1.2B-Instruct"
  device: "cuda"  # or "cpu"
  load_in_8bit: false  # enable for lower memory
  temperature: 0.1  # lower = more consistent
  max_new_tokens: 1024
```

### RAG Parameters

```yaml
rag:
  section_top_k: 10  # retrieve 10 laws per section
  clause_top_k: 5    # retrieve 5 laws per clause
  min_similarity_score: 0.5  # minimum relevance threshold
```

### Output Settings

```yaml
output:
  formats: ["json", "html", "pdf"]
  default_format: "json"
  reports_directory: "./reports"
```

## Examples

### Example 1: Analyze Sample Contract

```bash
# The sample contract has intentional violations
python -m src.main analyze \
  --contract "data/sample_contracts/sample_contract.txt" \
  --output "reports/sample_analysis.json" \
  --verbose
```

Expected violations:
1. **60 hours/week** → Violates 48-hour limit
2. **10 days annual leave** → Violates 21-day minimum
3. **Termination without notice** → Violates notice requirements

### Example 2: Batch Processing

```bash
# Process multiple contracts
for contract in contracts/*.pdf; do
  python -m src.main analyze \
    --contract "$contract" \
    --output "reports/$(basename $contract .pdf).json"
done
```

### Example 3: Custom API Integration

```python
import requests

# Upload contract
with open('contract.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f}
    )

result = response.json()
contract_id = result['contract_id']

# Get detailed report
report = requests.get(
    f'http://localhost:8000/report/{contract_id}',
    params={'format': 'json'}
).json()

print(f"Found {report['total_issues']} issues")
for issue in report['issues']:
    print(f"- {issue['issue_description']}")
```

## Advanced Features

### Custom Prompts

Edit `src/llm_client.py` to customize analysis prompts:

```python
def create_analysis_prompt(self, contract_clause, retrieved_laws):
    # Customize prompt here
    pass
```

### Additional Laws

Add new JSONL files to `data/` directory:

```bash
python -m src.data_ingestion --file data/new_law.jsonl
```

JSONL format:
```json
{"id": "law_art_1", "law": "قانون...", "article": "1", "chunk_index": 1, "text": "...", "source": "law_id"}
```

### Fine-tune Retrieval

Adjust retrieval quality in `config.yaml`:

```yaml
rag:
  section_top_k: 15  # more context (slower)
  min_similarity_score: 0.6  # stricter matching
  use_mmr: true  # enable diversity
  mmr_diversity: 0.3
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Enable quantization
```yaml
llm:
  load_in_8bit: true
```

**Solution 2**: Use CPU
```yaml
llm:
  device: "cpu"
embeddings:
  device: "cpu"
```

### Issue: Slow Analysis

**Solutions**:
- Reduce `section_top_k` and `clause_top_k`
- Use GPU instead of CPU
- Enable 8-bit quantization
- Decrease `max_new_tokens`

### Issue: Poor Quality Results

**Solutions**:
- Increase `section_top_k` and `clause_top_k`
- Lower `min_similarity_score`
- Adjust LLM `temperature` (try 0.2)
- Add more relevant laws to database

### Issue: PDF Parsing Errors

**Solutions**:
- Try converting PDF to TXT first
- Use OCR for scanned PDFs
- Check PDF is not encrypted/protected
- Install pdfplumber dependencies

## Performance

### Typical Processing Times

- **Data Ingestion**: 2-5 minutes (one-time)
- **Contract Parsing**: 1-5 seconds
- **RAG Retrieval**: 0.5-2 seconds per section
- **LLM Analysis**: 5-15 seconds per section
- **Total (5-section contract)**: 30-90 seconds

### Memory Requirements

- **CPU Mode**: ~8GB RAM
- **GPU Mode**: ~6GB VRAM + 4GB RAM
- **8-bit Quantization**: ~3GB VRAM + 4GB RAM

## Best Practices

1. **Use GPU** for faster processing
2. **Enable 8-bit quantization** if memory limited
3. **Adjust top_k** based on contract complexity
4. **Review sample outputs** before production use
5. **Validate results** with legal experts
6. **Keep logs** for debugging (`legal_rag.log`)

## Limitations

- **Language**: Primarily Arabic (MSA)
- **Scope**: Egyptian Labor Law No. 14/2025 only
- **Model Size**: 1.2B parameters (limited reasoning)
- **Accuracy**: Not 100% - requires human review
- **Legal Value**: Informational only, not legal advice

## Support & Contribution

- **Logs**: Check `legal_rag.log` for errors
- **Config**: Review `config.yaml` settings
- **Models**: Ensure HuggingFace access
- **Data**: Verify JSONL format correctness

## License

Educational and research purposes only.
