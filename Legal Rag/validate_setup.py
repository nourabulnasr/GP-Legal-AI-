"""
Validation script to check project setup without requiring all dependencies
"""

import sys
import json
from pathlib import Path


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def check_mark(passed: bool) -> str:
    """Return colored checkmark or X"""
    if passed:
        return f"{Colors.GREEN}[OK]{Colors.END}"
    else:
        return f"{Colors.RED}[X]{Colors.END}"


def validate_project_structure():
    """Validate project directory structure"""
    print("\n[Project Structure]")

    required_dirs = [
        "src",
        "data",
        "templates",
        "tests"
    ]

    all_exist = True
    for dir_name in required_dirs:
        exists = Path(dir_name).is_dir()
        print(f"  {check_mark(exists)} {dir_name}/")
        all_exist = all_exist and exists

    return all_exist


def validate_source_files():
    """Validate all source files exist"""
    print("\n[Source Files]")

    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/vector_store.py",
        "src/data_ingestion.py",
        "src/contract_parser.py",
        "src/rag_engine.py",
        "src/llm_client.py",
        "src/contract_analyzer.py",
        "src/report_generator.py",
        "src/main.py",
        "src/api.py"
    ]

    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).is_file()
        print(f"  {check_mark(exists)} {file_path}")
        all_exist = all_exist and exists

    return all_exist


def validate_config_files():
    """Validate configuration files"""
    print("\n[Configuration Files]")

    files = {
        "config.yaml": True,
        "requirements.txt": True,
        "README.md": True,
        "SETUP.md": True,
        "USAGE_GUIDE.md": True
    }

    all_exist = True
    for file_name, required in files.items():
        exists = Path(file_name).is_file()
        print(f"  {check_mark(exists)} {file_name}")
        if required:
            all_exist = all_exist and exists

    return all_exist


def validate_data_files():
    """Validate data files exist"""
    print("\n[Data Files]")

    data_file = Path("data/labor14_2025_chunks.cleaned.jsonl")
    sample_contract = Path("data/sample_contracts/sample_contract.txt")

    results = {
        "Labor law data": data_file.is_file(),
        "Sample contract": sample_contract.is_file()
    }

    all_exist = True
    for name, exists in results.items():
        print(f"  {check_mark(exists)} {name}")
        all_exist = all_exist and exists

    # Validate JSONL structure if file exists
    if data_file.is_file():
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                doc = json.loads(first_line)

                required_fields = ['id', 'law', 'article', 'text']
                has_all_fields = all(field in doc for field in required_fields)

                print(f"  {check_mark(has_all_fields)} JSONL structure valid")
                all_exist = all_exist and has_all_fields

                # Count lines
                f.seek(0)
                line_count = sum(1 for _ in f)
                print(f"  {Colors.BLUE}[i]{Colors.END} {line_count} law chunks in database")

        except Exception as e:
            print(f"  {check_mark(False)} Error reading JSONL: {e}")
            all_exist = False

    return all_exist


def validate_templates():
    """Validate template files"""
    print("\n[Templates]")

    template_file = Path("templates/report_template_ar.html")
    exists = template_file.is_file()

    print(f"  {check_mark(exists)} templates/report_template_ar.html")

    if exists:
        content = template_file.read_text(encoding='utf-8')
        has_rtl = 'dir="rtl"' in content
        print(f"  {check_mark(has_rtl)} RTL support in template")
        return has_rtl

    return exists


def validate_test_files():
    """Validate test files exist"""
    print("\n[Test Files]")

    test_files = [
        "tests/__init__.py",
        "tests/test_config.py",
        "tests/test_contract_parser.py",
        "tests/test_data_ingestion.py",
        "tests/test_rag_engine.py",
        "tests/test_integration.py"
    ]

    all_exist = True
    for file_path in test_files:
        exists = Path(file_path).is_file()
        print(f"  {check_mark(exists)} {file_path}")
        all_exist = all_exist and exists

    return all_exist


def validate_python_syntax():
    """Validate Python syntax of all source files"""
    print("\n[Python Syntax]")

    import py_compile

    src_files = list(Path("src").glob("*.py"))
    all_valid = True

    for file_path in src_files:
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"  {check_mark(True)} {file_path.name}")
        except py_compile.PyCompileError as e:
            print(f"  {check_mark(False)} {file_path.name}: {e}")
            all_valid = False

    return all_valid


def check_imports():
    """Check if modules can be imported"""
    print("\n[Module Imports]")

    sys.path.insert(0, str(Path.cwd()))

    modules = [
        "src.config",
        "src.contract_parser",
        "src.data_ingestion"
    ]

    all_imported = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  {check_mark(True)} {module_name}")
        except Exception as e:
            print(f"  {check_mark(False)} {module_name}: {e}")
            all_imported = False

    return all_imported


def main():
    """Main validation function"""
    print(f"\n{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BLUE}Legal RAG System - Setup Validation{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 80}{Colors.END}")

    checks = [
        ("Project Structure", validate_project_structure),
        ("Source Files", validate_source_files),
        ("Configuration Files", validate_config_files),
        ("Data Files", validate_data_files),
        ("Templates", validate_templates),
        ("Test Files", validate_test_files),
        ("Python Syntax", validate_python_syntax),
        ("Module Imports", check_imports)
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n  {Colors.RED}Error in {check_name}: {e}{Colors.END}")
            results[check_name] = False

    # Summary
    print(f"\n{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BLUE}VALIDATION SUMMARY{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 80}{Colors.END}\n")

    all_passed = all(results.values())

    for check_name, passed in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.END}" if passed else f"{Colors.RED}FAILED{Colors.END}"
        print(f"  {check_name:25s} {status}")

    print(f"\n{Colors.BLUE}{'=' * 80}{Colors.END}")

    if all_passed:
        print(f"\n{Colors.GREEN}[OK] All validation checks passed!{Colors.END}")
        print(f"\n{Colors.BLUE}Next steps:{Colors.END}")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Load data: python -m src.data_ingestion")
        print("  3. Run tests: python run_tests.py")
        print("  4. Analyze contract: python -m src.main analyze --contract <file> --output report.json")
        return 0
    else:
        print(f"\n{Colors.YELLOW}[WARNING] Some validation checks failed.{Colors.END}")
        print("Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
