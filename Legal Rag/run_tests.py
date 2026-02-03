"""
Test runner script for Legal RAG System
"""

import sys
import subprocess
from pathlib import Path


def run_pytest():
    """Run pytest test suite"""
    print("=" * 80)
    print("Running pytest test suite...")
    print("=" * 80)

    try:
        result = subprocess.run(
            ["pytest", "tests/", "-v", "--tb=short"],
            capture_output=False,
            cwd=Path(__file__).parent
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("\n❌ pytest not installed. Install with: pip install pytest")
        return False


def run_smoke_tests():
    """Run basic smoke tests without dependencies"""
    print("\n" + "=" * 80)
    print("Running smoke tests...")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, "validate_setup.py"],
            capture_output=False,
            cwd=Path(__file__).parent
        )
        return result.returncode == 0
    except Exception as e:
        print(f"\n❌ Error running smoke tests: {e}")
        return False


def main():
    """Main test runner"""
    print("\n🧪 Legal RAG System - Test Suite\n")

    # Run smoke tests first
    smoke_passed = run_smoke_tests()

    if not smoke_passed:
        print("\n❌ Smoke tests failed! Fix issues before running full tests.")
        return 1

    # Run full test suite
    tests_passed = run_pytest()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Smoke Tests: {'✅ PASSED' if smoke_passed else '❌ FAILED'}")
    print(f"Unit Tests:  {'✅ PASSED' if tests_passed else '❌ FAILED'}")

    if smoke_passed and tests_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
