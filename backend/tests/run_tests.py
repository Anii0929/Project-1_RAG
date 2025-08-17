#!/usr/bin/env python3
"""
Test runner for the RAG system tests.
Runs all tests and provides detailed output about failures.
"""

import os
import sys
import unittest

# Add backend directory to path for imports
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, backend_path)


def run_all_tests():
    """Run all test modules and report results"""

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(
                f"- {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}"
            )

    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(
                f"- {test}: {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'Import or setup error'}"
            )

    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
