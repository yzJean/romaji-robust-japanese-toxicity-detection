#!/usr/bin/env python3
"""
Test runner for the romaji-robust-japanese-toxicity-detection project.
Runs all unit and integration tests.
"""

import sys
import os
import unittest

# Add project directories to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'scripts'))
sys.path.insert(0, os.path.join(project_root, 'tests'))


def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Get the tests directory
    tests_dir = os.path.join(project_root, 'tests')
    
    # Discover all tests
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
