#!/usr/bin/env python3
"""
Comprehensive test suite for SEAR extraction feature.

Tests the extract_relevant_content() function and reconstruct_text_from_chunks() helper.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Import SEAR core functions
from sear.core import (
    delete_corpus,
    extract_relevant_content,
    index_file,
)


def create_test_corpus(tmpdir):
    """
    Create a test corpus with sample content for extraction testing.

    Returns:
        tuple: (test_file_path, corpus_name)
    """
    test_content = """
# Authentication System Documentation

## Overview
This document describes the authentication and security validation system.

## Authentication Methods

### Username/Password Authentication
The system supports traditional username and password authentication.
Users must provide valid credentials to access the system.
Password requirements:
- Minimum 8 characters
- Must contain uppercase and lowercase letters
- Must contain at least one number
- Must contain at least one special character

### OAuth2 Authentication
OAuth2 is supported for third-party authentication providers.
Supported providers:
- Google
- GitHub
- Microsoft

## Security Validation

### Input Validation
All user inputs are validated for security:
- SQL injection prevention
- XSS attack prevention
- CSRF token validation

### Session Management
Sessions are managed securely:
- Session tokens are cryptographically secure
- Sessions expire after 30 minutes of inactivity
- Logout invalidates all session tokens

## Testing Methodologies

### Unit Tests
Unit tests cover individual authentication functions:
- Password hashing verification
- Token generation and validation
- Input sanitization

### Integration Tests
Integration tests verify end-to-end authentication flows:
- User registration and login
- OAuth provider integration
- Session lifecycle management

### Security Testing
Security tests include:
- Penetration testing for common vulnerabilities
- Brute force attack prevention
- Rate limiting verification

## Machine Learning Components

### Neural Networks for Fraud Detection
A neural network model detects fraudulent login attempts.
The model uses attention mechanisms to analyze login patterns.

Key features:
- Real-time fraud scoring
- Transformer-based architecture
- 99.2% accuracy in production

### Natural Language Processing
NLP is used for analyzing user support requests:
- Sentiment analysis for priority routing
- Named entity recognition for issue classification
- Question answering for automated responses

## Performance Metrics

### Response Time
- Average login time: 150ms
- OAuth redirect time: 200ms
- Session validation: 5ms

### Availability
- System uptime: 99.95%
- Failover time: <30 seconds
- Load capacity: 10,000 concurrent users
"""

    # Create temporary test file
    test_file = Path(tmpdir) / "test_auth_docs.txt"
    test_file.write_text(test_content)

    # Create test corpus
    corpus_name = "test_extract_corpus"

    print(f"Creating test corpus '{corpus_name}'...")
    index_file(str(test_file), corpus_name=corpus_name, verbose=False)

    return str(test_file), corpus_name


def test_1_extract_with_default_threshold():
    """
    Test 1: Extract with default threshold (0.3).
    """
    print("\n" + "=" * 80)
    print("TEST 1: Extract with default threshold (0.3)")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_1.txt"

    try:
        test_file, corpus_name = create_test_corpus(tmpdir)

        # Extract authentication-related content
        result = extract_relevant_content(
            query="authentication security validation",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.3,
            verbose=True,
        )

        # Validate results
        assert result["total_chunks"] > 0, "Should find at least one chunk"
        assert result["output_file"] == str(output_file.absolute()), "Output file path should match"
        assert output_file.exists(), "Output file should be created"

        # Check output content
        content = output_file.read_text()
        assert "SEAR EXTRACTED CONTENT" in content, "Output should have header"
        assert "Query:" in content, "Output should include query"
        assert "Min Score:" in content, "Output should include min score"
        assert "authentication" in content.lower(), "Output should contain relevant content"

        print(f"✅ PASSED: Found {result['total_chunks']} chunks")
        print(f"   Score range: {result['score_range'][0]:.3f} - {result['score_range'][1]:.3f}")
        print(f"   Output file: {result['output_file']}")

        # Cleanup
        delete_corpus(corpus_name)

    finally:
        shutil.rmtree(tmpdir)


def test_2_extract_with_custom_threshold():
    """
    Test 2: Extract with custom threshold (0.4).
    """
    print("\n" + "=" * 80)
    print("TEST 2: Extract with custom threshold (0.4)")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_2.txt"

    try:
        test_file, corpus_name = create_test_corpus(tmpdir)

        # Extract with higher threshold
        result_high = extract_relevant_content(
            query="authentication security",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.4,
            verbose=True,
        )

        # Extract with lower threshold for comparison
        output_file_low = Path(tmpdir) / "test_output_2_low.txt"
        result_low = extract_relevant_content(
            query="authentication security",
            corpuses=[corpus_name],
            output_file=str(output_file_low),
            min_score=0.2,
            verbose=False,
        )

        # Validate: higher threshold should have fewer results
        assert (
            result_high["total_chunks"] <= result_low["total_chunks"]
        ), "Higher threshold should return fewer or equal chunks"

        print("✅ PASSED: Threshold filtering works correctly")
        print(f"   Min score 0.4: {result_high['total_chunks']} chunks")
        print(f"   Min score 0.2: {result_low['total_chunks']} chunks")

        # Cleanup
        delete_corpus(corpus_name)

    finally:
        shutil.rmtree(tmpdir)


def test_3_extract_with_max_chunks_limit():
    """
    Test 3: Extract with max_chunks limit.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Extract with max_chunks limit")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_3.txt"

    try:
        test_file, corpus_name = create_test_corpus(tmpdir)

        # Extract with limit
        max_limit = 3
        result = extract_relevant_content(
            query="authentication security testing neural network",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.2,
            max_chunks=max_limit,
            verbose=True,
        )

        # Validate
        assert (
            result["total_chunks"] <= max_limit
        ), f"Should not exceed max_chunks limit of {max_limit}"

        print("✅ PASSED: Max chunks limit respected")
        print(f"   Requested limit: {max_limit}")
        print(f"   Actual chunks: {result['total_chunks']}")

        # Cleanup
        delete_corpus(corpus_name)

    finally:
        shutil.rmtree(tmpdir)


def test_4_extract_from_multiple_corpuses():
    """
    Test 4: Extract from multiple corpuses.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Extract from multiple corpuses")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_4.txt"

    try:
        # Create first corpus
        test_file1, corpus_name1 = create_test_corpus(tmpdir)

        # Create second corpus with different content
        test_content2 = """
# Testing Documentation

## Unit Testing
Unit tests are essential for code quality.
Test coverage should exceed 80%.

## Integration Testing
Integration tests verify component interactions.
End-to-end testing validates user workflows.

## Security Testing
Security testing includes vulnerability scanning.
Penetration testing identifies weaknesses.
"""
        test_file2 = Path(tmpdir) / "test_testing_docs.txt"
        test_file2.write_text(test_content2)
        corpus_name2 = "test_extract_corpus_2"
        index_file(str(test_file2), corpus_name=corpus_name2, verbose=False)

        # Extract from both corpuses
        result = extract_relevant_content(
            query="testing security",
            corpuses=[corpus_name1, corpus_name2],
            output_file=str(output_file),
            min_score=0.3,
            verbose=True,
        )

        # Validate
        assert result["total_chunks"] > 0, "Should find chunks from multiple corpuses"
        assert len(result["sources"]) > 0, "Should have source files"

        content = output_file.read_text()
        assert "SOURCE:" in content, "Output should contain source sections"

        print("✅ PASSED: Multi-corpus extraction works")
        print(f"   Total chunks: {result['total_chunks']}")
        print(f"   Sources: {len(result['sources'])}")

        # Cleanup
        delete_corpus(corpus_name1)
        delete_corpus(corpus_name2)

    finally:
        shutil.rmtree(tmpdir)


def test_5_verify_output_format():
    """
    Test 5: Verify output file format.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Verify output file format")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_5.txt"

    try:
        test_file, corpus_name = create_test_corpus(tmpdir)

        # Extract content
        extract_relevant_content(
            query="authentication",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.3,
            verbose=False,
        )

        # Read and validate output format
        content = output_file.read_text()

        # Check header
        assert "SEAR EXTRACTED CONTENT" in content, "Missing header"
        assert "Query:" in content, "Missing query field"
        assert "Corpus:" in content, "Missing corpus field"
        assert "Min Score:" in content, "Missing min score field"
        assert "Total Chunks:" in content, "Missing total chunks field"
        assert "Score Range:" in content, "Missing score range field"
        assert "Date:" in content, "Missing date field"

        # Check source sections
        assert "SOURCE:" in content, "Missing source section"

        # Check footer
        assert "END OF EXTRACTION" in content, "Missing footer"

        print("✅ PASSED: Output format is correct")

        # Cleanup
        delete_corpus(corpus_name)

    finally:
        shutil.rmtree(tmpdir)


def test_6_verify_chunk_order_preservation():
    """
    Test 6: Verify chunks are in document order.
    """
    print("\n" + "=" * 80)
    print("TEST 6: Verify chunk order preservation")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_6.txt"

    try:
        test_file, corpus_name = create_test_corpus(tmpdir)

        # Extract content
        extract_relevant_content(
            query="authentication OAuth testing neural",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.2,
            verbose=False,
        )

        # Read output
        content = output_file.read_text()

        # Verify document order: these topics should appear in this sequence
        auth_pos = content.find("Username/Password")
        oauth_pos = content.find("OAuth2")
        testing_pos = content.find("Testing Methodologies")
        neural_pos = content.find("Neural Networks")

        # If all found, check order (allowing for some not being found)
        found_positions = [
            (pos, name)
            for pos, name in [
                (auth_pos, "Authentication"),
                (oauth_pos, "OAuth"),
                (testing_pos, "Testing"),
                (neural_pos, "Neural"),
            ]
            if pos != -1
        ]

        if len(found_positions) >= 2:
            # Check that found items are in increasing position order
            positions = [pos for pos, _ in found_positions]
            assert positions == sorted(positions), "Content should maintain document order"
            print("✅ PASSED: Document order preserved")
            print(f"   Found sections in order: {[name for _, name in found_positions]}")
        else:
            print("⚠️  WARNING: Not enough sections found to verify order")
            print(f"   Found: {len(found_positions)} sections")

        # Cleanup
        delete_corpus(corpus_name)

    finally:
        shutil.rmtree(tmpdir)


def test_7_handle_missing_source_files():
    """
    Test 7: Handle missing source files gracefully.
    """
    print("\n" + "=" * 80)
    print("TEST 7: Handle missing source files gracefully")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_7.txt"

    try:
        test_file, corpus_name = create_test_corpus(tmpdir)

        # Delete the source file to simulate missing file
        os.remove(test_file)

        # Extract should still work but warn about missing source
        extract_relevant_content(
            query="authentication",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.3,
            verbose=True,
        )

        # Should complete without crashing
        assert output_file.exists(), "Output file should still be created"

        print("✅ PASSED: Handles missing source files gracefully")

        # Cleanup
        delete_corpus(corpus_name)

    finally:
        shutil.rmtree(tmpdir)


def test_8_empty_results():
    """
    Test 8: Handle empty results (no chunks above threshold).
    """
    print("\n" + "=" * 80)
    print("TEST 8: Handle empty results")
    print("=" * 80)

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "test_output_8.txt"

    try:
        test_file, corpus_name = create_test_corpus(tmpdir)

        # Query for something completely unrelated with high threshold
        result = extract_relevant_content(
            query="quantum physics spacetime continuum",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.9,  # Very high threshold
            verbose=True,
        )

        # Should handle gracefully
        assert result["total_chunks"] == 0, "Should find no chunks"
        assert result["chunks_extracted"] == 0, "Should extract no chunks"

        print("✅ PASSED: Handles empty results gracefully")

        # Cleanup
        delete_corpus(corpus_name)

    finally:
        shutil.rmtree(tmpdir)


def run_all_tests():
    """
    Run all test cases.
    """
    print("\n" + "=" * 80)
    print("SEAR EXTRACTION FEATURE TEST SUITE")
    print("=" * 80)

    tests = [
        test_1_extract_with_default_threshold,
        test_2_extract_with_custom_threshold,
        test_3_extract_with_max_chunks_limit,
        test_4_extract_from_multiple_corpuses,
        test_5_verify_output_format,
        test_6_verify_chunk_order_preservation,
        test_7_handle_missing_source_files,
        test_8_empty_results,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
