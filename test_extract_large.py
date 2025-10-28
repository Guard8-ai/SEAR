#!/usr/bin/env python3
"""
Large corpus extraction test using gitingest-style content.

Tests extraction on a realistic large corpus with wide-ranging queries
that will match many chunks (50+ chunks expected).
"""

import sys
import tempfile
import shutil
from pathlib import Path
from sear_core import index_file, extract_relevant_content, delete_corpus


def create_large_gitingest_corpus(tmpdir):
    """
    Create a large corpus simulating a gitingest output of a Python web application.
    This creates realistic code/docs that will match many chunks for broad queries.
    """

    # Simulate a large gitingest-style repository dump
    content = """
================================================================================
Repository: web-app-example
Generated: 2025-10-23
Total Files: 45
Total Lines: 8,523
================================================================================

================================================================================
File: src/auth/authentication.py
================================================================================
\"\"\"
Authentication module for user login and session management.
Handles password hashing, JWT tokens, and OAuth integration.
\"\"\"

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

class AuthenticationManager:
    \"\"\"Manages user authentication and session tokens.\"\"\"

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.sessions = {}

    def hash_password(self, password: str) -> str:
        \"\"\"Hash password using SHA-256 with salt.\"\"\"
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
        return f"{salt}:{hashed}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        \"\"\"Verify password against stored hash.\"\"\"
        salt, hashed = stored_hash.split(':')
        new_hash = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
        return new_hash == hashed

    def create_session_token(self, user_id: int) -> str:
        \"\"\"Create a new session token for authenticated user.\"\"\"
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {
            'user_id': user_id,
            'created': datetime.now(),
            'expires': datetime.now() + timedelta(hours=24)
        }
        return token

    def validate_token(self, token: str) -> Optional[int]:
        \"\"\"Validate session token and return user_id if valid.\"\"\"
        session = self.sessions.get(token)
        if not session:
            return None
        if datetime.now() > session['expires']:
            del self.sessions[token]
            return None
        return session['user_id']

================================================================================
File: src/auth/oauth.py
================================================================================
\"\"\"
OAuth2 integration for third-party authentication providers.
Supports Google, GitHub, and Microsoft authentication.
\"\"\"

import requests
from typing import Dict, Optional

class OAuthProvider:
    \"\"\"Base class for OAuth providers.\"\"\"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret

    def get_authorization_url(self) -> str:
        raise NotImplementedError

    def exchange_code_for_token(self, code: str) -> Optional[str]:
        raise NotImplementedError

    def get_user_info(self, access_token: str) -> Optional[Dict]:
        raise NotImplementedError

class GoogleOAuthProvider(OAuthProvider):
    \"\"\"Google OAuth2 provider implementation.\"\"\"

    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

    def get_authorization_url(self) -> str:
        return f"{self.AUTH_URL}?client_id={self.client_id}&redirect_uri=..."

================================================================================
File: src/security/validation.py
================================================================================
\"\"\"
Input validation and security functions.
Prevents SQL injection, XSS, CSRF attacks.
\"\"\"

import re
from typing import Any

class SecurityValidator:
    \"\"\"Security validation utilities.\"\"\"

    @staticmethod
    def sanitize_sql_input(value: str) -> str:
        \"\"\"Sanitize input to prevent SQL injection.\"\"\"
        # Remove SQL keywords and special characters
        dangerous_patterns = [
            r"--", r";", r"'", r'"', r"\\",
            r"DROP", r"DELETE", r"INSERT", r"UPDATE",
            r"UNION", r"SELECT"
        ]
        sanitized = value
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        return sanitized

    @staticmethod
    def sanitize_html(value: str) -> str:
        \"\"\"Sanitize HTML to prevent XSS attacks.\"\"\"
        dangerous_tags = ['script', 'iframe', 'object', 'embed']
        sanitized = value
        for tag in dangerous_tags:
            sanitized = re.sub(f'<{tag}.*?>', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(f'</{tag}>', '', sanitized, flags=re.IGNORECASE)
        return sanitized

    @staticmethod
    def validate_csrf_token(token: str, session_token: str) -> bool:
        \"\"\"Validate CSRF token against session.\"\"\"
        return token == session_token

================================================================================
File: src/api/endpoints.py
================================================================================
\"\"\"
REST API endpoints for the web application.
Handles user management, data CRUD operations.
\"\"\"

from flask import Flask, request, jsonify
from src.auth.authentication import AuthenticationManager
from src.security.validation import SecurityValidator

app = Flask(__name__)
auth_manager = AuthenticationManager(secret_key="your-secret-key")
validator = SecurityValidator()

@app.route('/api/auth/login', methods=['POST'])
def login():
    \"\"\"User login endpoint.\"\"\"
    data = request.json
    username = validator.sanitize_sql_input(data.get('username', ''))
    password = data.get('password', '')

    # Authenticate user
    user = authenticate_user(username, password)
    if user:
        token = auth_manager.create_session_token(user.id)
        return jsonify({'token': token, 'user_id': user.id})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    \"\"\"User logout endpoint.\"\"\"
    token = request.headers.get('Authorization')
    # Invalidate session
    return jsonify({'message': 'Logged out successfully'})

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    \"\"\"Get user information.\"\"\"
    token = request.headers.get('Authorization')
    user_id_from_token = auth_manager.validate_token(token)
    if not user_id_from_token:
        return jsonify({'error': 'Unauthorized'}), 401
    # Return user data
    return jsonify({'user_id': user_id, 'username': 'example'})

================================================================================
File: tests/test_authentication.py
================================================================================
\"\"\"
Unit tests for authentication module.
Tests password hashing, token generation, and session management.
\"\"\"

import unittest
from src.auth.authentication import AuthenticationManager

class TestAuthentication(unittest.TestCase):
    \"\"\"Test cases for authentication functionality.\"\"\"

    def setUp(self):
        self.auth = AuthenticationManager(secret_key="test-secret")

    def test_password_hashing(self):
        \"\"\"Test that password hashing works correctly.\"\"\"
        password = "SecurePass123!"
        hashed = self.auth.hash_password(password)

        # Verify password
        self.assertTrue(self.auth.verify_password(password, hashed))
        self.assertFalse(self.auth.verify_password("WrongPass", hashed))

    def test_token_generation(self):
        \"\"\"Test session token generation and validation.\"\"\"
        user_id = 123
        token = self.auth.create_session_token(user_id)

        # Validate token
        self.assertEqual(self.auth.validate_token(token), user_id)
        self.assertIsNone(self.auth.validate_token("invalid-token"))

    def test_token_expiration(self):
        \"\"\"Test that expired tokens are invalidated.\"\"\"
        # This would test token expiration logic
        pass

================================================================================
File: tests/test_security.py
================================================================================
\"\"\"
Security validation tests.
Tests SQL injection prevention, XSS sanitization, CSRF validation.
\"\"\"

import unittest
from src.security.validation import SecurityValidator

class TestSecurityValidation(unittest.TestCase):
    \"\"\"Test cases for security validation.\"\"\"

    def test_sql_injection_prevention(self):
        \"\"\"Test SQL injection attack prevention.\"\"\"
        malicious_input = "admin' OR '1'='1"
        sanitized = SecurityValidator.sanitize_sql_input(malicious_input)

        # Should remove SQL keywords
        self.assertNotIn("OR", sanitized)
        self.assertNotIn("'", sanitized)

    def test_xss_prevention(self):
        \"\"\"Test XSS attack prevention through HTML sanitization.\"\"\"
        malicious_html = "<script>alert('XSS')</script>Hello"
        sanitized = SecurityValidator.sanitize_html(malicious_html)

        # Should remove script tags
        self.assertNotIn("<script>", sanitized)
        self.assertIn("Hello", sanitized)

    def test_csrf_validation(self):
        \"\"\"Test CSRF token validation.\"\"\"
        session_token = "valid-csrf-token"

        # Valid token should pass
        self.assertTrue(SecurityValidator.validate_csrf_token(
            session_token, session_token))

        # Invalid token should fail
        self.assertFalse(SecurityValidator.validate_csrf_token(
            "wrong-token", session_token))

================================================================================
File: tests/test_integration.py
================================================================================
\"\"\"
Integration tests for authentication flows.
Tests end-to-end user registration, login, and session management.
\"\"\"

import unittest
from src.api.endpoints import app
from src.auth.authentication import AuthenticationManager

class TestAuthenticationIntegration(unittest.TestCase):
    \"\"\"Integration test cases for authentication flows.\"\"\"

    def setUp(self):
        self.app = app.test_client()
        self.auth = AuthenticationManager(secret_key="test-secret")

    def test_user_login_flow(self):
        \"\"\"Test complete user login workflow.\"\"\"
        # Attempt login
        response = self.app.post('/api/auth/login', json={
            'username': 'testuser',
            'password': 'TestPass123!'
        })

        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn('token', data)
        self.assertIn('user_id', data)

    def test_protected_endpoint_access(self):
        \"\"\"Test accessing protected endpoints with valid token.\"\"\"
        # Login first
        login_response = self.app.post('/api/auth/login', json={
            'username': 'testuser',
            'password': 'TestPass123!'
        })
        token = login_response.json['token']

        # Access protected endpoint
        response = self.app.get('/api/users/123', headers={
            'Authorization': token
        })

        self.assertEqual(response.status_code, 200)

    def test_logout_flow(self):
        \"\"\"Test user logout invalidates session.\"\"\"
        # Login and get token
        login_response = self.app.post('/api/auth/login', json={
            'username': 'testuser',
            'password': 'TestPass123!'
        })
        token = login_response.json['token']

        # Logout
        logout_response = self.app.post('/api/auth/logout', headers={
            'Authorization': token
        })

        self.assertEqual(logout_response.status_code, 200)

================================================================================
File: src/ml/neural_network.py
================================================================================
\"\"\"
Neural network model for fraud detection.
Uses transformer architecture with attention mechanisms.
\"\"\"

import numpy as np
from typing import List, Tuple

class FraudDetectionModel:
    \"\"\"Neural network model for detecting fraudulent login attempts.\"\"\"

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, output_dim))

    def attention_mechanism(self, query, key, value):
        \"\"\"Compute attention weights and apply to values.\"\"\"
        # Scaled dot-product attention
        scores = np.dot(query, key.T) / np.sqrt(key.shape[-1])
        attention_weights = self.softmax(scores)
        output = np.dot(attention_weights, value)
        return output, attention_weights

    def softmax(self, x):
        \"\"\"Compute softmax activation.\"\"\"
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, X):
        \"\"\"Forward pass through the network.\"\"\"
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def predict_fraud_score(self, login_features: List[float]) -> float:
        \"\"\"Predict fraud probability for a login attempt.\"\"\"
        X = np.array(login_features).reshape(1, -1)
        prediction = self.forward(X)
        return float(prediction[0][1])  # Probability of fraud

================================================================================
File: src/ml/transformer.py
================================================================================
\"\"\"
Transformer architecture for login pattern analysis.
Implements multi-head self-attention and feed-forward layers.
\"\"\"

import numpy as np

class TransformerBlock:
    \"\"\"Single transformer block with self-attention and FFN.\"\"\"

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Multi-head attention parameters
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01

        # Feed-forward network
        self.W_ff1 = np.random.randn(d_model, d_ff) * 0.01
        self.W_ff2 = np.random.randn(d_ff, d_model) * 0.01

    def multi_head_attention(self, x):
        \"\"\"Compute multi-head self-attention.\"\"\"
        # Split into multiple heads
        batch_size = x.shape[0]
        head_dim = self.d_model // self.num_heads

        # Compute Q, K, V
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Scaled dot-product attention
        scores = np.dot(Q, K.T) / np.sqrt(head_dim)
        attention = self.softmax(scores)
        output = np.dot(attention, V)

        return output

    def feed_forward(self, x):
        \"\"\"Feed-forward network with ReLU activation.\"\"\"
        hidden = np.maximum(0, np.dot(x, self.W_ff1))
        output = np.dot(hidden, self.W_ff2)
        return output

    def forward(self, x):
        \"\"\"Forward pass through transformer block.\"\"\"
        # Self-attention with residual connection
        attn_output = self.multi_head_attention(x)
        x = x + attn_output

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output

        return x

    @staticmethod
    def softmax(x):
        \"\"\"Compute softmax activation.\"\"\"
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

================================================================================
File: docs/API_DOCUMENTATION.md
================================================================================
# API Documentation

## Authentication Endpoints

### POST /api/auth/login
Authenticate user and receive session token.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "token": "string",
  "user_id": "integer"
}
```

### POST /api/auth/logout
Logout and invalidate session token.

**Headers:**
- Authorization: Bearer token

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

## Security Features

### Input Validation
All inputs are validated and sanitized:
- SQL injection prevention
- XSS attack prevention
- CSRF token validation

### Session Management
- Tokens expire after 24 hours
- Secure session storage
- Automatic cleanup of expired sessions

## Testing Guidelines

### Unit Tests
Run unit tests with:
```bash
python -m pytest tests/test_authentication.py
python -m pytest tests/test_security.py
```

### Integration Tests
Run integration tests with:
```bash
python -m pytest tests/test_integration.py
```

### Security Testing
Perform security testing:
- Penetration testing for vulnerabilities
- Rate limiting verification
- Brute force attack prevention

## Machine Learning Models

### Fraud Detection Neural Network
The neural network model analyzes login patterns:
- Input: Login features (IP, time, location, etc.)
- Output: Fraud probability score (0-1)
- Architecture: 2-layer neural network with attention

### Transformer Model
Transformer-based architecture for pattern analysis:
- Multi-head self-attention mechanism
- Feed-forward layers with ReLU
- Residual connections
- 99.2% accuracy in production

================================================================================
End of Repository Dump
================================================================================
"""

    # Write to file
    test_file = Path(tmpdir) / "large_repo_dump.txt"
    test_file.write_text(content)

    return str(test_file)


def test_large_corpus_extraction():
    """
    Test extraction on large corpus with wide query matching many chunks.
    """
    print("\n" + "="*80)
    print("LARGE CORPUS EXTRACTION TEST")
    print("="*80)
    print("\nCreating large gitingest-style corpus...")

    tmpdir = tempfile.mkdtemp()
    output_file = Path(tmpdir) / "large_extraction.txt"

    try:
        # Create corpus
        test_file = create_large_gitingest_corpus(tmpdir)
        corpus_name = "large_test_corpus"

        print(f"Indexing large corpus '{corpus_name}'...")
        chunk_count = index_file(test_file, corpus_name=corpus_name, verbose=True)
        print(f"✅ Indexed {chunk_count} chunks")

        # Test 1: Wide query that should match many chunks
        print("\n" + "="*80)
        print("TEST 1: Wide query - authentication security testing validation")
        print("="*80)

        result1 = extract_relevant_content(
            query="authentication security testing validation neural network transformer",
            corpuses=[corpus_name],
            output_file=str(output_file),
            min_score=0.25,  # Lower threshold to get more results
            verbose=True
        )

        print(f"\n📊 RESULTS:")
        print(f"   Total chunks extracted: {result1['total_chunks']}")
        print(f"   Score range: {result1['score_range'][0]:.3f} - {result1['score_range'][1]:.3f}")
        print(f"   Output file size: {output_file.stat().st_size:,} bytes")
        print(f"   Sources: {len(result1['sources'])}")

        # Validate results
        assert result1['total_chunks'] >= 10, \
            f"Should extract at least 10 chunks, got {result1['total_chunks']}"
        assert output_file.exists(), "Output file should be created"

        # Check output content
        content = output_file.read_text()
        assert len(content) > 1000, "Output should have substantial content"
        assert "authentication" in content.lower(), "Should contain authentication content"
        assert "security" in content.lower(), "Should contain security content"
        assert "testing" in content.lower(), "Should contain testing content"

        print(f"✅ PASSED: Extracted {result1['total_chunks']} chunks successfully")

        # Test 2: Specific query with higher threshold
        print("\n" + "="*80)
        print("TEST 2: Specific query - OAuth Google authentication")
        print("="*80)

        output_file2 = Path(tmpdir) / "specific_extraction.txt"
        result2 = extract_relevant_content(
            query="OAuth Google authentication provider",
            corpuses=[corpus_name],
            output_file=str(output_file2),
            min_score=0.35,
            verbose=True
        )

        print(f"\n📊 RESULTS:")
        print(f"   Total chunks extracted: {result2['total_chunks']}")
        print(f"   Score range: {result2['score_range'][0]:.3f} - {result2['score_range'][1]:.3f}" if result2['total_chunks'] > 0 else "   No chunks found")

        # Validate
        assert result2['total_chunks'] >= 1, "Should find at least one OAuth-related chunk"

        content2 = output_file2.read_text()
        assert "OAuth" in content2 or "oauth" in content2, "Should contain OAuth content"

        print(f"✅ PASSED: Specific query worked correctly")

        # Test 3: Max chunks limit on large result set
        print("\n" + "="*80)
        print("TEST 3: Max chunks limit - limit to 15 chunks")
        print("="*80)

        output_file3 = Path(tmpdir) / "limited_extraction.txt"
        result3 = extract_relevant_content(
            query="authentication security testing neural network API endpoint",
            corpuses=[corpus_name],
            output_file=str(output_file3),
            min_score=0.2,
            max_chunks=15,
            verbose=True
        )

        print(f"\n📊 RESULTS:")
        print(f"   Total chunks extracted: {result3['total_chunks']}")
        print(f"   Max chunks requested: 15")

        # Validate
        assert result3['total_chunks'] <= 15, \
            f"Should not exceed 15 chunks, got {result3['total_chunks']}"

        print(f"✅ PASSED: Max chunks limit respected")

        # Test 4: Performance check
        print("\n" + "="*80)
        print("TEST 4: Performance - extract all relevant chunks")
        print("="*80)

        import time
        output_file4 = Path(tmpdir) / "performance_extraction.txt"

        start_time = time.time()
        result4 = extract_relevant_content(
            query="authentication security validation testing integration unit",
            corpuses=[corpus_name],
            output_file=str(output_file4),
            min_score=0.2,
            verbose=True
        )
        elapsed = time.time() - start_time

        print(f"\n📊 PERFORMANCE:")
        print(f"   Total chunks: {result4['total_chunks']}")
        print(f"   Time elapsed: {elapsed:.2f} seconds")
        print(f"   Chunks per second: {result4['total_chunks'] / elapsed:.1f}")

        # Performance should be reasonable (< 10 seconds for < 100 chunks)
        if result4['total_chunks'] < 100:
            assert elapsed < 10, f"Performance too slow: {elapsed:.2f}s for {result4['total_chunks']} chunks"

        print(f"✅ PASSED: Performance acceptable")

        # Summary
        print("\n" + "="*80)
        print("LARGE CORPUS TEST SUMMARY")
        print("="*80)
        print(f"✅ All large corpus tests passed!")
        print(f"   Corpus size: {chunk_count} chunks")
        print(f"   Wide query extracted: {result1['total_chunks']} chunks")
        print(f"   Specific query extracted: {result2['total_chunks']} chunks")
        print(f"   Limited extraction: {result3['total_chunks']} chunks (max: 15)")
        print(f"   Performance test: {result4['total_chunks']} chunks in {elapsed:.2f}s")
        print("="*80)

        # Cleanup
        delete_corpus(corpus_name)

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    success = test_large_corpus_extraction()
    sys.exit(0 if success else 1)
