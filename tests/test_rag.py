"""
Test suite for the RAG chatbot functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from app.rag_ollama import RAGChatbot
from app.config import STORAGE_CONFIG

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return str(tmp_path / "test_data")

@pytest.fixture
def chatbot(test_data_dir):
    """Create a test instance of RAGChatbot."""
    os.makedirs(test_data_dir, exist_ok=True)
    return RAGChatbot(data_dir=test_data_dir)

def test_chat_initialization(chatbot):
    """Test chatbot initialization."""
    assert chatbot.data_dir is not None
    assert chatbot.collection is not None

@patch('requests.post')
def test_query_with_context(mock_post, chatbot):
    """Test query method with context."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"response": "Test answer"}
    mock_post.return_value = mock_response

    response = chatbot.query("test question")
    assert response == "Test answer"
    assert mock_post.called

@patch('requests.post')
def test_query_error_handling(mock_post, chatbot):
    """Test query method error handling."""
    mock_post.side_effect = Exception("API Error")
    response = chatbot.query("test question")
    assert "Error" in response

def test_build_index_empty_dir(chatbot, test_data_dir):
    """Test build_index with empty directory."""
    chatbot.build_index()
    # Should not raise any errors for empty directory

def test_build_index_with_file(chatbot, test_data_dir):
    """Test build_index with a test file."""
    test_file = os.path.join(test_data_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Test content for indexing")
    
    chatbot.build_index()
    # Verify the collection has been updated
