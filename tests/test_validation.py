"""
Unit tests for the validation module.
"""

import pytest
import os
from core.validation import (
    validate_query,
    validate_top_k,
    validate_file_path,
    validate_url,
    validate_blob_name,
    sanitize_html,
    validate_json_string,
)
from core.exceptions import ValidationError


class TestValidateQuery:
    """Tests for validate_query function."""

    def test_valid_query(self):
        """Test valid query passes validation."""
        result = validate_query("What is machine learning?")
        assert result == "What is machine learning?"

    def test_query_strips_whitespace(self):
        """Test query whitespace is stripped."""
        result = validate_query("  hello world  ")
        assert result == "hello world"

    def test_empty_query_raises(self):
        """Test empty query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query("")
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_raises(self):
        """Test whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_query("   ")

    def test_none_query_raises(self):
        """Test None query raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_query(None)

    def test_non_string_raises(self):
        """Test non-string query raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_query(123)

    def test_max_length_exceeded(self):
        """Test query exceeding max length raises."""
        long_query = "a" * 10001
        with pytest.raises(ValidationError) as exc_info:
            validate_query(long_query)
        assert "maximum length" in str(exc_info.value).lower()

    def test_custom_max_length(self):
        """Test custom max length."""
        with pytest.raises(ValidationError):
            validate_query("hello", max_length=3)


class TestValidateTopK:
    """Tests for validate_top_k function."""

    def test_valid_top_k(self):
        """Test valid top_k values."""
        assert validate_top_k(5) == 5
        assert validate_top_k(1) == 1
        assert validate_top_k(100) == 100

    def test_string_conversion(self):
        """Test string to int conversion."""
        assert validate_top_k("10") == 10

    def test_below_min_raises(self):
        """Test value below minimum raises."""
        with pytest.raises(ValidationError):
            validate_top_k(0)

    def test_above_max_raises(self):
        """Test value above maximum raises."""
        with pytest.raises(ValidationError):
            validate_top_k(101)

    def test_non_integer_raises(self):
        """Test non-integer value raises."""
        with pytest.raises(ValidationError):
            validate_top_k("not_a_number")

    def test_custom_range(self):
        """Test custom min/max range."""
        assert validate_top_k(5, min_val=1, max_val=10) == 5

        with pytest.raises(ValidationError):
            validate_top_k(15, min_val=1, max_val=10)


class TestValidateFilePath:
    """Tests for validate_file_path function."""

    def test_valid_path(self, temp_test_dir):
        """Test valid file path."""
        file_path = temp_test_dir / "test.txt"
        file_path.write_text("test")

        result = validate_file_path(str(file_path), must_exist=True)
        assert result is not None

    def test_empty_path_raises(self):
        """Test empty path raises."""
        with pytest.raises(ValidationError):
            validate_file_path("")

    def test_path_traversal_blocked(self):
        """Test path traversal is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path("../../etc/passwd")
        assert "traversal" in str(exc_info.value).lower()

    def test_non_existent_file(self, temp_test_dir):
        """Test non-existent file with must_exist=True."""
        with pytest.raises(ValidationError):
            validate_file_path(str(temp_test_dir / "nonexistent.txt"), must_exist=True)

    def test_path_length_limit(self):
        """Test path length limit."""
        long_path = "a" * 501
        with pytest.raises(ValidationError):
            validate_file_path(long_path)


class TestValidateUrl:
    """Tests for validate_url function."""

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        url = "https://example.com/path"
        result = validate_url(url)
        assert result == url

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        url = "http://example.com"
        result = validate_url(url)
        assert result == url

    def test_empty_url_raises(self):
        """Test empty URL raises."""
        with pytest.raises(ValidationError):
            validate_url("")

    def test_invalid_scheme_raises(self):
        """Test invalid URL scheme raises."""
        with pytest.raises(ValidationError):
            validate_url("ftp://example.com")

    def test_custom_allowed_schemes(self):
        """Test custom allowed schemes."""
        url = "ftp://example.com"
        result = validate_url(url, allowed_schemes=["ftp", "sftp"])
        assert result == url

    def test_missing_host_raises(self):
        """Test URL without host raises."""
        with pytest.raises(ValidationError):
            validate_url("https://")


class TestValidateBlobName:
    """Tests for validate_blob_name function."""

    def test_valid_blob_name(self):
        """Test valid blob name."""
        result = validate_blob_name("image.png")
        assert result == "image.png"

    def test_blob_with_path(self):
        """Test blob name with path."""
        result = validate_blob_name("folder/subfolder/image.png")
        assert result == "folder/subfolder/image.png"

    def test_empty_blob_name_raises(self):
        """Test empty blob name raises."""
        with pytest.raises(ValidationError):
            validate_blob_name("")

    def test_invalid_characters_raise(self):
        """Test invalid characters raise."""
        with pytest.raises(ValidationError):
            validate_blob_name("image:test.png")

        with pytest.raises(ValidationError):
            validate_blob_name("image*.png")


class TestSanitizeHtml:
    """Tests for sanitize_html function."""

    def test_removes_script_tags(self):
        """Test script tags are removed."""
        html = '<script>alert("xss")</script>Hello'
        result = sanitize_html(html)
        assert "script" not in result.lower()
        assert "alert" not in result

    def test_removes_event_handlers(self):
        """Test event handlers are removed."""
        html = '<div onclick="evil()">Click me</div>'
        result = sanitize_html(html)
        assert "onclick" not in result.lower()

    def test_escapes_html_entities(self):
        """Test HTML entities are escaped."""
        html = "<div>Test & more</div>"
        result = sanitize_html(html)
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_empty_string(self):
        """Test empty string returns empty."""
        result = sanitize_html("")
        assert result == ""


class TestValidateJsonString:
    """Tests for validate_json_string function."""

    def test_valid_json(self):
        """Test valid JSON passes."""
        json_str = '{"key": "value", "number": 123}'
        result = validate_json_string(json_str)
        assert result == json_str

    def test_valid_json_array(self):
        """Test valid JSON array passes."""
        json_str = '[1, 2, 3, "four"]'
        result = validate_json_string(json_str)
        assert result == json_str

    def test_empty_string_passes(self):
        """Test empty string passes."""
        result = validate_json_string("")
        assert result == ""

    def test_invalid_json_raises(self):
        """Test invalid JSON raises."""
        with pytest.raises(ValidationError) as exc_info:
            validate_json_string("{invalid json}")
        assert "Invalid JSON" in str(exc_info.value)
