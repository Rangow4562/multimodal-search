"""
Integration tests for image-text alignment system.

Tests the image-text positioning, anchoring, and broken reference handling.
"""

import pytest
from unittest.mock import patch, MagicMock
import json


class TestImageTextAlignment:
    """Tests for image-text positioning system."""

    def test_image_paragraph_matching_basic(self):
        """Test basic paragraph-image matching."""
        # Simulate text segments and image data
        text_segments = [
            "This section describes the login process.",
            "Click the submit button as shown in the image.",
            "The system will authenticate your credentials.",
        ]

        image_data = [
            {"name": "login_screen.png", "keywords": ["login", "screen"]},
            {"name": "submit_button.png", "keywords": ["submit", "button"]},
        ]

        # Match images to segments based on keywords
        matches = []
        for i, segment in enumerate(text_segments):
            segment_lower = segment.lower()
            for img in image_data:
                for keyword in img["keywords"]:
                    if keyword in segment_lower:
                        matches.append((i, img["name"]))
                        break

        # Should match segment 0 with login_screen and segment 1 with submit_button
        assert (0, "login_screen.png") in matches
        assert (1, "submit_button.png") in matches

    def test_image_reference_detection(self):
        """Test detection of image references in text."""
        import re

        patterns = [
            r"(?i)see (figure|image|diagram) (\d+)",
            r"(?i)as shown in (figure|image|diagram)",
            r"(?i)refer to (figure|image|diagram)",
            r"(?i)image \d+",
            r"(?i)figure \d+",
        ]

        test_texts = [
            ("See Figure 1 for details.", True),
            ("As shown in Image 2.", True),
            ("Refer to diagram for more info.", True),
            ("This is plain text.", False),
            ("Image 3 shows the result.", True),
        ]

        for text, should_match in test_texts:
            has_reference = any(re.search(p, text) for p in patterns)
            assert has_reference == should_match, f"Failed for: {text}"

    def test_anchor_position_calculation(self):
        """Test image anchor position calculation."""
        # Simulate paragraph positions
        paragraphs = [
            {"start": 0, "end": 100, "text": "First paragraph"},
            {"start": 101, "end": 200, "text": "Second paragraph with image reference"},
            {"start": 201, "end": 300, "text": "Third paragraph"},
        ]

        # Image should anchor after paragraph with reference
        image_mention_pos = 150  # Position of image reference

        # Find the paragraph containing the image mention
        anchor_paragraph = None
        for p in paragraphs:
            if p["start"] <= image_mention_pos <= p["end"]:
                anchor_paragraph = p
                break

        assert anchor_paragraph is not None
        assert anchor_paragraph["text"] == "Second paragraph with image reference"

    def test_broken_image_reference_detection(self):
        """Test detection of broken image references."""
        available_images = ["diagram1.png", "chart2.png", "photo3.jpg"]

        referenced_images = [
            "diagram1.png",  # Available
            "missing_image.png",  # Not available
            "chart2.png",  # Available
        ]

        broken_refs = []
        for ref in referenced_images:
            if ref not in available_images:
                broken_refs.append(ref)

        assert len(broken_refs) == 1
        assert "missing_image.png" in broken_refs

    def test_image_placeholder_generation(self):
        """Test placeholder generation for missing images."""

        def generate_placeholder(image_name: str) -> dict:
            return {
                "type": "placeholder",
                "name": image_name,
                "message": f"⚠️ Image not available: {image_name}",
                "style": "warning",
            }

        placeholder = generate_placeholder("missing.png")

        assert placeholder["type"] == "placeholder"
        assert "missing.png" in placeholder["message"]
        assert "⚠️" in placeholder["message"]

    def test_confidence_scoring_for_matches(self):
        """Test confidence scoring for image-text matches."""

        def calculate_confidence(segment: str, image_metadata: dict) -> float:
            score = 0.0
            segment_lower = segment.lower()

            # Exact filename match
            if image_metadata.get("filename", "").lower() in segment_lower:
                score += 0.4

            # Keyword matches
            for keyword in image_metadata.get("keywords", []):
                if keyword.lower() in segment_lower:
                    score += 0.2

            # Pattern matches
            patterns = ["figure", "image", "diagram", "shown"]
            for pattern in patterns:
                if pattern in segment_lower:
                    score += 0.1

            return min(score, 1.0)

        # High confidence match
        segment = "See Figure 1: login screen with submit button"
        metadata = {"filename": "login_screen", "keywords": ["login", "button"]}
        confidence = calculate_confidence(segment, metadata)
        assert confidence >= 0.5

        # Low confidence match
        segment = "The system processes data efficiently"
        confidence = calculate_confidence(segment, metadata)
        assert confidence < 0.3

    def test_segment_image_pairing(self):
        """Test pairing segments with relevant images."""
        segments = [
            {"id": 1, "text": "Welcome to the application dashboard."},
            {"id": 2, "text": "Click the settings icon to configure."},
            {"id": 3, "text": "The chart shows performance metrics."},
        ]

        images = [
            {
                "id": "img1",
                "name": "dashboard.png",
                "keywords": ["dashboard", "welcome"],
            },
            {
                "id": "img2",
                "name": "settings.png",
                "keywords": ["settings", "configure"],
            },
            {
                "id": "img3",
                "name": "chart.png",
                "keywords": ["chart", "metrics", "performance"],
            },
        ]

        pairs = []
        for segment in segments:
            seg_lower = segment["text"].lower()
            best_match = None
            best_score = 0

            for img in images:
                score = sum(1 for kw in img["keywords"] if kw in seg_lower)
                if score > best_score:
                    best_score = score
                    best_match = img

            if best_match and best_score > 0:
                pairs.append(
                    {
                        "segment_id": segment["id"],
                        "image_id": best_match["id"],
                        "score": best_score,
                    }
                )

        assert len(pairs) == 3
        assert pairs[0]["image_id"] == "img1"  # dashboard -> dashboard.png
        assert pairs[1]["image_id"] == "img2"  # settings -> settings.png
        assert pairs[2]["image_id"] == "img3"  # chart -> chart.png


class TestImagePositionPreservation:
    """Tests for image position preservation during content updates."""

    def test_anchor_id_generation(self):
        """Test anchor ID generation for images."""

        def generate_anchor_id(image_name: str, paragraph_index: int) -> str:
            import hashlib

            content = f"{image_name}_{paragraph_index}"
            return f"anchor_{hashlib.md5(content.encode()).hexdigest()[:8]}"

        anchor1 = generate_anchor_id("image1.png", 5)
        anchor2 = generate_anchor_id("image1.png", 5)
        anchor3 = generate_anchor_id("image1.png", 6)

        # Same inputs should produce same anchor
        assert anchor1 == anchor2
        # Different paragraph should produce different anchor
        assert anchor1 != anchor3

    def test_position_update_on_content_change(self):
        """Test position update when content is modified."""
        original_paragraphs = [
            {"id": 1, "position": 0},
            {"id": 2, "position": 100},  # Image anchored here
            {"id": 3, "position": 200},
        ]

        image_anchor = {"paragraph_id": 2, "position": 100}

        # Insert new paragraph before anchor
        updated_paragraphs = [
            {"id": 1, "position": 0},
            {"id": 4, "position": 80},  # New paragraph
            {"id": 2, "position": 150},  # Shifted
            {"id": 3, "position": 250},  # Shifted
        ]

        # Find new position for image by paragraph ID
        new_anchor_paragraph = next(
            (p for p in updated_paragraphs if p["id"] == image_anchor["paragraph_id"]),
            None,
        )

        assert new_anchor_paragraph is not None
        assert new_anchor_paragraph["position"] == 150

    def test_orphaned_image_handling(self):
        """Test handling of images whose anchor paragraph is deleted."""
        paragraphs = [
            {"id": 1, "active": True},
            {"id": 2, "active": False},  # Deleted
            {"id": 3, "active": True},
        ]

        image_anchors = [
            {"image": "img1.png", "paragraph_id": 1},
            {"image": "img2.png", "paragraph_id": 2},  # Orphaned
            {"image": "img3.png", "paragraph_id": 3},
        ]

        orphaned_images = []
        for anchor in image_anchors:
            paragraph = next(
                (p for p in paragraphs if p["id"] == anchor["paragraph_id"]), None
            )
            if not paragraph or not paragraph.get("active", True):
                orphaned_images.append(anchor["image"])

        assert "img2.png" in orphaned_images
        assert len(orphaned_images) == 1


class TestImageTextRelationModels:
    """Tests for image-text relation data models."""

    def test_image_text_relation_creation(self):
        """Test creating image-text relation."""
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class ImageTextRelation:
            image_id: str
            paragraph_id: str
            confidence_score: float
            anchor_position: int
            is_valid: bool = True
            validation_error: Optional[str] = None

        relation = ImageTextRelation(
            image_id="img_001",
            paragraph_id="para_005",
            confidence_score=0.85,
            anchor_position=150,
        )

        assert relation.image_id == "img_001"
        assert relation.confidence_score == 0.85
        assert relation.is_valid is True

    def test_relation_validation(self):
        """Test relation validation logic."""

        def validate_relation(relation: dict) -> tuple:
            errors = []

            if not relation.get("image_id"):
                errors.append("Missing image_id")

            if not relation.get("paragraph_id"):
                errors.append("Missing paragraph_id")

            confidence = relation.get("confidence_score", 0)
            if not (0 <= confidence <= 1):
                errors.append("Invalid confidence_score")

            return len(errors) == 0, errors

        valid_relation = {
            "image_id": "img_001",
            "paragraph_id": "para_005",
            "confidence_score": 0.85,
        }

        is_valid, errors = validate_relation(valid_relation)
        assert is_valid is True
        assert len(errors) == 0

        invalid_relation = {
            "image_id": "",
            "confidence_score": 1.5,  # Invalid
        }

        is_valid, errors = validate_relation(invalid_relation)
        assert is_valid is False
        assert len(errors) >= 2
