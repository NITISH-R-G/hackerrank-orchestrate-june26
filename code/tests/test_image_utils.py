"""Sprint 2 — Image utilities tests (image_utils.py).

Written FIRST (RED) against specs/data_contract.md §2.5 and behavior_spec.md
scenario 5/9 (image IDs, missing images). Run from code/:

    pytest tests/test_image_utils.py -v
"""
import io
import pytest

from PIL import Image

from image_utils import (
    parse_image_paths,
    image_id_from_path,
    load_image,
    load_images,
    image_hash,
)


# ---------- parse_image_paths ----------

class TestParseImagePaths:
    def test_single_path(self):
        assert parse_image_paths("images/test/case_001/img_1.jpg") == [
            "images/test/case_001/img_1.jpg"
        ]

    def test_multiple_paths_semicolon(self):
        s = "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg"
        assert parse_image_paths(s) == [
            "images/test/case_001/img_1.jpg",
            "images/test/case_001/img_2.jpg",
        ]

    def test_strips_whitespace(self):
        s = " images/test/c/img_1.jpg ; images/test/c/img_2.jpg "
        out = parse_image_paths(s)
        assert out == ["images/test/c/img_1.jpg", "images/test/c/img_2.jpg"]

    def test_empty_string_returns_empty_list(self):
        assert parse_image_paths("") == []


# ---------- image_id_from_path ----------

class TestImageIdFromPath:
    def test_jpg_without_extension(self):
        assert image_id_from_path("images/test/case_001/img_1.jpg") == "img_1"

    def test_png(self):
        assert image_id_from_path("a/b/photo.png") == "photo"

    def test_already_just_id(self):
        assert image_id_from_path("img_7") == "img_7"


# ---------- load_image / load_images ----------

class TestLoadImage:
    def test_loads_real_sample_image(self):
        img = load_image("images/sample/case_001/img_1.jpg")
        assert img is not None
        assert isinstance(img, Image.Image)

    def test_missing_path_returns_none(self):
        # Scenario 5: missing/unreadable image -> None, no exception
        assert load_image("images/sample/does_not_exist_xyz.jpg") is None

    def test_load_images_skips_missing(self):
        out = load_images([
            "images/sample/case_001/img_1.jpg",          # exists
            "images/sample/this_is_missing_9999.jpg",    # missing
        ])
        # Returns list of (path, image) for the ones that loaded
        assert len(out) == 1
        assert out[0][0].endswith("img_1.jpg")


# ---------- image_hash (for cache keys) ----------

class TestImageHash:
    def _make_bytes(self, color=(255, 0, 0), size=(8, 8)):
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_same_bytes_same_hash(self):
        b = self._make_bytes()
        assert image_hash(b) == image_hash(b)

    def test_different_bytes_different_hash(self):
        a = self._make_bytes(color=(255, 0, 0))
        b = self._make_bytes(color=(0, 255, 0))
        assert image_hash(a) != image_hash(b)

    def test_hash_is_hex_string(self):
        h = image_hash(self._make_bytes())
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)
