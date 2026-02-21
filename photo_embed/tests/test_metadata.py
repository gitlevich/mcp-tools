"""Tests for metadata text extraction."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL.ExifTags import GPS

from metadata import (
    _dms_to_decimal,
    _extract_exif,
    _format_date,
    _parse_filename,
    _parse_folder_path,
    _parse_gps_ifd,
    _reverse_geocode,
    extract_metadata,
)


# -- _parse_filename --

def test_parse_filename_strips_img_prefix():
    assert _parse_filename(Path("IMG_1234.jpg")) == ""

def test_parse_filename_strips_dsc_prefix():
    assert _parse_filename(Path("DSC_5678.jpg")) == ""

def test_parse_filename_strips_pxl_prefix():
    assert _parse_filename(Path("PXL_20230101.jpg")) == ""

def test_parse_filename_strips_screenshot_prefix():
    assert _parse_filename(Path("Screenshot_note.png")) == "note"

def test_parse_filename_meaningful_name():
    assert _parse_filename(Path("beach_sunset_2023.jpg")) == "beach sunset 2023"

def test_parse_filename_pure_numeric():
    assert _parse_filename(Path("123456.jpg")) == ""

def test_parse_filename_mixed():
    assert _parse_filename(Path("IMG_birthday-party.jpg")) == "birthday party"


# -- _parse_folder_path --

def test_parse_folder_ignores_generic():
    path = Path("/Users/home/Pictures/photos/vacation/beach/img.jpg")
    result = _parse_folder_path(path)
    assert "vacation" in result
    assert "beach" in result
    assert "photos" not in result.lower()
    assert "pictures" not in result.lower()

def test_parse_folder_keeps_year():
    path = Path("/photos/2023/summer/img.jpg")
    result = _parse_folder_path(path)
    assert "2023" in result
    assert "summer" in result

def test_parse_folder_empty_for_generic_only():
    path = Path("/Users/Pictures/photos/img.jpg")
    result = _parse_folder_path(path)
    assert result.strip() == ""


# -- _format_date --

def test_format_date_standard():
    assert _format_date("2023:07:15 14:30:00") == "July 2023"

def test_format_date_january():
    assert _format_date("2024:01:01 00:00:00") == "January 2024"

def test_format_date_invalid():
    assert _format_date("not a date") == ""

def test_format_date_empty():
    assert _format_date("") == ""


# -- _dms_to_decimal --

def test_dms_to_decimal_north():
    # 48 degrees, 51 minutes, 24 seconds N = 48.8567
    result = _dms_to_decimal((48, 51, 24.0), "N")
    assert abs(result - 48.8567) < 0.001

def test_dms_to_decimal_south():
    result = _dms_to_decimal((33, 52, 10.0), "S")
    assert result < 0
    assert abs(result - (-33.8694)) < 0.001

def test_dms_to_decimal_east():
    result = _dms_to_decimal((2, 21, 7.0), "E")
    assert abs(result - 2.3519) < 0.001

def test_dms_to_decimal_west():
    result = _dms_to_decimal((118, 14, 34.0), "W")
    assert result < 0
    assert abs(result - (-118.2428)) < 0.001


# -- _parse_gps_ifd --

def test_parse_gps_ifd_valid():
    gps_ifd = {
        GPS.GPSLatitude: (48, 51, 24.0),
        GPS.GPSLatitudeRef: "N",
        GPS.GPSLongitude: (2, 21, 7.0),
        GPS.GPSLongitudeRef: "E",
    }
    result = _parse_gps_ifd(gps_ifd)
    assert result is not None
    lat, lon = result
    assert abs(lat - 48.8567) < 0.001
    assert abs(lon - 2.3519) < 0.001

def test_parse_gps_ifd_null_island():
    gps_ifd = {
        GPS.GPSLatitude: (0, 0, 0.0),
        GPS.GPSLatitudeRef: "N",
        GPS.GPSLongitude: (0, 0, 0.0),
        GPS.GPSLongitudeRef: "E",
    }
    assert _parse_gps_ifd(gps_ifd) is None

def test_parse_gps_ifd_missing_fields():
    assert _parse_gps_ifd({}) is None
    assert _parse_gps_ifd({GPS.GPSLatitude: (48, 51, 24.0)}) is None

def test_parse_gps_ifd_invalid_range():
    gps_ifd = {
        GPS.GPSLatitude: (200, 0, 0.0),
        GPS.GPSLatitudeRef: "N",
        GPS.GPSLongitude: (0, 0, 0.0),
        GPS.GPSLongitudeRef: "E",
    }
    assert _parse_gps_ifd(gps_ifd) is None

def test_parse_gps_ifd_corrupt_values():
    gps_ifd = {
        GPS.GPSLatitude: "bad",
        GPS.GPSLatitudeRef: "N",
        GPS.GPSLongitude: (2, 21, 7.0),
        GPS.GPSLongitudeRef: "E",
    }
    assert _parse_gps_ifd(gps_ifd) is None


# -- _reverse_geocode --

def test_reverse_geocode_known_location():
    result = _reverse_geocode(48.8566, 2.3522)
    assert "Paris" in result
    assert "France" in result

def test_reverse_geocode_returns_string():
    result = _reverse_geocode(-33.8688, 151.2093)
    assert isinstance(result, str)
    assert len(result) > 0


# -- _extract_exif --

def test_extract_exif_no_exif(tmp_path):
    """PIL-generated images have no EXIF — should return empty text and no coords."""
    from PIL import Image
    img_path = tmp_path / "plain.jpg"
    Image.new("RGB", (10, 10)).save(img_path)
    text, coords, _ = _extract_exif(img_path)
    assert text == ""
    assert coords is None

def test_extract_exif_nonexistent():
    text, coords, _ = _extract_exif(Path("/nonexistent/image.jpg"))
    assert text == ""
    assert coords is None


# -- extract_metadata (integration) --

def test_extract_metadata_meaningful_filename(tmp_path):
    from PIL import Image
    img_path = tmp_path / "birthday_party.jpg"
    Image.new("RGB", (10, 10)).save(img_path)
    text, lat, lon, _ = extract_metadata(img_path)
    assert "birthday party" in text

def test_extract_metadata_generic_filename(tmp_path):
    from PIL import Image
    img_path = tmp_path / "IMG_1234.jpg"
    Image.new("RGB", (10, 10)).save(img_path)
    text, lat, lon, _ = extract_metadata(img_path)
    # Nothing meaningful from filename or EXIF
    # May still have folder path components
    assert "IMG" not in text

def test_extract_metadata_with_extra():
    text, lat, lon, _ = extract_metadata(
        Path("/photos/img.jpg"),
        extra={
            "title": "Family Reunion",
            "people": ["Alice", "Bob"],
            "keywords": ["outdoor", "summer"],
            "date": "2023:06:15 10:00:00",
            "original_filename": "DSC_9999.jpg",
        },
    )
    assert "Family Reunion" in text
    assert "Alice" in text
    assert "Bob" in text
    assert "outdoor" in text
    assert "summer" in text
    assert "June 2023" in text

def test_extract_metadata_extra_date_used_when_no_exif(tmp_path):
    """Extra date is used when there's no EXIF date."""
    from PIL import Image

    img_path = tmp_path / "no_exif.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    text, lat, lon, date_taken = extract_metadata(img_path, extra={"date": "2023:01:01 00:00:00"})
    assert "January 2023" in text
    assert date_taken == "2023:01:01 00:00:00"

def test_extract_metadata_empty_for_generic():
    """Generic path with no EXIF and no extra should produce minimal/empty text."""
    text, lat, lon, _ = extract_metadata(Path("/Users/Pictures/photos/IMG_0001.jpg"))
    # Nothing meaningful: generic folders + numeric filename
    assert text == "" or len(text) < 5

def test_extract_metadata_collapses_whitespace():
    text, lat, lon, _ = extract_metadata(
        Path("/img.jpg"),
        extra={"title": "  lots   of   spaces  "},
    )
    assert "  " not in text

def test_extract_metadata_returns_coordinates_from_extra():
    """GPS from extra dict should be returned as lat/lon."""
    text, lat, lon, _ = extract_metadata(
        Path("/photos/img.jpg"),
        extra={"latitude": 48.8566, "longitude": 2.3522},
    )
    assert lat == 48.8566
    assert lon == 2.3522
    assert "Paris" in text

def test_extract_metadata_no_gps_returns_none(tmp_path):
    from PIL import Image
    img_path = tmp_path / "plain.jpg"
    Image.new("RGB", (10, 10)).save(img_path)
    text, lat, lon, _ = extract_metadata(img_path)
    assert lat is None
    assert lon is None

def test_extract_metadata_apple_photos_gps_in_text():
    """Apple Photos GPS should produce a place name in metadata text."""
    text, lat, lon, _ = extract_metadata(
        Path("/photos/img.jpg"),
        extra={
            "title": "Vacation",
            "latitude": 35.6762,
            "longitude": 139.6503,
        },
    )
    assert lat == 35.6762
    assert lon == 139.6503
    # Should have reverse geocoded to a Japanese location
    assert "Japan" in text or "Tokyo" in text
