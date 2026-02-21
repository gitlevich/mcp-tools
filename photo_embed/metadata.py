"""Extract searchable text metadata from image files.

Combines EXIF data, filename, and folder path into a single text string
suitable for embedding with CLIP/SigLIP text encoders.
"""

import calendar
import logging
import re
from pathlib import Path

import reverse_geocode
from PIL import Image
from PIL.ExifTags import GPS, IFD, Base as ExifBase

logger = logging.getLogger(__name__)

# Filename prefixes that carry no semantic value
_STRIP_PREFIXES = re.compile(
    r"^(IMG|DSCN|DSCF|DSC|PXL|Screenshot|Screen Shot|Photo|P)"
    r"[\s_\-]*",
    re.IGNORECASE,
)

# Characters to replace with spaces
_SEPARATORS = re.compile(r"[_\-]+")

# Strings that are just numbers/dates with no meaning on their own
_PURE_NUMERIC = re.compile(r"^\d+$")

# Folder names to ignore
_IGNORE_FOLDERS = {
    "photos",
    "pictures",
    "images",
    "camera",
    "camera roll",
    "dcim",
    "originals",
    "masters",
    "thumbnails",
    "desktop",
    "downloads",
    "documents",
    "users",
    "home",
    "volumes",
}


def _parse_filename(path: Path) -> str:
    """Extract meaningful words from a filename."""
    name = path.stem
    name = _STRIP_PREFIXES.sub("", name)
    name = _SEPARATORS.sub(" ", name)
    name = name.strip()

    if not name or _PURE_NUMERIC.match(name):
        return ""
    return name


def _parse_folder_path(path: Path) -> str:
    """Extract meaningful folder components from the path."""
    parts = []
    for component in path.parent.parts:
        lower = component.lower()
        if len(component) <= 1 or lower in _IGNORE_FOLDERS:
            continue
        if _PURE_NUMERIC.match(component) and len(component) == 4:
            # Likely a year
            parts.append(component)
        elif not _PURE_NUMERIC.match(component):
            cleaned = _SEPARATORS.sub(" ", component)
            if cleaned.strip():
                parts.append(cleaned.strip())
    return " ".join(parts)


def _format_date(date_str: str) -> str:
    """Convert EXIF date string to readable format.

    EXIF format: "2023:07:15 14:30:00"
    Output: "July 2023"
    """
    try:
        parts = date_str.strip().split(" ")[0].split(":")
        year = int(parts[0])
        month = int(parts[1])
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return f"{calendar.month_name[month]} {year}"
    except (ValueError, IndexError):
        pass
    return ""


# ---------------------------------------------------------------------------
# GPS extraction and reverse geocoding
# ---------------------------------------------------------------------------


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert EXIF GPS degrees/minutes/seconds to decimal degrees."""
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_gps_ifd(gps_ifd: dict) -> tuple[float, float] | None:
    """Extract (lat, lon) from a GPS EXIF IFD dict. Returns None if missing/invalid."""
    lat_dms = gps_ifd.get(GPS.GPSLatitude)
    lat_ref = gps_ifd.get(GPS.GPSLatitudeRef)
    lon_dms = gps_ifd.get(GPS.GPSLongitude)
    lon_ref = gps_ifd.get(GPS.GPSLongitudeRef)

    if not (lat_dms and lat_ref and lon_dms and lon_ref):
        return None

    try:
        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)
    except (TypeError, ValueError, IndexError):
        return None

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None

    # Filter Null Island — cameras sometimes write (0, 0) as default
    if lat == 0.0 and lon == 0.0:
        return None

    return (lat, lon)


def _reverse_geocode(lat: float, lon: float) -> str:
    """Reverse geocode coordinates to 'City, Country'. Returns '' on error."""
    try:
        result = reverse_geocode.get((lat, lon))
        city = result.get("city", "")
        country = result.get("country", "")
        if city and country:
            return f"{city}, {country}"
        return city or country
    except Exception:
        logger.debug("Reverse geocoding failed for (%s, %s)", lat, lon, exc_info=True)
        return ""


def _extract_exif(path: Path) -> tuple[str, tuple[float, float] | None]:
    """Extract text-worthy EXIF fields and GPS coordinates from an image.

    Returns:
        (text, coords) where coords is (lat, lon) or None.
    """
    parts = []
    coords = None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return "", None

            # Date
            date_val = exif.get(ExifBase.DateTimeOriginal) or exif.get(
                ExifBase.DateTime
            )
            if date_val:
                formatted = _format_date(str(date_val))
                if formatted:
                    parts.append(formatted)

            # Camera make/model
            make = exif.get(ExifBase.Make, "").strip()
            model = exif.get(ExifBase.Model, "").strip()
            if model:
                # Model often includes make already (e.g., "Canon EOS R5")
                if make and make.lower() not in model.lower():
                    parts.append(f"{make} {model}")
                else:
                    parts.append(model)
            elif make:
                parts.append(make)

            # Description / UserComment
            desc = exif.get(ExifBase.ImageDescription, "").strip()
            if desc and desc.lower() not in ("untitled", ""):
                parts.append(desc)

            user_comment = exif.get(ExifBase.UserComment)
            if isinstance(user_comment, (str, bytes)):
                comment = (
                    user_comment.decode("utf-8", errors="ignore")
                    if isinstance(user_comment, bytes)
                    else user_comment
                )
                comment = comment.strip().strip("\x00")
                if comment and comment != desc:
                    parts.append(comment)

            # GPS
            gps_ifd = exif.get_ifd(IFD.GPSInfo)
            if gps_ifd:
                coords = _parse_gps_ifd(gps_ifd)

    except Exception:
        logger.debug("EXIF extraction failed for %s", path, exc_info=True)

    return " ".join(parts), coords


def extract_metadata(
    path: Path, extra: dict | None = None,
) -> tuple[str, float | None, float | None]:
    """Build a searchable text string from all available metadata.

    Args:
        path: Image file path.
        extra: Optional dict with additional metadata (e.g., from Apple Photos).
            Keys: title, people, keywords, date, original_filename, latitude, longitude.

    Returns:
        (metadata_text, latitude, longitude). Latitude/longitude are None when unavailable.
    """
    parts = []

    # Filename
    fname_text = _parse_filename(path)
    if fname_text:
        parts.append(fname_text)

    # Folder path
    folder_text = _parse_folder_path(path)
    if folder_text:
        parts.append(folder_text)

    # EXIF (text + GPS)
    exif_text, coords = _extract_exif(path)
    if exif_text:
        parts.append(exif_text)

    # GPS from Apple Photos extra metadata (fallback if EXIF had none)
    if not coords and extra:
        extra_lat = extra.get("latitude")
        extra_lon = extra.get("longitude")
        if extra_lat is not None and extra_lon is not None:
            coords = (extra_lat, extra_lon)

    # Reverse geocode GPS to place name
    if coords:
        place = _reverse_geocode(*coords)
        if place:
            parts.append(place)

    # Extra metadata (Apple Photos, etc.)
    if extra:
        if extra.get("title"):
            parts.append(extra["title"])
        if extra.get("original_filename"):
            orig = _parse_filename(Path(extra["original_filename"]))
            if orig and orig != fname_text:
                parts.append(orig)
        if extra.get("people"):
            parts.append(" ".join(extra["people"]))
        if extra.get("keywords"):
            parts.append(" ".join(extra["keywords"]))
        if extra.get("date") and not exif_text:
            # Only use Photos date if EXIF didn't provide one
            formatted = _format_date(str(extra["date"]))
            if formatted:
                parts.append(formatted)

    result = " ".join(parts).strip()
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result)

    lat = coords[0] if coords else None
    lon = coords[1] if coords else None
    return result, lat, lon
