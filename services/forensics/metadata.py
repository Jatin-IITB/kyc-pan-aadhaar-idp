from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

from PIL import Image, ExifTags


EDITING_SOFTWARE = {
    "photoshop", "gimp", "paint.net", "affinity", "pixlr",
    "snapseed", "lightroom", "capture one",
}


class MetadataForensics:
    """Analyze EXIF / image metadata for tampering indicators."""

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        if not image_bytes:
            return {"metadata_flags": [], "software_edited": False, "anomalies": []}

        flags: List[str] = []
        anomalies: List[Dict[str, str]] = []
        software_edited = False

        try:
            img = Image.open(BytesIO(image_bytes))
        except Exception:
            return {"metadata_flags": ["unreadable_image"], "software_edited": False, "anomalies": []}

        exif_data = {}
        try:
            raw_exif = img.getexif()
            exif_data = {
                ExifTags.TAGS.get(k, k): v
                for k, v in raw_exif.items()
            }
        except Exception:
            flags.append("no_exif")

        software = str(exif_data.get("Software", "")).lower()
        if software:
            for editor in EDITING_SOFTWARE:
                if editor in software:
                    software_edited = True
                    flags.append(f"edited_with_{editor}")
                    anomalies.append({
                        "type": "software_edit",
                        "detail": f"Software tag: {exif_data.get('Software')}",
                    })
                    break

        make = exif_data.get("Make", "")
        model = exif_data.get("Model", "")
        if not make and not model and not software:
            flags.append("no_device_info")

        datetime_orig = str(exif_data.get("DateTimeOriginal", ""))
        datetime_mod = str(exif_data.get("DateTime", ""))
        if datetime_orig and datetime_mod and datetime_orig != datetime_mod:
            flags.append("date_mismatch")
            anomalies.append({
                "type": "date_mismatch",
                "detail": f"Original: {datetime_orig}, Modified: {datetime_mod}",
            })

        if img.format == "JPEG":
            try:
                thumb_data = raw_exif.get_ifd(0x8769) if raw_exif else {}
                if thumb_data:
                    pass
            except Exception:
                pass

        return {
            "metadata_flags": flags,
            "software_edited": software_edited,
            "anomalies": anomalies,
        }
