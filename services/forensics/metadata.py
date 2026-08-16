from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ExifTags


EDITING_SOFTWARE = {
    "photoshop", "gimp", "paint.net", "affinity", "pixlr",
    "snapseed", "lightroom", "capture one",
}

# IJG standard luminance quantization table at quality 50.
_STD_LUM_Q50 = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


class MetadataForensics:
    """Analyze EXIF / image metadata for tampering indicators."""

    LOW_QUALITY_THRESHOLD = 88

    @staticmethod
    def estimate_jpeg_quality(img: Image.Image) -> int | None:
        """Estimate JPEG save quality from quantization tables.

        Reverses the IJG scaling formula by comparing the image's luminance
        quantization table to the standard Q=50 table.  Returns an integer
        quality estimate (1-100) or None if tables are unavailable.
        """
        qtables = getattr(img, "quantization", None)
        if not qtables:
            return None
        lum = qtables.get(0)
        if lum is None or len(lum) != 64:
            return None
        lum_arr = np.array(lum, dtype=np.float64)
        scales = 100.0 * lum_arr / _STD_LUM_Q50
        s = float(np.median(scales))
        if s < 1.0:
            return 100
        if s <= 100.0:
            return max(1, min(100, round((200.0 - s) / 2.0)))
        return max(1, min(100, round(5000.0 / s)))

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        if not image_bytes:
            return {
                "metadata_flags": [], "software_edited": False,
                "anomalies": [], "jpeg_quality": None,
            }

        flags: List[str] = []
        anomalies: List[Dict[str, str]] = []
        software_edited = False

        try:
            img = Image.open(BytesIO(image_bytes))
        except Exception:
            return {
                "metadata_flags": ["unreadable_image"],
                "software_edited": False, "anomalies": [],
                "jpeg_quality": None,
            }

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

        jpeg_quality = self.estimate_jpeg_quality(img)
        low_quality = (
            jpeg_quality is not None
            and jpeg_quality < self.LOW_QUALITY_THRESHOLD
        )
        if low_quality:
            flags.append("low_jpeg_quality")
            anomalies.append({
                "type": "low_jpeg_quality",
                "detail": f"JPEG quality {jpeg_quality} (expected ≥{self.LOW_QUALITY_THRESHOLD} for genuine ID scan)",
            })

        return {
            "metadata_flags": flags,
            "software_edited": software_edited,
            "anomalies": anomalies,
            "jpeg_quality": jpeg_quality,
        }
