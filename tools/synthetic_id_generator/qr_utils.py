from typing import Tuple
import numpy as np
import cv2
import qrcode


def generate_qr_image(data: str, size: int = 150) -> np.ndarray:
    """
    Generate a QR code image containing the given data string.
    
    Returns RGB numpy array.
    """
    qr = qrcode.QRCode(
        version=None,  # Auto-determine version based on data
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=1,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Generate PIL image
    img_pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    
    # Convert to numpy array (RGB)
    arr = np.array(img_pil)
    
    # Resize to desired size
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    
    return arr  # RGB format


def paste_rgb_roi(
    dst_bgr: np.ndarray, 
    src_rgb: np.ndarray, 
    xyxy: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Paste an RGB source image into a BGR destination image at the specified ROI.
    
    Args:
        dst_bgr: Destination image (BGR format, OpenCV default)
        src_rgb: Source image (RGB format, e.g., from PIL or QR generator)
        xyxy: Region of interest (x1, y1, x2, y2)
    
    Returns:
        Modified destination image (BGR)
    """
    x1, y1, x2, y2 = xyxy
    
    # Convert RGB source to BGR
    roi_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    
    # Resize to fit ROI exactly
    target_w = x2 - x1
    target_h = y2 - y1
    roi_bgr = cv2.resize(roi_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Paste into destination
    dst_bgr[y1:y2, x1:x2] = roi_bgr
    
    return dst_bgr
