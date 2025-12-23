import albumentations as A
import numpy as np  # CRITICAL FIX: was missing


def augmentation_pipeline(mode: str = "phone") -> A.Compose:
    """
    Acquisition realism pipeline.
    
    - 'scan': mild compression + slight blur/noise
    - 'phone': perspective + lighting + non-linear distortion + occlusions
    """
    if mode == "scan":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.6),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.15),
            A.ImageCompression(quality_lower=70, quality_upper=98, p=0.7),
        ])

    # 'phone' mode: Advanced realism
    return A.Compose([
        # Color/lighting variations (poor white balance)
        A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.30, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=3, p=0.2),
        
        A.MotionBlur(blur_limit=7, p=0.30),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.40),
        
        # Non-linear camera lens distortion (key for phone realism)
        A.ElasticTransform(alpha=120, sigma=120 * 0.09, alpha_affine=120 * 0.03, p=0.2),
        
        A.ImageCompression(quality_lower=40, quality_upper=95, p=0.55),
        A.CoarseDropout(max_holes=3, max_height=30, max_width=60, p=0.30),
        A.Perspective(scale=(0.02, 0.15), p=0.60),
        A.Rotate(limit=12, p=0.30),
    ], p=0.95)


def apply_augmentations(img: np.ndarray, mode: str) -> np.ndarray:
    """Apply configured augmentation pipeline."""
    pipeline = augmentation_pipeline(mode)
    augmented = pipeline(image=img)
    return augmented["image"]
