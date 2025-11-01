"""
HerbaScan Backend - Image Preprocessing Utilities
Handles image preprocessing for model inference
"""

import numpy as np
from PIL import Image
import io


def preprocess_image(image_bytes: bytes, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image_bytes: Raw image bytes from upload
        target_size: Target image size (width, height)
    
    Returns:
        Preprocessed image array ready for model inference [1, 224, 224, 3]
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1] range
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def array_to_pil_image(img_array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        img_array: Numpy array [H, W, 3] in range [0, 1]
    
    Returns:
        PIL Image object
    """
    # Ensure range [0, 255]
    if img_array.max() <= 1.0:
        img_array = img_array * 255.0
    
    # Clip values and convert to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def resize_image(image: Image.Image, target_size: tuple) -> Image.Image:
    """
    Resize PIL Image to target size.
    
    Args:
        image: PIL Image
        target_size: Target size (width, height)
    
    Returns:
        Resized PIL Image
    """
    return image.resize(target_size, Image.LANCZOS)

