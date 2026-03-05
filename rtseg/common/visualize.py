import numpy as np
import cv2

def decode_segmap(label_map: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """
    Decodes an HxW int class label map into an HxWx3 BGR uint8 image.
    
    label_map: HxW int class labels
    palette: Dictionary mapping class ID to a (B, G, R) tuple.
    Returns: HxWx3 BGR uint8 image.
    """
    assert label_map.ndim == 2, f"Expected 2D label map, got {label_map.ndim}D"
    
    h, w = label_map.shape
    bgr_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # We can vectorize mapping if we want to be fast, but for loop over classes is okay for ~19 classes
    # Best way for speed is to create a lookup table array.
    max_class_id = max(palette.keys()) if palette else 0
    lut = np.zeros((max_class_id + 1, 3), dtype=np.uint8)
    for cls_id, color_bgr in palette.items():
        if cls_id >= 0:
            lut[cls_id] = color_bgr
            
    # Map valid class IDs using fancy indexing, ignore out of bounds
    valid_mask = (label_map >= 0) & (label_map <= max_class_id)
    bgr_img[valid_mask] = lut[label_map[valid_mask]]
    
    return bgr_img
