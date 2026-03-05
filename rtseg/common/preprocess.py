import cv2
import numpy as np
import torch
from torchvision import transforms

def preprocess_bgr_for_segmentation(
    bgr_frame: np.ndarray,
    target_size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """
    Prepares a BGR numpy frame for inference (P2AT or DDRNet).
    
    Returns tensor of shape (1, 3, H, W) normalized.
    """
    # 1. Resize to target size (W, H) expected by cv2
    img = cv2.resize(bgr_frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 2. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Convert to float32 and normalize to [0, 1] then apply mean/std (mimics transforms.ToTensor() and transforms.Normalize())
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    tensor = t(img).unsqueeze(0)  # Shape: (1, 3, H, W)
    return tensor
