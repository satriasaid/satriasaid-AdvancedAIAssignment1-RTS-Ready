import os
import sys
import numpy as np
import cv2
import torch
from collections import OrderedDict

# Ensure third_party/ddrnet is importable
THIRD_PARTY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party'))
DDRNET_DIR = os.path.join(THIRD_PARTY_DIR, 'ddrnet')
if DDRNET_DIR not in sys.path:
    sys.path.insert(0, DDRNET_DIR)

from rtseg.common.preprocess import preprocess_bgr_for_segmentation
from rtseg.common.palette import CITYSCAPES_CAMVID_PALETTE
from rtseg.common.visualize import decode_segmap

try:
    from segmentation.DDRNet_23_slim import get_seg_model
except ImportError:
    # Handle graceful failure when repo is missing
    get_seg_model = None

class DDRNet23SlimSegmenter:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        checkpoint_path: Cityscapes or CamVid pretrained DDRNet_23_slim weights.
        """
        assert get_seg_model is not None, f"DDRNet official code not found at {DDRNET_DIR}"
        self.device = torch.device(device)
        self.target_size = (1024, 1024) # Usually 1024x1024 input for inference on Cityscapes/Camvid config depending, or 1024x2048, or fallback to webcam 16:9
        # Assuming typical sizes. We can always adjust. Let's use 1024x1024 layout to be safe or 1920x1080
        self.target_size = (1024, 512) 
        
        # Determine number of classes (usually 19 for Cityscapes, 11 for CamVid). 
        # We will assume Cityscapes (19) as a default.
        num_classes = 19
        
        # Assuming they use PyTorch default ImageNet normalization
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # Instantiate DDRNet 23 slim model
        # From DDRNet repo readmes, segmentation model:
        # models.get_seg_model(...) or model constructor
        # DDRNet repo provides network constructors. We use align_corners=False as recommended.
        try:
            # We mock the config object the DDRNet uses:
            import argparse
            class CfgNode:
                pass
            cfg = CfgNode()
            cfg.MODEL = CfgNode()
            cfg.MODEL.NAME = 'ddrnet_23_slim'
            cfg.MODEL.NUM_OUTPUTS = 2
            cfg.MODEL.EXTRA = CfgNode()
            cfg.DATASET = CfgNode()
            cfg.DATASET.NUM_CLASSES = num_classes
            cfg.MODEL.ALIGN_CORNERS = False

            # In older repositories, you might just do:
            self.model = get_seg_model(cfg=cfg)
        except Exception as e:
            # Fallback wrapper generic call
            print(f"Warning: Could not instantiate DDRNet cleanly: {e}")
            self.model = None

        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load weights
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            
            clean_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k.replace('module.', '')
                clean_state_dict[name] = v
                
            # Load strict=False because some heads might differ if pre-trained on ImageNet vs Cityscapes
            self.model.load_state_dict(clean_state_dict, strict=False)


    def segment(self, bgr_frame: np.ndarray) -> np.ndarray:
        """BGR uint8 -> BGR uint8 colorized segmentation."""
        if self.model is None:
            return bgr_frame

        orig_h, orig_w = bgr_frame.shape[:2]
        
        # Preprocess
        tensor = preprocess_bgr_for_segmentation(bgr_frame, self.target_size, self.mean, self.std)
        tensor = tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(tensor)
            # Depending on augment=True/False, returns tuple or single tensor
            if isinstance(outputs, (list, tuple)):
                out = outputs[0]  # Take main outputs
            else:
                out = outputs
                
        # Resize logits to match original frame size (DDRNet typical eval logic)
        import torch.nn.functional as F
        out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
        # out shape: (1, Num_Classes, H_out, W_out)
        logits = out.squeeze(0) # (Num_Classes, H_out, W_out)
        pred = logits.argmax(dim=0).cpu().numpy().astype(np.int32)
        
        # Colorize
        seg_bgr = decode_segmap(pred, CITYSCAPES_CAMVID_PALETTE)
        
        return seg_bgr
