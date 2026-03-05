import os
import sys
import numpy as np
import cv2
import torch
import yaml
from types import SimpleNamespace

# Ensure third_party/p2at is importable
THIRD_PARTY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party'))
P2AT_DIR = os.path.join(THIRD_PARTY_DIR, 'p2at')
if P2AT_DIR not in sys.path:
    sys.path.insert(0, P2AT_DIR)

from rtseg.common.preprocess import preprocess_bgr_for_segmentation
from rtseg.common.palette import CITYSCAPES_CAMVID_PALETTE
from rtseg.common.visualize import decode_segmap

try:
    from builders.model_builder import build_model
except ImportError:
    # Handle graceful failure when repo is missing
    build_model = None


class P2ATSegmenter:
    def __init__(self, cfg_path: str, checkpoint_path: str, device: str = "cuda"):
        """
        cfg_path: e.g. 'configs/camvid/p2at_small_camvid.yaml'
        checkpoint_path: e.g. 'checkpoints/camvid/p2at_small_Camvid.pth'
        """
        assert build_model is not None, f"P2AT official code not found at {P2AT_DIR}"
        self.device = torch.device(device)
        
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        # P2AT uses a configuration object/dictionary that gets passed to build_model.
        # It's usually a dictionary or an omegaconf style object. We represent it as dict first.
        # They usually have a config structure: cfg['MODEL']
        # For evaluation, we build it like tools/eval.py
        
        # P2AT expects specific class definitions, so we might need to mock or format the struct
        self.cfg = cfg
        
        # Load weights first to determine num_classes dynamically
        pretrained_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            
        # Some state dicts have 'module.' prefix if trained in DP / DDP, strip it.
        clean_state_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

        # Build network
        model_name = cfg.get('MODEL', {}).get('NAME', 'p2at_s')
        num_classes = cfg.get('DATASET', {}).get('NUM_CLASSES', 19)
        # Deduce from checkpoint to avoid shape mismatches
        if 'final_seg_head.conv2.weight' in clean_state_dict:
            num_classes = clean_state_dict['final_seg_head.conv2.weight'].shape[0]
            
        self.model = build_model('P2AT', 'test', model_name, num_classes=num_classes)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model.load_state_dict(clean_state_dict, strict=False)

        # Config properties for preprocessing. CamVid default shapes etc.
        # Often models use standard ImageNet mean/std or custom ones. We use ImageNet as fallback if not in cfg.
        # Assuming P2AT uses normalisation, (from their dataset transforms):
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # In CamVid configs, image sizes are usually 720x960, in Cityscapes 1024x2048.
        # We determine it from config if available, else default:
        self.target_size = (960, 720) # (W, H)
        if 'DATASET' in cfg and 'TEST_IMAGE_SIZE' in cfg['DATASET']:
            sz = cfg['DATASET']['TEST_IMAGE_SIZE']
            self.target_size = tuple(sz) if isinstance(sz, list) else (sz[1], sz[0])

    def segment(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Input:  HxWx3 BGR uint8 (OpenCV frame)
        Output: HxWx3 BGR uint8 colorized segmentation
        """
        orig_h, orig_w = bgr_frame.shape[:2]
        
        # Preprocess
        tensor = preprocess_bgr_for_segmentation(bgr_frame, self.target_size, self.mean, self.std)
        tensor = tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(tensor)
            # Some models return tuple (main_out, aux_out), some just main_out. P2AT/DDRNet returns tuple during train, single/tuple during eval depending on align_corners config.
            # Typically out[0].
            if isinstance(outputs, (list, tuple)):
                out = outputs[0]
            else:
                out = outputs
                
        # out shape: (1, Num_Classes, H_out, W_out)
        logits = out.squeeze(0) # (Num_Classes, H_out, W_out)
        pred = logits.argmax(dim=0).cpu().numpy().astype(np.int32)
        
        # Colorize
        seg_bgr = decode_segmap(pred, CITYSCAPES_CAMVID_PALETTE)
        
        # Resize back to original
        seg_bgr_resized = cv2.resize(seg_bgr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return seg_bgr_resized
