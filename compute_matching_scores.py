import os
import sys
import torch
import numpy as np
import PIL.Image
from typing import Tuple, List, Optional

# Add mast3r root to path to ensure imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Imports - let them fail if missing to aid debugging
import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images

class MASt3RModel:
    _instance = None
    _device = None

    @classmethod
    def get_instance(cls, device='cuda', model_name="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"):
        if cls._instance is None:
            full_name = f"naver/{model_name}"
            model = AsymmetricMASt3R.from_pretrained(full_name).to(device)
            model.eval()
            cls._instance = model
            cls._device = device
        return cls._instance

def compute_matching_scores(
    img1_path: str, 
    img2_path: str, 
    model: Optional[torch.nn.Module] = None,
    device: str = 'cuda',
    img_size: int = 512,
    subsample: int = 8
) -> Tuple[float, float]:
    """
    Compute feature and geometric matching scores for a pair of images.
    
    Returns:
        (feat_score, geom_score) - both as ratios in [0, 1+] typically.
        Higher = more matches = better matching quality.
    """
    if model is None:
        model = MASt3RModel.get_instance(device=device)

    # Load images
    imgs = load_images([img1_path, img2_path], size=img_size, verbose=False)
    
    # Make pairs - symmetrize=True gives us 2 pairs: (0,1) and (1,0)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    
    # Inference
    with torch.no_grad():
        out = inference(pairs, model, device, batch_size=1, verbose=False)
    
    # The output structure is FLAT:
    # out['pred1'] contains tensors with batch dimension for all pairs
    # Shape is [num_pairs, H, W, channels]
    # With symmetrize=True and 2 images, we get 2 pairs: (0->1) and (1->0)
    # We want the first pair (0->1), so index [0]
    
    pred1 = out['pred1']
    pred2 = out['pred2']
    
    # Check required keys exist
    if 'pts3d' not in pred1 or 'desc' not in pred1 or 'conf' not in pred1:
        print("Warning: Missing required keys in pred1")
        return 0.0, 0.0
    
    # Extract tensors for first pair (img1 -> img2)
    # The tensors should already be on device from inference
    desc1 = pred1['desc'][0]      # Shape: [H, W, 24]
    desc2 = pred2['desc'][0]
    
    # For geometric matching, use the 3D points
    pts3d_1 = pred1['pts3d'][0]  # Shape: [H, W, 3]
    if 'pts3d_in_other_view' in pred2:
        pts3d_2 = pred2['pts3d_in_other_view'][0]
    else:
        pts3d_2 = pred2['pts3d'][0]
    
    # Feature matching using descriptors with fast_reciprocal_NNs
    # This is what works in our debug test - it found 577 matches
    feat_xy1, feat_xy2 = fast_reciprocal_NNs(
        desc1, desc2,
        subsample_or_initxy1=subsample,
        device=device,
        dist='dot',
        block_size=2**13
    )
    
    # Geometric matching using 3D points
    geom_xy1, geom_xy2 = fast_reciprocal_NNs(
        pts3d_1, pts3d_2,
        subsample_or_initxy1=subsample,
        device=device,
        dist='dot',
        block_size=2**13
    )
    
    # Compute scores as ratio of matches to total grid points
    H, W = desc1.shape[:2]
    total_grid_points = (H // subsample) * (W // subsample) * 2
    
    if total_grid_points <= 0:
        return 0.0, 0.0
        
    feat_score = len(feat_xy1) / total_grid_points
    geom_score = len(geom_xy1) / total_grid_points
    
    return float(feat_score), float(geom_score)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compute_matching_scores.py img1 img2 [device]")
        sys.exit(1)
        
    img1 = sys.argv[1]
    img2 = sys.argv[2]
    dev = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
    
    try:
        s1, s2 = compute_matching_scores(img1, img2, device=dev)
        print(f"Feature Score: {s1}")
        print(f"Geometric Score: {s2}")
    except Exception as e:
        print(f"Error computing scores: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
