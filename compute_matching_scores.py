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


def encode_images(
    paths: List[Optional[str]], 
    model: Optional[torch.nn.Module] = None,
    device: str = 'cuda',
    img_size: int = 512,
    batch_size: int = 8
) -> List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Encode multiple images in batch, returning descriptors, 3D points, and confidence for each.
    
    Args:
        paths: List of image paths (can contain None for missing images)
        model: MASt3R model instance (will create if None)
        device: Device to run inference on
        img_size: Target image size for resizing
        batch_size: Batch size for inference (how many self-pairs to process together)
        
    Returns:
        List of (descriptors, pts3d, conf) tuples, or None for invalid/missing images.
        Each tensor has shape [H, W, channels] where channels=24 for desc, 3 for pts3d, 1 for conf.
    """
    if model is None:
        model = MASt3RModel.get_instance(device=device)
    
    # Initialize results list with None for all paths
    results = [None] * len(paths)
    valid_paths = []
    valid_indices = []
    
    # Filter out None paths and track indices
    for idx, path in enumerate(paths):
        if path is not None:
            valid_paths.append(path)
            valid_indices.append(idx)
    
    if not valid_paths:
        return results
    
    # Load all valid images
    try:
        imgs = load_images(valid_paths, size=img_size, verbose=False)
    except Exception as e:
        print(f"Warning: Failed to load images: {e}")
        # Results already initialized with None
        return results
    
    # Batch encode all images by creating self-pairs and processing in batches
    # Create self-pairs: duplicate each image to create (img, img) pairs
    # We'll process these in batches for better GPU utilization
    encoded_features = []
    num_images = len(imgs)
    
    # Process images in batches
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_imgs = imgs[batch_start:batch_end]
        batch_size_actual = len(batch_imgs)
        
        try:
            # Create self-pairs for this batch
            # Each self-pair is created by duplicating an image: [img, img]
            # make_pairs returns a list of pairs, so we combine them
            all_pairs = []
            for img in batch_imgs:
                # Create self-pair: [img, img]
                img_pair_list = [img, img]
                # make_pairs with 2 images and complete graph creates 1 pair (0->1)
                pairs = make_pairs(img_pair_list, scene_graph='complete', prefilter=None, symmetrize=False)
                all_pairs.extend(pairs)
            
            # Run batched inference on all self-pairs
            with torch.no_grad():
                out = inference(all_pairs, model, device, batch_size=batch_size_actual, verbose=False)
            
            pred1 = out['pred1']
            
            # Check required keys exist
            if 'pts3d' not in pred1 or 'desc' not in pred1:
                # If batch failed, fall back to None for this batch
                encoded_features.extend([None] * batch_size_actual)
                continue
            
            # Extract features for each image in the batch
            # pred1['desc'] has shape [num_pairs, H, W, channels]
            # We have batch_size_actual pairs (one per image)
            for i in range(batch_size_actual):
                try:
                    desc = pred1['desc'][i]      # Shape: [H, W, 24]
                    pts3d = pred1['pts3d'][i]     # Shape: [H, W, 3]
                    conf = pred1.get('conf', None)
                    if conf is not None:
                        conf = conf[i]          # Shape: [H, W, 1] or similar
                    else:
                        # If conf is not available, create a dummy tensor
                        H, W = desc.shape[:2]
                        conf = torch.ones((H, W, 1), device=desc.device, dtype=desc.dtype)
                    
                    encoded_features.append((desc, pts3d, conf))
                except (IndexError, KeyError) as e:
                    print(f"Warning: Failed to extract features for image {batch_start + i}: {e}")
                    encoded_features.append(None)
                    
        except Exception as e:
            print(f"Warning: Failed to encode batch starting at {batch_start}: {e}")
            import traceback
            traceback.print_exc()
            # Add None for all images in this failed batch
            encoded_features.extend([None] * batch_size_actual)
    
    # Map encoded features back to original indices
    encoded_idx = 0
    for orig_idx in range(len(paths)):
        if orig_idx in valid_indices:
            results[orig_idx] = encoded_features[encoded_idx]
            encoded_idx += 1
    
    return results


def match_from_embeddings(
    desc1: Optional[torch.Tensor],
    pts3d_1: Optional[torch.Tensor],
    desc2: Optional[torch.Tensor],
    pts3d_2: Optional[torch.Tensor],
    subsample: int = 8,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Compute matching scores from pre-computed embeddings (no model forward pass).
    
    Args:
        desc1, desc2: Descriptor tensors [H, W, 24] or None
        pts3d_1, pts3d_2: 3D point tensors [H, W, 3] or None
        subsample: Subsampling factor for matching
        device: Device for matching operations
        
    Returns:
        (feat_score, geom_score) - matching scores as ratios
    """
    # Handle None inputs
    if desc1 is None or desc2 is None or pts3d_1 is None or pts3d_2 is None:
        return 0.0, 0.0
    
    try:
        # Feature matching using descriptors
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
        
    except Exception as e:
        print(f"Warning: Matching from embeddings failed: {e}")
        return 0.0, 0.0


def compute_matching_scores_batch(
    paths_t: List[Optional[str]],
    paths_t1: List[Optional[str]],
    model: Optional[torch.nn.Module] = None,
    device: str = 'cuda',
    img_size: int = 512,
    subsample: int = 8,
    verbose: bool = False,
    encode_batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute K×K matching scores for two sets of images efficiently.
    Encodes each image once, then matches all pairs.
    
    Args:
        paths_t: List of K image paths at time t (can contain None)
        paths_t1: List of K image paths at time t+1 (can contain None)
        model: MASt3R model instance (will create if None)
        device: Device to run inference on
        img_size: Target image size for resizing
        subsample: Subsampling factor for matching
        verbose: If True, print timing information
        encode_batch_size: Batch size for encoding images (how many to process together)
        
    Returns:
        (feat_scores, geom_scores) - both as numpy arrays of shape [K, K]
    """
    import time
    start_time = time.time()
    
    if model is None:
        model = MASt3RModel.get_instance(device=device)
    
    K = len(paths_t)
    if len(paths_t1) != K:
        raise ValueError(f"paths_t and paths_t1 must have same length, got {K} and {len(paths_t1)}")
    
    # Initialize output arrays
    feat_scores = np.zeros((K, K), dtype=np.float32)
    geom_scores = np.zeros((K, K), dtype=np.float32)
    
    # Encode all images at t and t+1 once (with batching)
    encode_start = time.time()
    encoded_t = encode_images(paths_t, model=model, device=device, img_size=img_size, batch_size=encode_batch_size)
    encoded_t1 = encode_images(paths_t1, model=model, device=device, img_size=img_size, batch_size=encode_batch_size)
    encode_time = time.time() - encode_start
    
    if verbose:
        valid_t = sum(1 for x in encoded_t if x is not None)
        valid_t1 = sum(1 for x in encoded_t1 if x is not None)
        print(f"  Batch encoding: {valid_t + valid_t1} images in {encode_time:.2f}s (expected ~{2*K} encodings)")
    
    # Compute K×K matching scores
    match_start = time.time()
    for k in range(K):
        enc_k = encoded_t[k]
        if enc_k is None:
            continue  # Leave scores as 0.0 for missing images
            
        desc_k, pts3d_k, _ = enc_k
        
        for l in range(K):
            enc_l = encoded_t1[l]
            if enc_l is None:
                continue  # Leave scores as 0.0 for missing images
                
            desc_l, pts3d_l, _ = enc_l
            
            # Match from embeddings
            feat_score, geom_score = match_from_embeddings(
                desc_k, pts3d_k, desc_l, pts3d_l,
                subsample=subsample, device=device
            )
            
            feat_scores[k, l] = feat_score
            geom_scores[k, l] = geom_score
    match_time = time.time() - match_start
    
    total_time = time.time() - start_time
    if verbose:
        print(f"  Batch matching: {K*K} pairs in {match_time:.2f}s")
        print(f"  Total batch time: {total_time:.2f}s (encoding: {encode_time:.2f}s, matching: {match_time:.2f}s)")
    
    return feat_scores, geom_scores

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
