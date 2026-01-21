"""
Step 2.5: Geometric Transform Estimation & Quad Filtering (RANSAC-based)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import procrustes
from sklearn.linear_model import RANSACRegressor

from utilities import *
logger = logging.getLogger("neuron_mapping_ransac")

# ==============================================================================
# Transformation Estimation
# ==============================================================================

def estimate_affine_transform_from_points(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    allow_scaling: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate transformation: dst = A @ src + t
    
    Args:
        src_points: Source points (N, 2)
        dst_points: Destination points (N, 2)
        allow_scaling: If False, force scale = 1.0 (rigid transform)
                      If True, allow uniform scaling (similarity transform)
    
    Returns:
        A: 2x2 rotation/scale matrix
        t: 2x1 translation vector
        rotation_deg: rotation angle in degrees
    """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[1] == 2
    
    n = len(src_points)
    
    # Center the points
    src_center = src_points.mean(axis=0)
    dst_center = dst_points.mean(axis=0)
    
    print(f"      [TRANSFORM] Source center: ({src_center[0]:.2f}, {src_center[1]:.2f})")
    print(f"      [TRANSFORM] Dest center: ({dst_center[0]:.2f}, {dst_center[1]:.2f})")
    
    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center
    
    # Compute optimal rotation matrix using SVD
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection (ensure det(R) > 0)
    if np.linalg.det(R) < 0:
        print(f"      [TRANSFORM] Fixing reflection (det(R) < 0)")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Estimate scale
    if allow_scaling:
        scale = np.sum(S) / np.sum(src_centered**2)
        print(f"      [TRANSFORM] Scale factor: {scale:.4f} (scaling allowed)")
    else:
        scale = 1.0
        print(f"      [TRANSFORM] Scale factor: 1.0 (LOCKED - rigid transform)")
    
    # Combined matrix
    A = scale * R
    
    # Translation
    t = dst_center - A @ src_center
    
    # Extract rotation angle
    rotation_rad = np.arctan2(R[1, 0], R[0, 0])
    rotation_deg = np.degrees(rotation_rad)
    
    print(f"      [TRANSFORM] Rotation: {rotation_deg:.2f}°")
    print(f"      [TRANSFORM] Translation: ({t[0]:.2f}, {t[1]:.2f})")
    
    return A, t, rotation_deg

def compute_transform_residuals(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Compute residual errors for each point pair under transformation.
    
    residual_i = ||dst_i - (A @ src_i + t)||
    """
    predicted = (A @ src_points.T).T + t
    residuals = np.linalg.norm(dst_points - predicted, axis=1)
    
    print(f"      [RESIDUALS] Min: {np.min(residuals):.2f}, Max: {np.max(residuals):.2f}, Mean: {np.mean(residuals):.2f}")
    
    return residuals

def ransac_estimate_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    max_residual: float = 5.0,
    n_iterations: int = 1000,
    min_samples: int = 3,
    stop_inlier_ratio: float = 0.5,
    allow_scaling: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, np.ndarray]:
    """
    Use RANSAC to robustly estimate transformation.
    
    Args:
        allow_scaling: If False, use rigid transform (rotation + translation only)
                      If True, use similarity transform (+ uniform scaling)
    
    Returns:
        A: 2x2 transformation matrix (None if failed)
        t: 2x1 translation vector (None if failed)
        rotation_deg: rotation angle
        inlier_mask: boolean mask of inlier points
    """
    n_points = len(src_points)
    
    print(f"    [RANSAC] Starting with {n_points} point pairs")
    print(f"    [RANSAC] Transform mode: {'SIMILARITY (rotation + translation + scale)' if allow_scaling else 'RIGID (rotation + translation, scale=1.0)'}")
    print(f"    [RANSAC] Source points range: X=[{src_points[:, 0].min():.1f}, {src_points[:, 0].max():.1f}], Y=[{src_points[:, 1].min():.1f}, {src_points[:, 1].max():.1f}]")
    print(f"    [RANSAC] Dest points range: X=[{dst_points[:, 0].min():.1f}, {dst_points[:, 0].max():.1f}], Y=[{dst_points[:, 1].min():.1f}, {dst_points[:, 1].max():.1f}]")
    
    if n_points < min_samples:
        print(f"    [RANSAC] ERROR: Not enough points ({n_points} < {min_samples})")
        return None, None, 0.0, np.zeros(n_points, dtype=bool)
    
    best_inliers = None
    best_A = None
    best_t = None
    best_rotation = 0.0
    max_inlier_count = 0
    
    for iteration in range(n_iterations):
        # Randomly sample points
        sample_idx = np.random.choice(n_points, min_samples, replace=False)
        src_sample = src_points[sample_idx]
        dst_sample = dst_points[sample_idx]
        
        # Estimate transform from sample
        try:
            A, t, rotation_deg = estimate_affine_transform_from_points(
                src_sample, dst_sample, allow_scaling=allow_scaling
            )
        except Exception as e:
            if iteration < 10:  # Only print first few failures
                print(f"      [RANSAC] Iteration {iteration} failed: {e}")
            continue
        
        # Compute residuals for all points
        residuals = compute_transform_residuals(src_points, dst_points, A, t)
        
        # Find inliers
        inliers = residuals <= max_residual
        n_inliers = np.sum(inliers)
        
        if iteration < 10 or n_inliers > max_inlier_count:
            print(f"      [RANSAC] Iteration {iteration}: {n_inliers}/{n_points} inliers ({100*n_inliers/n_points:.1f}%)")
        
        # Update best model
        if n_inliers > max_inlier_count:
            max_inlier_count = n_inliers
            best_inliers = inliers
            best_A = A
            best_t = t
            best_rotation = rotation_deg
            
            print(f"      [RANSAC] ✓ New best: {n_inliers} inliers, rotation={rotation_deg:.2f}°")
            
            # Early stopping if there are enough inliers
            if n_inliers / n_points >= stop_inlier_ratio:
                print(f"      [RANSAC] Early stopping: {n_inliers}/{n_points} >= {stop_inlier_ratio}")
                break
    
    print(f"    [RANSAC] Best model: {max_inlier_count}/{n_points} inliers ({100*max_inlier_count/n_points:.1f}%)")
    
    # Refine transform using all inliers
    if best_inliers is not None and np.sum(best_inliers) >= min_samples:
        print(f"    [RANSAC] Refining transform with {np.sum(best_inliers)} inliers...")
        try:
            src_inliers = src_points[best_inliers]
            dst_inliers = dst_points[best_inliers]
            A_refined, t_refined, rotation_refined = estimate_affine_transform_from_points(
                src_inliers, dst_inliers, allow_scaling=allow_scaling
            )
            
            # Recompute inliers with refined transform
            residuals_refined = compute_transform_residuals(
                src_points, dst_points, A_refined, t_refined
            )
            inliers_refined = residuals_refined <= max_residual
            
            n_refined = np.sum(inliers_refined)
            print(f"    [RANSAC] Refined model: {n_refined}/{n_points} inliers")
            
            return A_refined, t_refined, rotation_refined, inliers_refined
            
        except Exception as e:
            print(f"    [RANSAC] Refinement failed: {e}, using initial estimate")
            return best_A, best_t, best_rotation, best_inliers
    
    print(f"    [RANSAC] Failed to find valid transform")
    return None, None, 0.0, np.zeros(n_points, dtype=bool)

# ==============================================================================
# Quad-based Geometric Filtering
# ==============================================================================

def extract_quad_centroids(
    quad_indices: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Extract centroid positions for all quads.
    
    Args:
        quad_indices: (N, 4) array of neuron indices
        centroids: (M, 2) array of neuron positions
        
    Returns:
        quad_centers: (N, 2) array of quad center positions
    """
    print(f"    [EXTRACT] Processing {len(quad_indices)} quads from {len(centroids)} neurons")
    print(f"    [EXTRACT] Quad indices range: [{quad_indices.min()}, {quad_indices.max()}]")
    print(f"    [EXTRACT] Centroids shape: {centroids.shape}")
    
    quad_positions = centroids[quad_indices]  # (N, 4, 2)
    quad_centers = quad_positions.mean(axis=1)  # (N, 2)
    
    print(f"    [EXTRACT] Quad centers range: X=[{quad_centers[:, 0].min():.1f}, {quad_centers[:, 0].max():.1f}], Y=[{quad_centers[:, 1].min():.1f}, {quad_centers[:, 1].max():.1f}]")
    
    return quad_centers

def filter_quads_by_transform(
    ref_quad_indices: np.ndarray,
    tgt_quad_indices: np.ndarray,
    ref_centroids: np.ndarray,
    tgt_centroids: np.ndarray,
    max_residual: float = 5.0,
    ransac_iterations: int = 1000,
    min_inlier_ratio: float = 0.1,
    allow_scaling: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Use RANSAC to find dominant transformation and filter quad matches.
    
    Args:
        allow_scaling: If False, use rigid transform (rotation + translation only)
                      If True, allow uniform scaling
    
    Key insight: For each quad match, the transformation that maps
    ref quad → tgt quad should be CONSISTENT across all true matches.
    
    Returns:
        inlier_mask: boolean mask of quads that fit the dominant transform
        transform_info: dict with transform parameters and statistics
    """
    n_quads = len(ref_quad_indices)
    
    logger.info(f"  Running RANSAC on {n_quads:,} quad matches...")
    logger.info(f"    Max residual: {max_residual:.1f}px")
    logger.info(f"    RANSAC iterations: {ransac_iterations}")
    logger.info(f"    Transform: {'Similarity (rotation + translation + scale)' if allow_scaling else 'Rigid (rotation + translation, scale=1.0)'}")
    
    print(f"\n  [FILTER] ===== Starting Geometric Filtering =====")
    print(f"  [FILTER] Input: {n_quads} quad matches")
    print(f"  [FILTER] Reference session has {len(ref_centroids)} neurons")
    print(f"  [FILTER] Target session has {len(tgt_centroids)} neurons")
    print(f"  [FILTER] Transform type: {'SIMILARITY (allows scaling)' if allow_scaling else 'RIGID (scale locked to 1.0)'}")
    
    # Extract quad center positions
    print(f"\n  [FILTER] Extracting REFERENCE quad centers...")
    ref_centers = extract_quad_centroids(ref_quad_indices, ref_centroids)
    
    print(f"\n  [FILTER] Extracting TARGET quad centers...")
    tgt_centers = extract_quad_centroids(tgt_quad_indices, tgt_centroids)
    
    print(f"\n  [FILTER] Computing transformation: REF → TGT")
    print(f"  [FILTER] This maps reference session coordinates to target session coordinates")
    
    # Run RANSAC
    A, t, rotation_deg, inlier_mask = ransac_estimate_transform(
        src_points=ref_centers,
        dst_points=tgt_centers,
        max_residual=max_residual,
        n_iterations=ransac_iterations,
        min_samples=4,
        stop_inlier_ratio=0.5,
        allow_scaling=allow_scaling,
    )
    
    n_inliers = np.sum(inlier_mask)
    inlier_ratio = n_inliers / n_quads if n_quads > 0 else 0.0
    
    print(f"\n  [FILTER] ===== RANSAC Results =====")
    print(f"  [FILTER] Total inliers: {n_inliers}/{n_quads} ({100*inlier_ratio:.1f}%)")
    
    # Assess transform quality
    def assess_transform_quality(rotation, translation_mag, scale, allow_scaling):
        """Determine if transform is realistic for day-to-day imaging."""
        issues = []
        if abs(rotation) > 20:
            issues.append(f"LARGE ROTATION ({abs(rotation):.1f}° > 20°)")
        if translation_mag > 100:
            issues.append(f"LARGE TRANSLATION ({translation_mag:.1f}px > 100px)")
        if allow_scaling:
            # Only check scale if scaling is allowed
            if scale < 0.8 or scale > 1.2:
                issues.append(f"UNREALISTIC SCALE ({scale:.3f} not in [0.8, 1.2])")
        # If rigid transform, scale should be exactly 1.0 (no issue to report)
        return issues
    
    # Compute statistics
    if A is not None and n_inliers > 0:
        residuals = compute_transform_residuals(ref_centers, tgt_centers, A, t)
        inlier_residuals = residuals[inlier_mask]
        
        # Decompose transformation
        scale = np.sqrt(np.linalg.det(A))
        translation_mag = np.linalg.norm(t)
        
        # Assess quality
        quality_issues = assess_transform_quality(rotation_deg, translation_mag, scale, allow_scaling)
        
        print(f"  [FILTER] Transform found:")
        print(f"    Translation: ({t[0]:.2f}, {t[1]:.2f}) pixels, magnitude: {translation_mag:.2f}px")
        print(f"    Rotation: {rotation_deg:.2f}°")
        print(f"    Scale: {scale:.4f}{'  (LOCKED)' if not allow_scaling else ''}")
        print(f"    Residuals - Mean: {np.mean(inlier_residuals):.2f}px, Median: {np.median(inlier_residuals):.2f}px")
        
        if quality_issues:
            print(f"  [FILTER] ⚠️  TRANSFORM QUALITY ISSUES:")
            for issue in quality_issues:
                print(f"    - {issue}")
            print(f"  [FILTER] This suggests descriptor matches are mostly FALSE POSITIVES")
        else:
            print(f"  [FILTER] ✓ Transform looks realistic for day-to-day imaging")
        
        transform_info = {
            'n_quads_total': int(n_quads),
            'n_inliers': int(n_inliers),
            'inlier_ratio': float(inlier_ratio),
            'translation_x': float(t[0]),
            'translation_y': float(t[1]),
            'translation_magnitude': float(np.linalg.norm(t)),
            'rotation_deg': float(rotation_deg),
            'scale': float(scale),
            'mean_residual': float(np.mean(inlier_residuals)),
            'median_residual': float(np.median(inlier_residuals)),
            'max_residual_threshold': float(max_residual),
            'transform_matrix': A.tolist(),
            'transform_translation': t.tolist(),
        }
        
        logger.info(f"    ✓ Found transform: {n_inliers:,}/{n_quads:,} inliers ({100*inlier_ratio:.1f}%)")
        logger.info(f"    Translation: ({t[0]:.1f}, {t[1]:.1f})px")
        logger.info(f"    Rotation: {rotation_deg:.2f}°")
        logger.info(f"    Scale: {scale:.3f}")
        logger.info(f"    Median residual: {np.median(inlier_residuals):.2f}px")
        
    else:
        print(f"  [FILTER] ✗ RANSAC FAILED - no valid transform found")
        logger.warning(f"    ✗ RANSAC failed to find transform")
        transform_info = {
            'n_quads_total': int(n_quads),
            'n_inliers': 0,
            'inlier_ratio': 0.0,
            'transform_matrix': None,
        }
    
    # Only accept if there are enough inliers
    if inlier_ratio < min_inlier_ratio:
        print(f"  [FILTER] ✗ Insufficient inliers: {inlier_ratio:.1%} < {min_inlier_ratio:.1%}")
        logger.warning(f"    Inlier ratio {inlier_ratio:.1%} below threshold {min_inlier_ratio:.1%}")
        return np.zeros(n_quads, dtype=bool), transform_info
    
    print(f"  [FILTER] ✓ Transform accepted ({inlier_ratio:.1%} >= {min_inlier_ratio:.1%})")
    print(f"  [FILTER] ===== Filtering Complete =====\n")
    
    return inlier_mask, transform_info

# ==============================================================================
# Session Pair Processing
# ==============================================================================

def process_session_pair(
    match_file: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> Optional[Dict[str, Any]]:
    """
    Process one session pair: load descriptor matches, apply RANSAC filtering.
    """
    print(f"\n{'='*100}")
    print(f"[PAIR] Loading: {match_file.name}")
    print(f"{'='*100}")
    
    try:
        data = np.load(match_file, allow_pickle=False)
        print(f"[PAIR] ✓ File loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load {match_file}: {e}")
        print(f"[PAIR] ✗ ERROR loading file: {e}")
        return None
    
    animal_id = decode_string_field(data['animal_id'])
    pair_name = decode_string_field(data['pair_name'])
    ref_session = decode_string_field(data['ref_session'])
    target_session = decode_string_field(data['target_session'])
    
    print(f"[PAIR] Animal: {animal_id}")
    print(f"[PAIR] Pair: {pair_name}")
    print(f"[PAIR] Reference session: {ref_session}")
    print(f"[PAIR] Target session: {target_session}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {ref_session} → {target_session}")
    logger.info(f"{'='*80}")
    
    ref_centroids = data['ref_centroids']
    tgt_centroids = data['target_centroids']
    match_indices = data['match_indices']  # (N, 8): [ref_quad_4, tgt_quad_4]
    
    n_descriptor_matches = len(match_indices)
    
    print(f"[PAIR] Reference session: {len(ref_centroids)} neurons")
    print(f"[PAIR]   Centroid range: X=[{ref_centroids[:, 0].min():.1f}, {ref_centroids[:, 0].max():.1f}], Y=[{ref_centroids[:, 1].min():.1f}, {ref_centroids[:, 1].max():.1f}]")
    print(f"[PAIR] Target session: {len(tgt_centroids)} neurons")
    print(f"[PAIR]   Centroid range: X=[{tgt_centroids[:, 0].min():.1f}, {tgt_centroids[:, 0].max():.1f}], Y=[{tgt_centroids[:, 1].min():.1f}, {tgt_centroids[:, 1].max():.1f}]")
    print(f"[PAIR] Descriptor matches: {n_descriptor_matches:,} quad pairs")
    
    logger.info(f"  Neurons: {len(ref_centroids)} → {len(tgt_centroids)}")
    logger.info(f"  Descriptor matches: {n_descriptor_matches:,} quads")
    
    if n_descriptor_matches == 0:
        logger.warning(f"  No descriptor matches for {pair_name}")
        print(f"[PAIR] ✗ No descriptor matches - skipping")
        return None
    
    # Extract ref and target quad indices
    ref_quad_indices = match_indices[:, :4].astype(int)
    tgt_quad_indices = match_indices[:, 4:].astype(int)
    
    print(f"[PAIR] Quad indices extracted:")
    print(f"  Reference quads: {ref_quad_indices.shape}, indices in [{ref_quad_indices.min()}, {ref_quad_indices.max()}]")
    print(f"  Target quads: {tgt_quad_indices.shape}, indices in [{tgt_quad_indices.min()}, {tgt_quad_indices.max()}]")
    
    # Run RANSAC to find geometric consensus
    inlier_mask, transform_info = filter_quads_by_transform(
        ref_quad_indices=ref_quad_indices,
        tgt_quad_indices=tgt_quad_indices,
        ref_centroids=ref_centroids,
        tgt_centroids=tgt_centroids,
        max_residual=config.ransac_max_residual,
        ransac_iterations=config.ransac_iterations,
        min_inlier_ratio=config.ransac_min_inlier_ratio,
        allow_scaling=config.ransac_allow_scaling,
    )
    
    # Filter to inliers only
    filtered_match_indices = match_indices[inlier_mask]
    n_filtered = len(filtered_match_indices)
    
    print(f"\n[PAIR] ===== Final Results =====")
    print(f"[PAIR] Filtered: {n_descriptor_matches:,} → {n_filtered:,} quad matches ({100*n_filtered/n_descriptor_matches:.1f}%)")
    
    logger.info(f"  Filtered: {n_descriptor_matches:,} → {n_filtered:,} quads ({100*n_filtered/n_descriptor_matches:.1f}%)")
    
    # Save filtered results
    result = {
        'animal_id': animal_id,
        'pair_name': pair_name,
        'ref_session': ref_session,
        'target_session': target_session,
        'n_ref_neurons': int(len(ref_centroids)),
        'n_target_neurons': int(len(tgt_centroids)),
        'n_descriptor_matches': int(n_descriptor_matches),
        'n_geometric_inliers': int(n_filtered),
        'filtering_ratio': float(n_filtered / n_descriptor_matches) if n_descriptor_matches > 0 else 0.0,
        **transform_info,
    }
    
    # Save filtered matches for Step 3
    output_file = output_dir / f"{pair_name}_filtered_matches.npz"
    np.savez_compressed(
        output_file,
        animal_id=animal_id,
        pair_name=pair_name,
        ref_session=ref_session,
        target_session=target_session,
        ref_centroids=ref_centroids.astype(np.float32),
        tgt_centroids=tgt_centroids.astype(np.float32),
        n_ref_neurons=len(ref_centroids),
        n_target_neurons=len(tgt_centroids),
        n_descriptor_matches=n_descriptor_matches,
        # Filtered matches only
        match_indices=filtered_match_indices.astype(np.int32),
        n_matches=n_filtered,
        # Transform info
        transform_matrix=np.array(transform_info['transform_matrix']) if transform_info.get('transform_matrix') else np.eye(2),
        transform_translation=np.array(transform_info['transform_translation']) if transform_info.get('transform_translation') else np.zeros(2),
        rotation_deg=transform_info.get('rotation_deg', 0.0),
        scale=transform_info.get('scale', 1.0),
    )
    
    print(f"[PAIR] Saved: {output_file.name}")
    print(f"{'='*100}\n")
    
    logger.info(f"  Saved: {output_file.name}")
    
    return result

# ==============================================================================
# Helper Functions for GUI Compatibility
# ==============================================================================

def discover_animals(step2_dir: Path, pattern: str = "*_matches_light.npz", verbose: bool = False) -> List[str]:
    """
    Discover unique animal IDs from Step 2 match files.
    Required for GUI compatibility.
    
    Args:
        step2_dir: Path to step_2_results directory
        pattern: File pattern to match (default: *_matches_light.npz)
        verbose: Whether to print discovery info
    """
    match_files = sorted(step2_dir.glob(pattern))
    
    animals = set()
    for f in match_files:
        try:
            data = np.load(f, allow_pickle=False)
            animal_id = decode_string_field(data['animal_id'])
            animals.add(animal_id)
        except Exception as e:
            if verbose:
                logger.warning(f"Could not read animal ID from {f.name}: {e}")
    
    if verbose:
        logger.info(f"Discovered {len(animals)} animals: {sorted(animals)}")
    
    return sorted(animals)

def load_animal_data(animal_id: str, step2_dir: Path) -> List[Path]:
    """
    Load all match files for a specific animal.
    Required for GUI compatibility.
    """
    pattern = f"{animal_id}_*_matches_light.npz"
    match_files = sorted(step2_dir.glob(pattern))
    return match_files

# For backward compatibility with old GUI imports
def run_step_2_5_all_animals(
    input_dir: str,
    output_dir: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Alias for run_step_2_5_ransac for backward compatibility.
    """
    return run_step_2_5_ransac(input_dir, output_dir, verbose)

def sweep_shape_threshold_for_animal(
    output_dir: str,
    animal_id: str,
    shape_thresholds=None,  # Ignored - RANSAC doesn't use threshold sweep
    verbose: bool = True,
    max_pairs: Optional[int] = None,
    max_matches_per_pair: Optional[int] = None,
) -> Dict[str, Any]:
    """
    GUI-compatible wrapper for processing one animal with RANSAC.
    
    Note: RANSAC doesn't sweep thresholds - it finds geometric consensus.
    This wrapper processes all session pairs for one animal.
    """
    output_path = Path(output_dir)
    step2_dir = output_path / "step_2_results"
    step2_5_dir = ensure_output_dir(output_dir, 2.5, verbose=False)
    
    config = PipelineConfig(
        input_dir=output_dir,  # Not really used
        output_dir=output_dir,
        verbose=verbose,
    )
    
    # Find all match files for this animal
    pattern = f"{animal_id}_*_matches_light.npz"
    match_files = sorted(step2_dir.glob(pattern))
    
    if not match_files:
        logger.warning(f"No match files found for animal {animal_id}")
        return {
            'animal_id': animal_id,
            'n_pairs': 0,
            'n_geometric_inliers': 0,
            'error': 'No match files found'
        }
    
    logger.info(f"Processing {len(match_files)} session pairs for animal {animal_id}")
    
    pair_results = []
    total_desc_matches = 0
    total_inliers = 0
    
    for match_file in match_files:
        result = process_session_pair(match_file, step2_5_dir, config)
        if result:
            pair_results.append(result)
            total_desc_matches += result.get('n_descriptor_matches', 0)
            total_inliers += result.get('n_geometric_inliers', 0)
    
    # Return summary in format GUI expects
    return {
        'animal_id': animal_id,
        'n_pairs': len(pair_results),
        'n_descriptor_matches': total_desc_matches,
        'n_geometric_inliers': total_inliers,
        'filtering_ratio': total_inliers / total_desc_matches if total_desc_matches > 0 else 0.0,
        # For compatibility with old GUI that expects "optimal_threshold"
        # RANSAC doesn't have one threshold, but report average inlier ratio
        'optimal_threshold': total_inliers / total_desc_matches if total_desc_matches > 0 else 0.0,
    }

def save_all_animals_summary(output_dir: str, results: List[Dict[str, Any]]):
    """
    Save summary JSON file for all animals.
    GUI-compatible wrapper.
    """
    from utilities.step_info import get_step_output_dir
    
    output_path = get_step_output_dir(2.5, output_dir)
    summary_file = output_path / "all_animals_summary.json"
    save_json_summary(results, summary_file)

# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_step_2_5_ransac(
    input_dir: str,
    output_dir: str,
    ransac_max_residual: float = 5.0,
    ransac_iterations: int = 1000,
    ransac_min_inlier_ratio: float = 0.05,
    ransac_allow_scaling: bool = False,  # DEFAULT: Rigid transform (scale = 1.0)
    processes: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run Step 2.5: RANSAC-based geometric filtering of descriptor matches.
    
    Args:
        input_dir: Input directory (not used, kept for compatibility)
        output_dir: Output directory containing step_2_results
        ransac_max_residual: Maximum residual for RANSAC inliers (pixels)
        ransac_iterations: Number of RANSAC iterations
        ransac_min_inlier_ratio: Minimum acceptable inlier ratio
        ransac_allow_scaling: If False (default), use rigid transform (rotation + translation only, scale=1.0)
                             If True, allow uniform scaling (similarity transform)
                             For day-to-day neuron tracking, should be False (brain doesn't shrink/grow)
        processes: Number of parallel processes (unused, for compatibility)
        verbose: Enable verbose logging
    
    Returns:
        List of result dictionaries for each processed pair
    """
    log_dir = Path(output_dir) / "logs_step2_5_ransac"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=verbose)
    
    output_path = Path(output_dir)
    step2_dir = output_path / "step_2_results"
    
    if not step2_dir.exists():
        logger.error(f"Step 2 results not found: {step2_dir}")
        return []
    
    match_files = sorted(step2_dir.glob("*_matches_light.npz"))
    
    if not match_files:
        logger.error(f"No match files found in {step2_dir}")
        return []
    
    print(f"\n{'#'*100}")
    print(f"# STEP 2.5: RANSAC GEOMETRIC FILTERING")
    print(f"{'#'*100}")
    print(f"Found {len(match_files)} session pairs to process")
    print(f"\nRANSAC Configuration:")
    print(f"  Max residual threshold: {ransac_max_residual:.1f} pixels")
    print(f"  RANSAC iterations: {ransac_iterations}")
    print(f"  Minimum inlier ratio: {ransac_min_inlier_ratio:.2%}")
    print(f"  Transform type: {'SIMILARITY (allows scaling)' if ransac_allow_scaling else 'RIGID (scale locked to 1.0)'}")
    if not ransac_allow_scaling:
        print(f"  NOTE: Scale is locked to 1.0 for day-to-day tracking (brain doesn't shrink/grow)")
    print(f"{'#'*100}\n")
    
    logger.info(f"Found {len(match_files)} session pairs to process")
    logger.info(f"RANSAC parameters:")
    logger.info(f"  Max residual: {ransac_max_residual:.1f}px")
    logger.info(f"  Iterations: {ransac_iterations}")
    logger.info(f"  Min inlier ratio: {ransac_min_inlier_ratio:.2%}")
    logger.info(f"  Transform: {'Similarity (scaling allowed)' if ransac_allow_scaling else 'Rigid (scale=1.0)'}")
    
    # Use step_2_5_results for GUI compatibility
    step2_5_dir = ensure_output_dir(output_dir, 2.5, verbose=False)
    
    # Create config with RANSAC parameters
    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=verbose,
    )
    # Add RANSAC parameters to config
    config.ransac_max_residual = ransac_max_residual
    config.ransac_iterations = ransac_iterations
    config.ransac_min_inlier_ratio = ransac_min_inlier_ratio
    config.ransac_allow_scaling = ransac_allow_scaling
    
    results = []
    
    for i, match_file in enumerate(match_files, 1):
        print(f"\n{'='*100}")
        print(f"[{i}/{len(match_files)}] Processing pair {i} of {len(match_files)}")
        print(f"{'='*100}")
        logger.info(f"\n[{i}/{len(match_files)}] Processing {match_file.name}")
        result = process_session_pair(match_file, step2_5_dir, config)
        if result:
            results.append(result)
    
    # Save summary
    summary_file = step2_5_dir / "all_pairs_summary.json"
    save_json_summary(results, summary_file)
    
    # Print summary statistics
    print(f"\n{'#'*100}")
    print(f"# SUMMARY")
    print(f"{'#'*100}")
    
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Processed {len(results)} session pairs")
    
    if results:
        total_desc = sum(r['n_descriptor_matches'] for r in results)
        total_filtered = sum(r['n_geometric_inliers'] for r in results)
        avg_filter_ratio = np.mean([r['filtering_ratio'] for r in results])
        
        print(f"Processed: {len(results)} session pairs")
        print(f"Total descriptor matches: {total_desc:,}")
        print(f"Total geometric inliers: {total_filtered:,}")
        print(f"Overall filtering ratio: {100*total_filtered/total_desc:.1f}%")
        print(f"Average filtering ratio: {100*avg_filter_ratio:.1f}%")
        
        # Diagnostic: assess data quality
        print(f"\n{'='*100}")
        print(f"DIAGNOSTIC: Data Quality Assessment")
        print(f"{'='*100}")
        
        if total_filtered / total_desc < 0.05:  # Less than 5% inliers
            print(f"⚠️  WARNING: Very low inlier ratio ({100*total_filtered/total_desc:.2f}%)")
            print(f"")
            print(f"This suggests your descriptor matches from Step 2 are mostly FALSE POSITIVES.")
            print(f"")
            print(f"Expected for good data:")
            print(f"  - Inlier ratio: > 10% (ideally 20-50%)")
            print(f"  - Rotation: < 20°")
            print(f"  - Translation: < 100 pixels")
            if ransac_allow_scaling:
                print(f"  - Scale: 0.8-1.2 (or use rigid transform to lock scale=1.0)")
            else:
                print(f"  - Scale: 1.0 (LOCKED in rigid transform mode)")
            print(f"")
            print(f"Your data shows:")
            
            # Compute average transform parameters
            rotations = [abs(r.get('rotation_deg', 0)) for r in results if r.get('rotation_deg') is not None]
            translations = [r.get('translation_magnitude', 0) for r in results if r.get('translation_magnitude') is not None]
            scales = [r.get('scale', 1.0) for r in results if r.get('scale') is not None]
            
            if rotations:
                print(f"  - Avg rotation: {np.mean(rotations):.1f}° (median: {np.median(rotations):.1f}°)")
            if translations:
                print(f"  - Avg translation: {np.mean(translations):.1f}px (median: {np.median(translations):.1f}px)")
            if scales and ransac_allow_scaling:
                print(f"  - Avg scale: {np.mean(scales):.3f} (median: {np.median(scales):.3f})")
            elif not ransac_allow_scaling:
                print(f"  - Scale: 1.0 (locked - using rigid transform)")
            
            print(f"")
            print(f"Recommended fixes:")
            print(f"  1. Increase descriptor_threshold in Step 2 (try 0.9 or 0.95)")
            print(f"  2. Add more stringent geometric constraints in Step 2")
            print(f"  3. Check if your imaging sessions are properly aligned")
            print(f"  4. Verify neuron centroids are accurate")
            if ransac_allow_scaling:
                print(f"  5. Consider using rigid transform mode (ransac_allow_scaling=False)")
        else:
            print(f"✓ Data quality looks good ({100*total_filtered/total_desc:.1f}% inliers)")
        
        print(f"{'='*100}\n")
        
        logger.info(f"Total descriptor matches: {total_desc:,}")
        logger.info(f"Total geometric inliers: {total_filtered:,}")
        logger.info(f"Average filtering ratio: {100*avg_filter_ratio:.1f}%")
        
        # Group by animal
        animals = {}
        for r in results:
            aid = r['animal_id']
            if aid not in animals:
                animals[aid] = []
            animals[aid].append(r)
        
        print(f"\nBy Animal:")
        logger.info(f"\nBy animal:")
        for aid, animal_results in sorted(animals.items()):
            n_pairs = len(animal_results)
            avg_inliers = np.mean([r['n_geometric_inliers'] for r in animal_results])
            avg_rotation = np.mean([abs(r.get('rotation_deg', 0)) for r in animal_results if r.get('rotation_deg')])
            avg_translation = np.mean([r.get('translation_magnitude', 0) for r in animal_results if r.get('translation_magnitude')])
            
            print(f"  {aid}:")
            print(f"    Pairs: {n_pairs}")
            print(f"    Avg inliers/pair: {avg_inliers:.0f}")
            print(f"    Avg rotation: {avg_rotation:.1f}°")
            print(f"    Avg translation: {avg_translation:.1f} pixels")
            
            logger.info(f"  {aid}: {n_pairs} pairs, avg {avg_inliers:.0f} inliers/pair")
            logger.info(f"    Avg transform: rotation={avg_rotation:.1f}°, translation={avg_translation:.1f}px")
    
    print(f"{'#'*100}\n")
    
    return results

# Alias for GUI compatibility - main entry point
def run_step_2_5_descriptor_sweep(
    input_dir: str,
    output_dir: str,
    ransac_allow_scaling: bool = False,  # Default: rigid transform
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Main entry point for Step 2.5 (RANSAC filtering).
    Named for backward compatibility with GUI that expects descriptor_sweep.
    
    By default uses rigid transform (rotation + translation, scale=1.0)
    since brain doesn't shrink/grow between imaging days.
    """
    return run_step_2_5_ransac(
        input_dir, 
        output_dir, 
        ransac_allow_scaling=ransac_allow_scaling,
        verbose=verbose
    )
