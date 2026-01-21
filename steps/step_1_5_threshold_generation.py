"""
Step 1.5: Global similarity threshold calibration with √N scaling.
"""

import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from utilities import *
logger = logging.getLogger("neuron_mapping")

# ==============================================================================
# Per-Animal Threshold Calibration
# ==============================================================================

def _convert_matches_to_array(matches: List) -> np.ndarray:
    """
    Convert match list to numpy array for saving.
    
    Parameters
    ----------
    matches : list
        List of match tuples: (ref_quad_idx, tgt_quad_idx, ref_desc, tgt_desc, distance)
        
    Returns
    -------
    np.ndarray
        Array of shape (N, 9) with [ref_idx_0..3, tgt_idx_0..3, distance]
    """
    if not matches:
        return np.array([])
    
    result = []
    for match in matches:
        ref_idx = match[0]  # tuple of 4 indices
        tgt_idx = match[1]  # tuple of 4 indices
        distance = match[4] if len(match) > 4 else 0.0
        
        row = list(ref_idx) + list(tgt_idx) + [distance]
        result.append(row)
    
    return np.array(result, dtype=np.float32)

def compute_optimal_thresholds_per_pair(
    sessions: List[Dict],
    config: PipelineConfig,
    sample_size: int = 150,
    test_thresholds: Optional[np.ndarray] = None,
    target_quality: float = 0.95,
    verbose: bool = True
) -> Dict:
    """
    Compute optimal threshold for each session pair using ONLY descriptor distance.
    
    Returns
    -------
    dict with keys:
        N_values : list of float
            Average neuron counts for each pair
        tau_values : list of float
            Optimal thresholds for each pair
        test_thresholds : np.ndarray
            Thresholds that were tested
        per_pair_qualities : np.ndarray
            Quality curves for each pair (n_pairs x n_thresholds)
        pair_names : list of str
            Names of each pair
        example_matches : list or None
            Example quad matches for visualization
        example_ref_centroids : np.ndarray or None
            Reference centroids for example matches
        example_tgt_centroids : np.ndarray or None
            Target centroids for example matches
        # NEW: Match count data
        n_matches_per_threshold : np.ndarray
            Raw match counts (n_pairs x n_thresholds)
        n_filtered_per_threshold : np.ndarray
            Filtered match counts (n_pairs x n_thresholds)
        reference_sizes : list of int
            Reference size for each pair (for normalization)
    """
    empty_result = {
        'N_values': [],
        'tau_values': [],
        'test_thresholds': np.array([]),
        'per_pair_qualities': np.array([]),
        'pair_names': [],
        'example_matches': None,
        'example_ref_centroids': None,
        'example_tgt_centroids': None,
        'n_matches_per_threshold': np.array([]),
        'n_filtered_per_threshold': np.array([]),
        'reference_sizes': [],
    }
    
    if len(sessions) < 2:
        return empty_result
    
    if test_thresholds is None:
        test_thresholds = np.linspace(0.0, 1.0, 50)
    
    N_values = []
    tau_values = []
    all_qualities = []
    pair_names = []
    
    # Track match counts per threshold per pair
    all_n_matches = []      # (n_pairs, n_thresholds)
    all_n_filtered = []     # (n_pairs, n_thresholds)
    reference_sizes = []    # (n_pairs,)
    
    # Track example matches for visualization
    example_matches = None
    example_ref_centroids = None
    example_tgt_centroids = None
    best_match_count = 0
    
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            sess_i = sessions[i]
            sess_j = sessions[j]
            
            # Always use session with more neurons as reference
            if sess_i['n_neurons'] >= sess_j['n_neurons']:
                ref_data = sess_i
                tgt_data = sess_j
            else:
                ref_data = sess_j
                tgt_data = sess_i
            
            N_ref = ref_data["n_neurons"]
            N_tgt = tgt_data["n_neurons"]
            N_avg = (N_ref + N_tgt) / 2
            
            # Build sample
            pair_sample = _build_single_pair_sample(
                ref_data, tgt_data, sample_size
            )
            
            if pair_sample is None:
                continue
            
            # Store pair name
            pair_name = f"{ref_data['session_name']}->{tgt_data['session_name']}"
            pair_names.append(pair_name)
            
            # Reference size for normalization
            min_size = min(
                pair_sample["ref_desc"].shape[0],
                pair_sample["tgt_desc"].shape[0]
            )
            reference_sizes.append(min_size)
            
            # Test thresholds to find quality curve
            qualities = []
            n_matches_for_pair = []      
            n_filtered_for_pair = []   
            
            # Track best matches for this pair (at optimal threshold)
            best_filtered_for_pair = []
            
            for thr in test_thresholds:
                # Use descriptor-only matching (no geometric filters)
                matches = match_quads_descriptor_only(
                    pair_sample["ref_desc"], 
                    pair_sample["ref_idx"],
                    pair_sample["tgt_desc"], 
                    pair_sample["tgt_idx"],
                    similarity_threshold=thr,
                    distance_metric=config.distance_metric,
                    top_k=1,
                    verbose=False,
                )
                
                n_raw = len(matches)
                
                if matches:
                    # Apply consistency filter only
                    filtered = filter_quad_matches_by_consistency(
                        matches,
                        pair_sample["ref_centroids"],
                        pair_sample["tgt_centroids"],
                        consistency_threshold=config.consistency_threshold,
                    )
                    
                    n_filt = len(filtered)
                    
                    quality = compute_match_quality(
                        n_raw,
                        n_filt,
                        min_size
                    )
                    
                    # Track best filtered matches for this pair
                    if len(filtered) > len(best_filtered_for_pair):
                        best_filtered_for_pair = filtered
                else:
                    n_filt = 0
                    quality = 0.0
                
                qualities.append(quality)
                n_matches_for_pair.append(n_raw)
                n_filtered_for_pair.append(n_filt)
            
            # Store full curves for this pair
            all_qualities.append(qualities)
            all_n_matches.append(n_matches_for_pair)
            all_n_filtered.append(n_filtered_for_pair)
            
            # Find optimal threshold (peak of quality curve)
            tau = find_threshold_for_quality_target(
                test_thresholds,
                np.array(qualities),
                target_quality
            )
            
            N_values.append(N_avg)
            tau_values.append(tau)
            
            # Keep example matches from the pair with most matches
            if len(best_filtered_for_pair) > best_match_count:
                best_match_count = len(best_filtered_for_pair)
                example_matches = _convert_matches_to_array(best_filtered_for_pair[:50])
                example_ref_centroids = pair_sample["ref_centroids"]
                example_tgt_centroids = pair_sample["tgt_centroids"]
                if verbose:
                    logger.info(f"    Saved {len(best_filtered_for_pair[:50])} example matches from {pair_name}")
            
            if verbose:
                # Show match rate at optimal threshold
                opt_idx = np.argmin(np.abs(test_thresholds - tau))
                n_at_opt = n_filtered_for_pair[opt_idx]
                rate_at_opt = n_at_opt / min_size if min_size > 0 else 0
                logger.info(
                    f"  Pair {pair_name}: "
                    f"N={N_avg:.0f}, tau={tau:.4f}, "
                    f"matches={n_at_opt}/{min_size} ({rate_at_opt:.1%})"
                )
    
    # Convert to arrays
    per_pair_qualities = np.array(all_qualities) if all_qualities else np.array([])
    n_matches_per_threshold = np.array(all_n_matches) if all_n_matches else np.array([])
    n_filtered_per_threshold = np.array(all_n_filtered) if all_n_filtered else np.array([])
    
    return {
        'N_values': N_values,
        'tau_values': tau_values,
        'test_thresholds': test_thresholds,
        'per_pair_qualities': per_pair_qualities,
        'pair_names': pair_names,
        'example_matches': example_matches,
        'example_ref_centroids': example_ref_centroids,
        'example_tgt_centroids': example_tgt_centroids,
        'n_matches_per_threshold': n_matches_per_threshold,
        'n_filtered_per_threshold': n_filtered_per_threshold,
        'reference_sizes': reference_sizes,
    }

def _build_single_pair_sample(
    ref_data: Dict,
    tgt_data: Dict,
    sample_size: int
) -> Optional[Dict]:
    """Build sample for one session pair."""
    n_ref_quads = ref_data["n_quads"]
    n_tgt_quads = tgt_data["n_quads"]
    
    if n_ref_quads == 0 or n_tgt_quads == 0:
        return None
    
    n_ref_sample = min(sample_size, n_ref_quads)
    n_tgt_sample = min(sample_size, n_tgt_quads)
    
    ref_sel = np.random.choice(n_ref_quads, n_ref_sample, replace=False)
    tgt_sel = np.random.choice(n_tgt_quads, n_tgt_sample, replace=False)
    
    return {
        "ref_desc": ref_data["quad_desc"][ref_sel],
        "ref_idx": ref_data["quad_idx"][ref_sel],
        "tgt_desc": tgt_data["quad_desc"][tgt_sel],
        "tgt_idx": tgt_data["quad_idx"][tgt_sel],
        "ref_centroids": ref_data["centroids"],
        "tgt_centroids": tgt_data["centroids"],
    }

def auto_tune_threshold_with_scaling(
    config: PipelineConfig,
    sample_size: int = 150,
    target_quality: float = 0.95,
    test_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Auto-tune threshold using √N scaling for one animal.
    """
    logger.info("")
    logger.info(f"DESCRIPTOR THRESHOLD CALIBRATION (sqrt(N)) FOR ANIMAL {config.animal_id}")
    logger.info("=" * 80)
    logger.info("Using descriptor distance ONLY (no geometric filters)")
    
    # Load session data
    animals = load_session_data(config.output_dir, config.animal_id, verbose=True)
    
    if not animals or config.animal_id not in animals:
        logger.warning(f"No data for animal {config.animal_id}")
        return {
            'animal_id': config.animal_id,
            'C': 0.0,
            'C_std': 0.0,
            'r_squared': 0.0,
            'n_pairs': 0,
        }
    
    sessions = animals[config.animal_id]
    
    logger.info(f"Found {len(sessions)} sessions")
    for s in sessions:
        logger.info(
            f"  {s['session_name']}: {s['n_neurons']} neurons, "
            f"{s['n_quads']:,} quads"
        )
    
    # Use provided thresholds or default
    if test_thresholds is None:
        test_thresholds = np.linspace(0.0, 1.0, 50)
    
    # Compute thresholds - now returns a dict
    pair_results = compute_optimal_thresholds_per_pair(
        sessions, config, sample_size, 
        test_thresholds=test_thresholds,
        target_quality=target_quality
    )
    
    N_values = pair_results['N_values']
    tau_values = pair_results['tau_values']
    
    if len(N_values) < 2:
        logger.warning(f"Insufficient pairs for animal {config.animal_id}")
        return {
            'animal_id': config.animal_id,
            'C': 0.0,
            'C_std': 0.0,
            'r_squared': 0.0,
            'n_pairs': len(N_values),
        }
    
    # Compute C value
    C, C_std, r_squared = compute_C_value_from_pairs(N_values, tau_values)
    
    # Compute mean curves across all pairs
    per_pair_qualities = pair_results['per_pair_qualities']
    n_filtered_per_threshold = pair_results['n_filtered_per_threshold']
    n_matches_per_threshold = pair_results['n_matches_per_threshold']
    
    mean_quality = np.mean(per_pair_qualities, axis=0) if len(per_pair_qualities) > 0 else np.array([])
    mean_n_matches = np.mean(n_matches_per_threshold, axis=0) if len(n_matches_per_threshold) > 0 else np.array([])
    mean_n_filtered = np.mean(n_filtered_per_threshold, axis=0) if len(n_filtered_per_threshold) > 0 else np.array([])
    
    # Find optimal threshold from mean curve
    if len(mean_quality) > 0:
        optimal_idx = np.argmax(mean_quality)
        optimal_threshold = test_thresholds[optimal_idx]
    else:
        optimal_threshold = 0.0
        optimal_idx = 0
    
    # Compute match rate at optimal threshold
    avg_ref_size = np.mean(pair_results['reference_sizes']) if pair_results['reference_sizes'] else 1
    match_rate_at_optimal = mean_n_filtered[optimal_idx] / avg_ref_size if avg_ref_size > 0 else 0
    
    logger.info("")
    logger.info(f">>> C = {C:.4f} ± {C_std:.4f}")
    logger.info(f"    R² = {r_squared:.3f}")
    logger.info(f"    From {len(N_values)} session pairs")
    logger.info(f"    Formula: tau = {C:.4f} * sqrt(N)")
    logger.info(f"    Optimal threshold (mean curve): {optimal_threshold:.4f}")
    logger.info(f"    Match rate at optimal: {match_rate_at_optimal:.1%}")
    
    # Build result dict
    result = {
        'animal_id': config.animal_id,
        'C': C,
        'C_std': C_std,
        'r_squared': r_squared,
        'n_pairs': len(N_values),
        'N_values': [float(n) for n in N_values],
        'tau_values': [float(t) for t in tau_values],
        # Quality curve data
        'test_thresholds': test_thresholds,
        'per_pair_qualities': per_pair_qualities,
        'mean_quality': mean_quality,
        'pair_names': pair_results['pair_names'],
        'optimal_threshold': optimal_threshold,
        # Example matches for visualization
        'example_matches': pair_results['example_matches'],
        'example_ref_centroids': pair_results['example_ref_centroids'],
        'example_tgt_centroids': pair_results['example_tgt_centroids'],
        # Match count data for Match Rate tab
        'n_matches_per_threshold': n_matches_per_threshold,
        'n_filtered_per_threshold': n_filtered_per_threshold,
        'mean_n_matches': mean_n_matches,
        'mean_n_filtered': mean_n_filtered,
        'reference_sizes': pair_results['reference_sizes'],
    }
    
    _save_calibration_results(result, config)
    
    return result

def _save_calibration_results(result: Dict, config: PipelineConfig):
    """Save C value calibration results including quality curves, match counts, and example matches."""
    animal_id = result['animal_id']
    
    calib_dir = ensure_output_dir(config.output_dir, 1.5, verbose=False)
    
    # Save NPZ
    npz_path = calib_dir / f"{animal_id}_threshold_calibration.npz"
    
    save_dict = {
        'animal_id': animal_id,
        'C': result['C'],
        'C_std': result['C_std'],
        'r_squared': result['r_squared'],
        'n_pairs': result['n_pairs'],
        'N_values': np.array(result['N_values']),
        'tau_values': np.array(result['tau_values']),
    }
    
    # Add quality curve fields if present
    if 'test_thresholds' in result and result['test_thresholds'] is not None:
        save_dict['test_thresholds'] = np.array(result['test_thresholds'])
    if 'per_pair_qualities' in result and result['per_pair_qualities'] is not None:
        save_dict['per_pair_qualities'] = np.array(result['per_pair_qualities'])
    if 'mean_quality' in result and result['mean_quality'] is not None:
        save_dict['mean_quality'] = np.array(result['mean_quality'])
    if 'pair_names' in result and result['pair_names'] is not None:
        save_dict['pair_names'] = np.array(result['pair_names'], dtype=object)
    if 'optimal_threshold' in result:
        save_dict['optimal_threshold'] = result['optimal_threshold']
    
    # Save match count data for Match Rate tab
    if 'n_matches_per_threshold' in result and len(result['n_matches_per_threshold']) > 0:
        save_dict['n_matches_per_threshold'] = np.array(result['n_matches_per_threshold'])
    if 'n_filtered_per_threshold' in result and len(result['n_filtered_per_threshold']) > 0:
        save_dict['n_filtered_per_threshold'] = np.array(result['n_filtered_per_threshold'])
    if 'mean_n_matches' in result and len(result['mean_n_matches']) > 0:
        save_dict['mean_n_matches'] = np.array(result['mean_n_matches'])
    if 'mean_n_filtered' in result and len(result['mean_n_filtered']) > 0:
        save_dict['mean_n_filtered'] = np.array(result['mean_n_filtered'])
    if 'reference_sizes' in result and result['reference_sizes']:
        save_dict['reference_sizes'] = np.array(result['reference_sizes'])
    
    # Save example matches for visualization
    if 'example_matches' in result and result['example_matches'] is not None:
        save_dict['example_matches'] = np.array(result['example_matches'])
        logger.info(f"  Saving {len(result['example_matches'])} example matches")
    if 'example_ref_centroids' in result and result['example_ref_centroids'] is not None:
        save_dict['example_ref_centroids'] = np.array(result['example_ref_centroids'])
    if 'example_tgt_centroids' in result and result['example_tgt_centroids'] is not None:
        save_dict['example_tgt_centroids'] = np.array(result['example_tgt_centroids'])
    
    np.savez(npz_path, **save_dict)
    logger.info(f"Saved calibration data: {npz_path}")
    
    # Save JSON (without large arrays)
    json_result = {
        'animal_id': result['animal_id'],
        'C': result['C'],
        'C_std': result['C_std'],
        'r_squared': result['r_squared'],
        'n_pairs': result['n_pairs'],
        'N_values': result['N_values'],
        'tau_values': result['tau_values'],
        'optimal_threshold': result.get('optimal_threshold', 0.0),
        'pair_names': result.get('pair_names', []),
        'n_example_matches': len(result['example_matches']) if result.get('example_matches') is not None else 0,
        'avg_reference_size': float(np.mean(result['reference_sizes'])) if result.get('reference_sizes') else 0,
    }
    json_path = calib_dir / f"{animal_id}_threshold_calibration.json"
    save_json_summary(json_result, json_path, verbose=False)
    
    # Create plot
    if len(result['N_values']) >= 2:
        _plot_calibration(result, calib_dir)
      
def _plot_calibration(result: Dict, output_dir: Path):
    """Plot C value calibration: τ vs √N."""
    N_arr = np.array(result['N_values'])
    tau_arr = np.array(result['tau_values'])
    C = result['C']
    
    # Sort by N for cleaner visualization
    sort_idx = np.argsort(N_arr)
    N_sorted = N_arr[sort_idx]
    tau_sorted = tau_arr[sort_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data points (sorted)
    ax.scatter(np.sqrt(N_sorted), tau_sorted, s=100, alpha=0.6, label='Measured τ')
    
    # Fit line
    sqrt_N_range = np.linspace(np.sqrt(N_sorted).min(), np.sqrt(N_sorted).max(), 100)
    tau_fit = C * sqrt_N_range
    ax.plot(sqrt_N_range, tau_fit, 'r--', linewidth=2, 
            label=f'τ = {C:.3f} × √N (R²={result["r_squared"]:.3f})')
    
    ax.set_xlabel('√N (√neurons)', fontsize=12)
    ax.set_ylabel('Optimal Threshold τ', fontsize=12)
    ax.set_title(f'Threshold Calibration – Animal {result["animal_id"]}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plot_path = output_dir / f"{result['animal_id']}_threshold_calibration.png"
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved calibration plot: {plot_path}")
    
# ==============================================================================
# Multi-Animal Parallel Processing
# ==============================================================================

def _worker_tune_single_animal(args):
    """Worker function for computing C value for a single animal."""
    animal_id, base_cfg_dict, sample_size, target_quality, threshold_min, threshold_max, n_threshold_points = args

    # Recreate config
    cfg_dict = dict(base_cfg_dict)
    cfg_dict["animal_id"] = animal_id
    config = PipelineConfig.from_dict(cfg_dict)

    # Setup logging
    log_dir = Path(config.output_dir) / "logs_step1_5"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=config.verbose)

    log_worker_start("threshold calibration", animal_id)

    # Create test thresholds from parameters
    test_thresholds = np.linspace(threshold_min, threshold_max, n_threshold_points)

    # Run calibration with custom thresholds
    result = auto_tune_threshold_with_scaling(
        config,
        sample_size=sample_size,
        target_quality=target_quality,
        test_thresholds=test_thresholds,
    )

    log_worker_finish("threshold calibration", animal_id, result)

    return result

def run_global_tuning_all_animals(
    input_dir: str,
    output_dir: str,
    sample_size: int = 150,
    target_quality: float = 0.95,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    n_threshold_points: int = 50,
    processes: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Discover all animals and run threshold calibration in parallel.
    
    Parameters
    ----------
    input_dir : str
        Input directory
    output_dir : str
        Output directory containing step_1_results/
    sample_size : int
        Quads to sample per session
    target_quality : float
        Target quality (fraction of max, e.g., 0.95)
    threshold_min : float
        Minimum threshold to test
    threshold_max : float
        Maximum threshold to test
    n_threshold_points : int
        Number of threshold points to test
    processes : int, optional
        Number of parallel processes
    verbose : bool
        Verbose logging
        
    Returns
    -------
    list of dict
        Results for each animal
    """
    # Base config
    base_config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=verbose,
        animal_id=None,
        skip_existing=True,
    )

    base_cfg_dict = base_config.to_dict()
    
    # Add threshold sweep parameters to config dict
    base_cfg_dict['calib_threshold_min'] = threshold_min
    base_cfg_dict['calib_threshold_max'] = threshold_max
    base_cfg_dict['calib_threshold_n_points'] = n_threshold_points

    # Discover animals from NPZ filenames
    step1_dir = Path(output_dir) / "step_1_results"
    npz_files = sorted(step1_dir.glob("*_centroids_quads.npz"))
    
    if not npz_files:
        raise RuntimeError(
            f"No *_centroids_quads.npz files found in {step1_dir}"
        )

    animal_ids = sorted({f.name.split("_")[0] for f in npz_files})
    logger.info(f"Discovered {len(animal_ids)} animals: {animal_ids}")
    logger.info(f"Threshold sweep: {threshold_min:.3f} to {threshold_max:.3f} ({n_threshold_points} points)")

    # Build worker arguments - include new threshold parameters
    args_list = [
        (animal_id, base_cfg_dict, sample_size, target_quality,
         threshold_min, threshold_max, n_threshold_points)
        for animal_id in animal_ids
    ]

    # Run in parallel
    max_workers = compute_max_workers(cpu_fraction=0.25) if processes is None else processes
    results = run_parallel_animals(
        _worker_tune_single_animal,
        args_list,
        max_workers=max_workers,
        verbose=verbose
    )

    # Save summary (JSON-safe version - exclude large arrays)
    step1_5_dir = ensure_output_dir(output_dir, 1.5, verbose=False)
    summary_path = step1_5_dir / "all_animals_summary.json"
    
    # Create JSON-safe results (exclude numpy arrays)
    json_safe_results = []
    for r in results:
        json_safe = {
            'animal_id': r.get('animal_id'),
            'C': float(r.get('C', 0.0)),
            'C_std': float(r.get('C_std', 0.0)),
            'r_squared': float(r.get('r_squared', 0.0)),
            'n_pairs': int(r.get('n_pairs', 0)),
            'optimal_threshold': float(r.get('optimal_threshold', 0.0)),
            # Convert lists/arrays to plain lists for JSON
            'N_values': [float(x) for x in r.get('N_values', [])],
            'tau_values': [float(x) for x in r.get('tau_values', [])],
            'pair_names': list(r.get('pair_names', [])) if r.get('pair_names') is not None else [],
        }
        json_safe_results.append(json_safe)
    
    save_json_summary(json_safe_results, summary_path)

    return results