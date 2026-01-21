"""
Step 2: Quad matching for all animals in parallel.
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from utilities import *
logger = logging.getLogger("neuron_mapping_step2")

# ==============================================================================
# Session Pair Matching 
# ==============================================================================

def match_session_pair(
    ref_data: Dict,
    target_data: Dict,
    config: PipelineConfig,
    threshold: float,
) -> Dict[str, Any]:
    """
    Match quads between two sessions using descriptor distance + consistency filter.
    
    Same logic as Step 1.5 calibration, but:
    - Uses fixed threshold (no sweep)
    - Uses ALL quads (no sampling)
    """
    pair_name = f"{ref_data['session_name']}_to_{target_data['session_name']}"
    
    N_ref = ref_data['n_neurons']
    N_tgt = target_data['n_neurons']
    N_avg = (N_ref + N_tgt) / 2

    logger.info(f"\nMatching {ref_data['session_name']} -> {target_data['session_name']}")
    logger.info(f"  Ref: {N_ref} neurons, {ref_data['n_quads']:,} quads")
    logger.info(f"  Target: {N_tgt} neurons, {target_data['n_quads']:,} quads")
    logger.info(f"  Threshold: {threshold:.4f}")

    start_time = time.time()

    # Step 1: Descriptor-only matching (same as Step 1.5)
    raw_matches = match_quads_descriptor_only(
        ref_data['quad_desc'],
        ref_data['quad_idx'],
        target_data['quad_desc'],
        target_data['quad_idx'],
        similarity_threshold=threshold,
        distance_metric=config.distance_metric,
        top_k=1,
        verbose=False,
    )

    # Step 2: Consistency filter (same as Step 1.5)
    if raw_matches:
        filtered_matches = filter_quad_matches_by_consistency(
            raw_matches,
            ref_data['centroids'],
            target_data['centroids'],
            consistency_threshold=config.consistency_threshold,
        )
    else:
        filtered_matches = []

    match_time = time.time() - start_time

    logger.info(f"  Raw matches: {len(raw_matches):,}")
    logger.info(f"  Filtered matches: {len(filtered_matches):,}")
    logger.info(f"  Time: {match_time:.1f}s")

    # Build result
    match_data = {
        'animal_id': ref_data['animal_id'],
        'ref_session': ref_data['session_name'],
        'target_session': target_data['session_name'],
        'pair_name': pair_name,
        'ref_centroids': ref_data['centroids'],
        'target_centroids': target_data['centroids'],
        'n_ref_neurons': N_ref,
        'n_target_neurons': N_tgt,
        'n_ref_quads': ref_data['n_quads'],
        'n_target_quads': target_data['n_quads'],
        'filtered_matches': filtered_matches,
        'n_raw_matches': len(raw_matches),
        'n_filtered_matches': len(filtered_matches),
        'threshold_used': threshold,
        'N_avg': N_avg,
        'match_time': match_time,
    }

    # Save results
    step2_dir = ensure_output_dir(config.output_dir, 2, verbose=False)
    
    # Full pickle (for Step 3)
    output_file = step2_dir / f"{pair_name}_matches.pkl"
    save_intermediate_data(match_data, output_file, compress=True)

    # Light NPZ (for viewer/quick loading)
    _save_light_npz(match_data, filtered_matches, ref_data, step2_dir, pair_name)

    return match_data

def _save_light_npz(
    match_data: Dict,
    filtered_matches: List,
    ref_data: Dict,
    step2_dir: Path,
    pair_name: str
):
    """Save lightweight NPZ for fast loading."""
    try:
        if filtered_matches:
            ref_idx_arr = np.array([m[0] for m in filtered_matches], dtype=np.int32)
            tgt_idx_arr = np.array([m[1] for m in filtered_matches], dtype=np.int32)
            match_indices = np.concatenate([ref_idx_arr, tgt_idx_arr], axis=1)

            ref_desc_arr = np.stack([m[2] for m in filtered_matches]).astype(np.float32)
            tgt_desc_arr = np.stack([m[3] for m in filtered_matches]).astype(np.float32)
            distances = np.array([m[4] for m in filtered_matches], dtype=np.float32)
        else:
            match_indices = np.zeros((0, 8), dtype=np.int32)
            desc_dim = ref_data['quad_desc'].shape[1] if ref_data['quad_desc'].ndim == 2 else 4
            ref_desc_arr = np.zeros((0, desc_dim), dtype=np.float32)
            tgt_desc_arr = np.zeros((0, desc_dim), dtype=np.float32)
            distances = np.zeros((0,), dtype=np.float32)

        light_path = step2_dir / f"{pair_name}_matches_light.npz"
        np.savez_compressed(
            light_path,
            animal_id=np.bytes_(match_data["animal_id"]),
            ref_session=np.bytes_(match_data["ref_session"]),
            target_session=np.bytes_(match_data["target_session"]),
            pair_name=np.bytes_(match_data["pair_name"]),
            ref_centroids=match_data["ref_centroids"].astype(np.float32),
            target_centroids=match_data["target_centroids"].astype(np.float32),
            n_ref_neurons=np.int32(match_data["n_ref_neurons"]),
            n_target_neurons=np.int32(match_data["n_target_neurons"]),
            match_indices=match_indices,
            ref_descriptors=ref_desc_arr,
            tgt_descriptors=tgt_desc_arr,
            distances=distances,
            threshold_used=np.float32(match_data["threshold_used"]),
            n_raw_matches=np.int32(match_data["n_raw_matches"]),
            n_filtered_matches=np.int32(match_data["n_filtered_matches"]),
        )
        logger.info(f"  Saved: {light_path.name}")

    except Exception as e:
        logger.warning(f"  Failed to write light NPZ: {e}")

# ==============================================================================
# Main Step 2 Runner
# ==============================================================================

def run_step_2(config: PipelineConfig, threshold: float) -> Dict[str, Any]:
    """Run Step 2: Quad matching for all session pairs of one animal."""
    logger.info(f"Loading Step 1 data for animal {config.animal_id}...")
    
    animals = load_session_data(config.output_dir, config.animal_id, verbose=True)
    
    if not animals or config.animal_id not in animals:
        logger.warning(f"No data found for animal {config.animal_id}")
        return {'n_pairs': 0, 'total_raw_matches': 0, 'total_filtered_matches': 0, 'match_data': []}

    sessions = animals[config.animal_id]
    
    if len(sessions) < 2:
        logger.info(f"Skipping animal {config.animal_id} - only {len(sessions)} session(s)")
        return {'n_pairs': 0, 'total_raw_matches': 0, 'total_filtered_matches': 0, 'match_data': []}

    logger.info(f"Processing animal {config.animal_id} with {len(sessions)} sessions")
    logger.info(f"Using threshold: {threshold:.4f}")
    logger.info(f"Distance metric: {config.distance_metric}")
    logger.info(f"Consistency threshold: {config.consistency_threshold}")

    pairs = config.get_session_pairs(config.animal_id, sessions)
    logger.info(f"Will process {len(pairs)} session pairs")

    all_matches = []
    total_raw = 0
    total_filtered = 0

    for ref_data, target_data in pairs:
        try:
            match_data = match_session_pair(ref_data, target_data, config, threshold)
            all_matches.append(match_data)
            total_raw += match_data['n_raw_matches']
            total_filtered += match_data['n_filtered_matches']
        except Exception as e:
            logger.error(f"Error matching {ref_data['session_name']} -> {target_data['session_name']}: {e}", exc_info=True)

    clean_memory()

    logger.info(f"\nAnimal {config.animal_id} Summary:")
    logger.info(f"  Pairs: {len(all_matches)}")
    logger.info(f"  Total raw matches: {total_raw:,}")
    logger.info(f"  Total filtered matches: {total_filtered:,}")

    return {
        'n_pairs': len(all_matches),
        'total_raw_matches': total_raw,
        'total_filtered_matches': total_filtered,
        'match_data': all_matches
    }

# ==============================================================================
# Multi-Animal Parallel Processing
# ==============================================================================

def _worker_run_step_2_single_animal(args):
    """Worker function for running Step 2 on a single animal."""
    animal_id, threshold, base_cfg_dict = args

    cfg_dict = dict(base_cfg_dict)
    cfg_dict["animal_id"] = animal_id
    config = PipelineConfig.from_dict(cfg_dict)

    log_dir = Path(config.output_dir) / "logs_step2"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=config.verbose)

    log_worker_start("Step 2", animal_id, {"threshold": threshold})

    result = run_step_2(config, threshold)

    log_worker_finish("Step 2", animal_id, result)

    return {
        "animal_id": str(animal_id),
        "threshold_used": float(threshold),
        "n_pairs": int(result.get("n_pairs", 0)),
        "total_raw_matches": int(result.get("total_raw_matches", 0)),
        "total_filtered_matches": int(result.get("total_filtered_matches", 0)),
    }

def run_step_2_all_animals_parallel(
    input_dir: str,
    output_dir: str,
    processes: Optional[int] = None,
    verbose: bool = True,
    distance_metric: str = 'cosine',
    consistency_threshold: float = 0.8,
) -> List[Dict]:
    """Run Step 2 for all animals in parallel using optimal thresholds from Step 1.5."""
    output_path = Path(output_dir)
    step1_dir = output_path / "step_1_results"
    thr_summary_path = output_path / "step_1_5_results" / "all_animals_summary.json"

    thr_data = load_json_summary(thr_summary_path, verbose=True)
    if thr_data is None:
        raise FileNotFoundError("Please run Step 1.5 first!")

    # Use optimal_threshold directly from Step 1.5
    threshold_map = {
        str(entry["animal_id"]): float(entry["optimal_threshold"]) 
        for entry in thr_data 
        if "optimal_threshold" in entry
    }
    logger.info(f"Loaded optimal thresholds for {len(threshold_map)} animals:")
    for aid, thr in sorted(threshold_map.items()):
        logger.info(f"  {aid}: {thr:.4f}")

    npz_files = sorted(step1_dir.glob("*_centroids_quads.npz"))
    if not npz_files:
        raise RuntimeError(f"No NPZ files found in {step1_dir}")

    animals_with_data = sorted({f.name.split("_")[0] for f in npz_files})
    logger.info(f"Discovered {len(animals_with_data)} animals with Step 1 data")

    animals_to_run = [aid for aid in animals_with_data if aid in threshold_map]

    if not animals_to_run:
        raise RuntimeError("No animals have both Step 1 data and Step 1.5 thresholds")

    logger.info(f"Will run Step 2 for {len(animals_to_run)} animals")

    base_config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        verbose=verbose,
        animal_id=None,
        skip_existing=False,
        distance_metric=distance_metric,
        consistency_threshold=consistency_threshold,
    )

    base_cfg_dict = base_config.to_dict()

    args_list = [(aid, threshold_map[aid], base_cfg_dict) for aid in animals_to_run]

    results = run_parallel_animals(
        _worker_run_step_2_single_animal,
        args_list,
        max_workers=processes,
        verbose=verbose
    )

    # Save summary
    step2_results_dir = ensure_output_dir(output_dir, 2, verbose=False)
    summary_path = step2_results_dir / "all_animals_summary.json"
    save_json_summary(results, summary_path)

    return results
