"""
Step 3: Hungarian Cost Sweep + Consolidated Neuron Tracking
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from utilities import *
logger = logging.getLogger("neuron_mapping_consolidated")

# ==============================================================================
# Cost Matrix Construction (From Filtered Quads)
# ==============================================================================

def build_cost_matrix_from_filtered_quads(
    ref_centroids: np.ndarray,
    tgt_centroids: np.ndarray,
    match_indices: np.ndarray,
    use_quad_voting: bool = True,
) -> np.ndarray:
    """
    Build neuron-to-neuron cost matrix from geometrically-filtered quad matches.
    
    Args:
        ref_centroids: (N_ref, 2) reference neuron positions
        tgt_centroids: (N_tgt, 2) target neuron positions  
        match_indices: (M, 8) filtered quad matches [ref_quad_4, tgt_quad_4]
        use_quad_voting: if True, cost = -vote_count; if False, cost = min_distance
        
    Returns:
        cost_matrix: (N_ref, N_tgt) costs for Hungarian
    """
    n_ref = len(ref_centroids)
    n_tgt = len(tgt_centroids)
    
    if use_quad_voting:
        # Cost = negative vote count (more votes = lower cost = better match)
        vote_matrix = np.zeros((n_ref, n_tgt), dtype=np.int32)
        
        for quad_match in match_indices:
            ref_neurons = quad_match[:4].astype(int)
            tgt_neurons = quad_match[4:].astype(int)
            
            # Each quad match is a "vote" for all neuron pairings it contains
            for i_ref in ref_neurons:
                for i_tgt in tgt_neurons:
                    vote_matrix[i_ref, i_tgt] += 1
        
        # Convert votes to costs (negate so more votes = lower cost)
        max_votes = vote_matrix.max() if vote_matrix.size > 0 else 1
        cost_matrix = (max_votes - vote_matrix).astype(np.float32)
        
        # Penalize zero-vote pairs heavily
        cost_matrix[vote_matrix == 0] = 1e6
        
    else:
        # Cost = minimum spatial distance across all quads containing this pair
        dist_matrix = np.full((n_ref, n_tgt), 1e6, dtype=np.float32)
        
        for quad_match in match_indices:
            ref_neurons = quad_match[:4].astype(int)
            tgt_neurons = quad_match[4:].astype(int)
            
            # For each neuron pairing in this quad, compute distance
            for i_ref in ref_neurons:
                for i_tgt in tgt_neurons:
                    dist = np.linalg.norm(
                        ref_centroids[i_ref] - tgt_centroids[i_tgt]
                    )
                    dist_matrix[i_ref, i_tgt] = min(
                        dist_matrix[i_ref, i_tgt], dist
                    )
        
        cost_matrix = dist_matrix
    
    logger.info(f"    Built cost matrix: {n_ref} × {n_tgt}")
    
    return cost_matrix

# ==============================================================================
# Session Pair Processing (Sweep + Matching)
# ==============================================================================

def process_session_pair_sweep(
    filter_file: Path,
    output_dir: Path,
    use_quad_voting: bool,
    hungarian_cost_values: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Process one session pair: sweep for optimal threshold, then get actual matches.
    
    Args:
        filter_file: Path to filtered matches file
        output_dir: Output directory for sweep results
        use_quad_voting: Whether to use quad voting
        hungarian_cost_values: Array of cost thresholds to test
        
    Returns:
        Dictionary with sweep results + optimal matches
    """
    print(f"\n{'='*100}")
    print(f"[SWEEP] Processing: {filter_file.name}")
    print(f"[SWEEP] Testing {len(hungarian_cost_values)} cost thresholds: {hungarian_cost_values[0]:.1f} to {hungarian_cost_values[-1]:.1f}")
    print(f"{'='*100}")
    
    try:
        data = np.load(filter_file, allow_pickle=False)
        print(f"[SWEEP] ✓ File loaded")
    except Exception as e:
        logger.error(f"Failed to load {filter_file}: {e}")
        print(f"[SWEEP] ✗ ERROR: {e}")
        return None
    
    # Extract metadata
    animal_id = decode_string_field(data.get('animal_id', ''))
    if not animal_id:
        animal_id = filter_file.stem.split('_')[0]
    
    pair_name = decode_string_field(data.get('pair_name', ''))
    if not pair_name:
        pair_name = filter_file.stem.replace('_filtered_matches', '')
    
    ref_session = decode_string_field(data.get('ref_session', ''))
    target_session = decode_string_field(data.get('target_session', ''))
    
    print(f"[SWEEP] Animal: {animal_id}")
    print(f"[SWEEP] Sessions: {ref_session} → {target_session}")
    
    # Get core data
    ref_centroids = data['ref_centroids']
    tgt_centroids = data['tgt_centroids']
    match_indices = data['match_indices']
    
    n_ref = len(ref_centroids)
    n_tgt = len(tgt_centroids)
    n_inlier_quads = len(match_indices)
    
    print(f"[SWEEP] Reference neurons: {n_ref}")
    print(f"[SWEEP] Target neurons: {n_tgt}")
    print(f"[SWEEP] RANSAC-filtered quads: {n_inlier_quads}")
    
    if n_inlier_quads == 0:
        logger.warning(f"  No inlier matches for {pair_name}")
        print(f"[SWEEP] ✗ No geometric inliers - skipping")
        print(f"{'='*100}\n")
        return None
    
    # Build cost matrix once (same for all thresholds)
    print(f"\n[SWEEP] Building cost matrix (once for all thresholds)...")
    cost_matrix = build_cost_matrix_from_filtered_quads(
        ref_centroids=ref_centroids,
        tgt_centroids=tgt_centroids,
        match_indices=match_indices,
        use_quad_voting=use_quad_voting,
    )
    
    # Run Hungarian once to get all assignments
    print(f"[SWEEP] Running Hungarian algorithm (once)...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    all_costs = cost_matrix[row_ind, col_ind]

    # Auto-scale input range [0, 100] to actual cost range [0, max_cost]
    actual_max_cost = np.max(cost_matrix[cost_matrix < 1e6])
    scale_factor = actual_max_cost / 100.0
    hungarian_cost_values_scaled = hungarian_cost_values * scale_factor

    print(f"[SWEEP] Auto-scaled costs: input [0-100] → actual [0-{actual_max_cost:.0f}]")
    print(f"[SWEEP] Testing thresholds from {hungarian_cost_values_scaled[0]:.1f} to {hungarian_cost_values_scaled[-1]:.1f}")

    # Store results for each threshold
    match_counts = []
    match_rates = []

    print(f"\n[SWEEP] Testing {len(hungarian_cost_values_scaled)} thresholds...")
    for cost_threshold in hungarian_cost_values_scaled:
        valid = all_costs <= cost_threshold  # ← Changed from < to <=
        n_matches = np.sum(valid)
        match_rate = n_matches / n_ref if n_ref > 0 else 0.0
        
        match_counts.append(n_matches)
        match_rates.append(match_rate)
    
    match_counts = np.array(match_counts, dtype=np.int32)
    match_rates = np.array(match_rates, dtype=np.float32)
        
    # Find optimal threshold (max matches)
    optimal_idx = np.argmax(match_counts)
    optimal_threshold = hungarian_cost_values[optimal_idx]  # Keep for reporting
    optimal_threshold_scaled = hungarian_cost_values_scaled[optimal_idx]  # ← Use this for extraction
    optimal_matches = match_counts[optimal_idx]
    optimal_rate = match_rates[optimal_idx]

    print(f"\n[SWEEP] ===== Sweep Results =====")
    print(f"[SWEEP] Optimal threshold: {optimal_threshold:.2f}")
    print(f"[SWEEP] → {optimal_matches} matches ({optimal_rate*100:.1f}%)")

    # Get actual matches at optimal threshold
    valid_mask = all_costs <= optimal_threshold_scaled  # ← Use SCALED threshold!
    matched_ref_indices = row_ind[valid_mask]
    matched_tgt_indices = col_ind[valid_mask]
    matched_costs = all_costs[valid_mask]
    
    print(f"[SWEEP] Extracted {len(matched_ref_indices)} neuron pairs at optimal threshold")
    
    # Save sweep results
    sweep_file = output_dir / f"{pair_name}_sweep.npz"
    
    np.savez_compressed(
        sweep_file,
        animal_id=animal_id,
        pair_name=pair_name,
        ref_session=ref_session,
        target_session=target_session,
        ref_centroids=ref_centroids,
        tgt_centroids=tgt_centroids,
        n_ref_neurons=n_ref,
        n_target_neurons=n_tgt,
        n_inlier_quads=n_inlier_quads,
        cost_thresholds=hungarian_cost_values,
        match_counts=match_counts,
        match_rates=match_rates,
        optimal_threshold=optimal_threshold,
        optimal_matches=optimal_matches,
        optimal_rate=optimal_rate,
        matched_ref_indices=matched_ref_indices,
        matched_tgt_indices=matched_tgt_indices,
        matched_costs=matched_costs,
    )
    
    print(f"[SWEEP] Saved: {sweep_file.name}")
    print(f"{'='*100}\n")
    
    return {
        'animal_id': animal_id,
        'pair_name': pair_name,
        'ref_session': ref_session,
        'target_session': target_session,
        'n_ref_neurons': n_ref,
        'n_target_neurons': n_tgt,
        'n_inlier_quads': n_inlier_quads,
        'optimal_threshold': float(optimal_threshold),
        'optimal_matches': int(optimal_matches),
        'optimal_rate': float(optimal_rate),
        'matched_ref_indices': matched_ref_indices,
        'matched_tgt_indices': matched_tgt_indices,
        'matched_costs': matched_costs,
    }

# ==============================================================================
# Neuron Track Consolidation
# ==============================================================================

def consolidate_neuron_tracks(
    animal_id: str,
    pair_results: List[Dict[str, Any]],
    step2_5_dir: Path,
) -> Dict[str, Any]:
    """
    Consolidate pairwise matches into global neuron tracks across all sessions.
    
    This builds a mapping: global_neuron_id -> {session_idx: local_neuron_idx}
    
    Args:
        animal_id: Animal identifier
        pair_results: List of pairwise matching results
        step2_5_dir: Directory with original centroid data
        
    Returns:
        Dictionary with consolidated tracking data
    """
    print(f"\n{'='*100}")
    print(f"[CONSOLIDATE] Building global neuron tracks for {animal_id}")
    print(f"{'='*100}")
    
    # Extract all unique sessions
    sessions_set = set()
    for result in pair_results:
        sessions_set.add(result['ref_session'])
        sessions_set.add(result['target_session'])
    
    sessions = sorted(sessions_set)
    n_sessions = len(sessions)
    session_to_idx = {s: i for i, s in enumerate(sessions)}
    
    print(f"[CONSOLIDATE] Sessions: {n_sessions} → {sessions}")
    
    # Load centroids for each session
    session_centroids = {}
    session_n_neurons = {}
    
    for session in sessions:
        # Find the filtered match file that contains this session's centroids
        found = False
        for result in pair_results:
            if result['ref_session'] == session:
                # Load from the sweep file
                sweep_file = step2_5_dir.parent / "step_3_results" / f"{result['pair_name']}_sweep.npz"
                if sweep_file.exists():
                    data = np.load(sweep_file, allow_pickle=False)
                    session_centroids[session] = data['ref_centroids']
                    session_n_neurons[session] = len(data['ref_centroids'])
                    found = True
                    break
            elif result['target_session'] == session:
                sweep_file = step2_5_dir.parent / "step_3_results" / f"{result['pair_name']}_sweep.npz"
                if sweep_file.exists():
                    data = np.load(sweep_file, allow_pickle=False)
                    session_centroids[session] = data['tgt_centroids']
                    session_n_neurons[session] = len(data['tgt_centroids'])
                    found = True
                    break
        
        if not found:
            logger.warning(f"Could not find centroids for session {session}")
    
    print(f"[CONSOLIDATE] Loaded centroids for {len(session_centroids)} sessions")
    
    # Build pairwise match graph
    # Structure: {(session1, session2): {local_idx1: local_idx2}}
    pairwise_matches = {}
    
    for result in pair_results:
        ref_session = result['ref_session']
        tgt_session = result['target_session']
        
        # Build mapping
        match_map = {}
        for ref_idx, tgt_idx in zip(result['matched_ref_indices'], result['matched_tgt_indices']):
            match_map[int(ref_idx)] = int(tgt_idx)
        
        pairwise_matches[(ref_session, tgt_session)] = match_map
        
        print(f"[CONSOLIDATE]   {ref_session} → {tgt_session}: {len(match_map)} matches")
    
    # Build global tracks using transitive closure
    print(f"\n[CONSOLIDATE] Building global neuron tracks...")
    
    # Start with first session - all neurons get initial global IDs
    first_session = sessions[0]
    n_first = session_n_neurons.get(first_session, 0)
    
    # Global tracks: {global_id: {session_idx: local_idx}}
    neuron_tracks = {}
    next_global_id = 0
    
    # Initialize with first session
    for local_idx in range(n_first):
        neuron_tracks[next_global_id] = {0: local_idx}
        next_global_id += 1
    
    print(f"[CONSOLIDATE] Initialized {n_first} tracks from {first_session}")
    
    # For each subsequent session, match to previous sessions
    for session_idx in range(1, n_sessions):
        current_session = sessions[session_idx]
        n_current = session_n_neurons.get(current_session, 0)
        
        print(f"\n[CONSOLIDATE] Processing session {session_idx}: {current_session} ({n_current} neurons)")
        
        # Track which local neurons in current session have been assigned
        assigned_locals = set()
        
        # Try to match to all previous sessions
        for prev_session_idx in range(session_idx):
            prev_session = sessions[prev_session_idx]
            
            # Check if there are matches between these sessions
            # Could be (prev, current) or (current, prev)
            matches_forward = pairwise_matches.get((prev_session, current_session), {})
            matches_backward = pairwise_matches.get((current_session, prev_session), {})
            
            # If backward, invert the mapping
            if matches_backward:
                matches = {v: k for k, v in matches_backward.items()}
            else:
                matches = matches_forward
            
            if not matches:
                continue
            
            print(f"[CONSOLIDATE]   Matching to {prev_session}: {len(matches)} pairs")
            
            # For each match, find the global ID from previous session
            for prev_local, curr_local in matches.items():
                if curr_local in assigned_locals:
                    continue  # Already assigned
                
                # Find which global track contains prev_local in prev_session_idx
                found_global_id = None
                for global_id, track in neuron_tracks.items():
                    if prev_session_idx in track and track[prev_session_idx] == prev_local:
                        found_global_id = global_id
                        break
                
                if found_global_id is not None:
                    # Extend this track to current session
                    neuron_tracks[found_global_id][session_idx] = curr_local
                    assigned_locals.add(curr_local)
        
        # Any unmatched neurons in current session get new global IDs
        n_new = 0
        for local_idx in range(n_current):
            if local_idx not in assigned_locals:
                neuron_tracks[next_global_id] = {session_idx: local_idx}
                next_global_id += 1
                n_new += 1
        
        print(f"[CONSOLIDATE]   Extended {len(assigned_locals)} existing tracks")
        print(f"[CONSOLIDATE]   Created {n_new} new tracks")
    
    # Compute track lengths
    track_lengths = np.array([len(track) for track in neuron_tracks.values()], dtype=np.int32)
    
    # Statistics
    n_total_tracks = len(neuron_tracks)
    avg_track_length = np.mean(track_lengths)
    max_track_length = np.max(track_lengths)
    
    # Count tracks by length
    full_length_tracks = np.sum(track_lengths == n_sessions)
    partial_tracks = n_total_tracks - full_length_tracks
    
    print(f"\n[CONSOLIDATE] ===== Consolidation Results =====")
    print(f"[CONSOLIDATE] Total global tracks: {n_total_tracks}")
    print(f"[CONSOLIDATE] Full-length tracks (all {n_sessions} sessions): {full_length_tracks}")
    print(f"[CONSOLIDATE] Partial tracks: {partial_tracks}")
    print(f"[CONSOLIDATE] Average track length: {avg_track_length:.1f} sessions")
    print(f"[CONSOLIDATE] Maximum track length: {max_track_length} sessions")
    print(f"{'='*100}\n")
    
    return {
        'animal_id': animal_id,
        'sessions': sessions,
        'n_sessions': n_sessions,
        'neuron_tracks': neuron_tracks,
        'track_lengths': track_lengths,
        'session_centroids': session_centroids,
        'n_total_tracks': n_total_tracks,
        'full_length_tracks': full_length_tracks,
        'avg_track_length': float(avg_track_length),
        'max_track_length': int(max_track_length),
    }


# ==============================================================================
# Animal Processing (Sweep + Consolidation)
# ==============================================================================

def process_animal_complete(
    animal_id: str,
    step2_5_dir: Path,
    output_dir: Path,
    use_quad_voting: bool,
    hungarian_cost_values: np.ndarray,
) -> Dict[str, Any]:
    """
    Process one animal: sweep for optimal thresholds, match at optimal, consolidate tracks.
    """
    print(f"\n{'#'*100}")
    print(f"# ANIMAL: {animal_id}")
    print(f"{'#'*100}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing animal: {animal_id}")
    logger.info(f"{'='*80}")
    
    # Find all filtered match files for this animal
    pattern = f"{animal_id}_*_filtered_matches.npz"
    filter_files = sorted(step2_5_dir.glob(pattern))
    
    if not filter_files:
        logger.warning(f"No filtered match files found for {animal_id}")
        print(f"✗ No filtered match files found for {animal_id}")
        print(f"{'#'*100}\n")
        return {
            'animal_id': animal_id,
            'n_pairs': 0,
            'error': 'No filtered match files found'
        }
    
    print(f"Found {len(filter_files)} session pairs")
    print(f"Testing {len(hungarian_cost_values)} cost thresholds")
    logger.info(f"Found {len(filter_files)} session pairs for {animal_id}")
    
    # PHASE 1: Sweep for optimal thresholds
    print(f"\n{'─'*100}")
    print(f"PHASE 1: THRESHOLD SWEEP")
    print(f"{'─'*100}")
    
    sweep_results = []
    
    for filter_file in filter_files:
        result = process_session_pair_sweep(
            filter_file, output_dir, use_quad_voting, hungarian_cost_values
        )
        if result:
            sweep_results.append(result)
    
    if not sweep_results:
        print(f"✗ No valid sweep results for {animal_id}")
        print(f"{'#'*100}\n")
        return {
            'animal_id': animal_id,
            'n_pairs': 0,
            'error': 'No valid sweep results'
        }
    
    # PHASE 2: Consolidate tracks (uses full sweep_results with numpy arrays)
    print(f"\n{'─'*100}")
    print(f"PHASE 2: TRACK CONSOLIDATION")
    print(f"{'─'*100}")
    
    tracking_data = consolidate_neuron_tracks(animal_id, sweep_results, step2_5_dir)
    
    # PHASE 3: Save consolidated tracking file
    print(f"\n{'─'*100}")
    print(f"PHASE 3: SAVE CONSOLIDATED TRACKING")
    print(f"{'─'*100}")
    
    tracking_file = output_dir / f"{animal_id}_consolidated_tracking.npz"
    
    np.savez_compressed(
        tracking_file,
        animal_id=animal_id,
        sessions=np.array(tracking_data['sessions'], dtype=object),
        n_sessions=tracking_data['n_sessions'],
        neuron_tracks=tracking_data['neuron_tracks'],
        track_lengths=tracking_data['track_lengths'],
        n_total_tracks=tracking_data['n_total_tracks'],
        full_length_tracks=tracking_data['full_length_tracks'],
        avg_track_length=tracking_data['avg_track_length'],
        max_track_length=tracking_data['max_track_length'],
    )
    
    print(f"[SAVE] ✓ Saved consolidated tracking: {tracking_file.name}")
    print(f"[SAVE]   - {tracking_data['n_total_tracks']} global tracks")
    print(f"[SAVE]   - {tracking_data['full_length_tracks']} full-length tracks")
    print(f"[SAVE]   - Average track length: {tracking_data['avg_track_length']:.1f} sessions")
    
    # Aggregate sweep statistics
    avg_optimal_threshold = np.mean([r['optimal_threshold'] for r in sweep_results])
    avg_optimal_rate = np.mean([r['optimal_rate'] for r in sweep_results])
    total_optimal_matches = sum(r['optimal_matches'] for r in sweep_results)
    
    # ========================================================================
    # CRITICAL: Strip numpy arrays for JSON serialization
    # ========================================================================
    json_safe_pairs = []
    for r in sweep_results:
        json_safe_pairs.append({
            'pair_name': r['pair_name'],
            'ref_session': r['ref_session'],
            'target_session': r['target_session'],
            'n_ref_neurons': r['n_ref_neurons'],
            'n_target_neurons': r['n_target_neurons'],
            'n_inlier_quads': r['n_inlier_quads'],
            'optimal_threshold': r['optimal_threshold'],
            'optimal_matches': r['optimal_matches'],
            'optimal_rate': r['optimal_rate'],
            # NOTE: matched_ref_indices, matched_tgt_indices, matched_costs
            # are numpy arrays saved in NPZ files, not included in JSON
        })
    
    print(f"\n{'#'*100}")
    print(f"# ANIMAL {animal_id} COMPLETE")
    print(f"# Session pairs: {len(sweep_results)}")
    print(f"# Avg optimal threshold: {avg_optimal_threshold:.2f}")
    print(f"# Avg optimal match rate: {avg_optimal_rate*100:.1f}%")
    print(f"# Total tracks: {tracking_data['n_total_tracks']}")
    print(f"# Full-length tracks: {tracking_data['full_length_tracks']}")
    print(f"{'#'*100}\n")
    
    return {
        'animal_id': animal_id,
        'n_pairs': len(sweep_results),
        'pair_results': json_safe_pairs,
        'avg_optimal_threshold': float(avg_optimal_threshold),
        'avg_optimal_rate': float(avg_optimal_rate),
        'total_optimal_matches': int(total_optimal_matches),
        'n_total_tracks': int(tracking_data['n_total_tracks']),  # ← Convert int64 to int
        'full_length_tracks': int(tracking_data['full_length_tracks']),  # ← Convert int64 to int
        'avg_track_length': float(tracking_data['avg_track_length']),  # ← Convert float64 to float
        'max_track_length': int(tracking_data['max_track_length']),  # ← Convert int64 to int
    }

def worker_process_animal_complete(args: Tuple) -> Dict[str, Any]:
    """Worker function for parallel animal processing."""
    animal_id, step2_5_dir_str, output_dir_str, use_quad_voting, hungarian_cost_values = args
    
    step2_5_dir = Path(step2_5_dir_str)
    output_dir = Path(output_dir_str)
    
    # Setup logging for this worker process
    log_dir = output_dir / "logs_step3"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=True)
    
    log_worker_start("Step 3 Complete", animal_id, {})
    
    result = process_animal_complete(animal_id, step2_5_dir, output_dir, use_quad_voting, hungarian_cost_values)
    
    log_worker_finish("Step 3 Complete", animal_id, result)
    
    return result

# ==============================================================================
# Main Pipeline Function
# ==============================================================================

def run_step_3_final_matching(
    input_dir: str,
    output_dir: str,
    hungarian_cost_min: float = 0.0,
    hungarian_cost_max: float = 2319.0,
    hungarian_cost_steps: int = 20,
    use_quad_voting: bool = True,
    processes: Optional[int] = None,
    verbose: bool = True,
    # Legacy parameters (ignored but accepted for compatibility)
    hungarian_max_cost: Optional[float] = None,
    target_match_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run Step 3: Hungarian sweep + consolidated neuron tracking.
    
    This is the complete Step 3 that:
    1. Sweeps over hungarian_max_cost values to find optimal thresholds
    2. Performs neuron matching at optimal thresholds
    3. Consolidates matches into global neuron tracks across sessions
    4. Saves both sweep results and consolidated tracking files
    
    Args:
        input_dir: Input directory (not used, kept for compatibility)
        output_dir: Output directory containing Step 2.5 results
        hungarian_cost_min: Minimum cost threshold to test
        hungarian_cost_max: Maximum cost threshold to test
        hungarian_cost_steps: Number of threshold values to test
        use_quad_voting: If True, use quad voting for cost matrix
        processes: Number of parallel workers (None = use all CPUs)
        verbose: Enable verbose logging
        
    Returns:
        List of results dictionaries, one per animal
    """
    output_path = Path(output_dir)
    
    # Find Step 2.5 results
    possible_dirs = [
        output_path / 'step_2_5_results',
        output_path / 'step_2_5',
    ]
    
    step2_5_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists():
            step2_5_dir = dir_path
            break
    
    if step2_5_dir is None:
        logger.error(f"Step 2.5 results not found. Checked:")
        for d in possible_dirs:
            logger.error(f"  - {d}")
        logger.error("Please run Step 2.5 RANSAC filtering first!")
        return []
    
    # Generate cost threshold values
    hungarian_cost_values = np.linspace(hungarian_cost_min, hungarian_cost_max, hungarian_cost_steps)
    
    logger.info(f"Loading Step 2.5 results from: {step2_5_dir}")
    logger.info(f"Hungarian cost sweep: {hungarian_cost_min} to {hungarian_cost_max} ({hungarian_cost_steps} steps)")
    
    # Look for filtered_matches.npz files
    filtered_files = sorted(step2_5_dir.glob("*_filtered_matches.npz"))
    
    if not filtered_files:
        logger.error(f"No *_filtered_matches.npz files found in {step2_5_dir}")
        return []
    
    print(f"\n{'#'*100}")
    print(f"# STEP 3: HUNGARIAN SWEEP + CONSOLIDATED TRACKING")
    print(f"{'#'*100}")
    print(f"Loading from: {step2_5_dir}")
    print(f"Found {len(filtered_files)} filtered match files")
    print(f"Cost range: {hungarian_cost_min} to {hungarian_cost_max}")
    print(f"Steps: {hungarian_cost_steps}")
    print(f"Testing thresholds: {hungarian_cost_values}")
    
    # Extract animal IDs
    animals = set()
    for f in filtered_files:
        animal_id = f.stem.split('_')[0]
        animals.add(animal_id)
    
    animals = sorted(animals)
    
    if not animals:
        logger.error("No animals found in Step 2.5 results")
        return []
    
    print(f"Animals to process: {len(animals)} → {animals}")
    print(f"Method: {'Quad Voting' if use_quad_voting else 'Minimum Distance'}")
    print(f"{'#'*100}\n")
    
    logger.info(f"Found {len(animals)} animals to process: {animals}")
    
    # Create output directory
    step3_dir = ensure_output_dir(output_dir, 3, verbose=False)
    
    # Build arguments for parallel processing
    args_list = [
        (animal_id, str(step2_5_dir), str(step3_dir), use_quad_voting, hungarian_cost_values)
        for animal_id in animals
    ]
    
    # Process animals in parallel
    results = run_parallel_animals(
        worker_process_animal_complete,
        args_list,
        max_workers=processes,
        verbose=verbose,
    )
    
    # Save summary
    summary_file = step3_dir / "step3_summary.json"
    save_json_summary({
        'cost_min': float(hungarian_cost_min),
        'cost_max': float(hungarian_cost_max),
        'cost_steps': int(hungarian_cost_steps),
        'cost_values': hungarian_cost_values.tolist(),
        'use_quad_voting': use_quad_voting,
        'n_animals': len(results),
        'animals': results,
    }, summary_file)
    
    # Print final summary
    print(f"\n{'#'*100}")
    print(f"# STEP 3 COMPLETE")
    print(f"{'#'*100}")
    print(f"Animals processed: {len(results)}")
    print(f"Results saved to: {step3_dir}")
    
    if results:
        print(f"\n{'Per-Animal Summary:':^100}")
        print(f"{'Animal ID':<15} {'Pairs':<8} {'Opt Thresh':<12} {'Match Rate':<12} {'Tracks':<10} {'Full Tracks':<12}")
        print(f"{'-'*100}")
        for r in results:
            if r['n_pairs'] > 0:
                print(f"{r['animal_id']:<15} {r['n_pairs']:<8} "
                      f"{r['avg_optimal_threshold']:>11.2f} {r['avg_optimal_rate']*100:>11.1f}% "
                      f"{r['n_total_tracks']:>9} {r['full_length_tracks']:>11}")
    
    print(f"{'#'*100}\n")
    
    logger.info(f"Step 3 complete: {len(results)} animals")
    
    if results:
        print(f"\n{'='*80}")
        print(f"SUMMARY - Match Rates by Animal:")
        print(f"{'='*80}")
        for r in results:
            if r.get('n_pairs', 0) > 0:
                match_rate = r.get('avg_optimal_rate', 0) * 100
                print(f"  {r['animal_id']}: {match_rate:.1f}% average match rate ({r['n_pairs']} pairs)")
        print(f"{'='*80}\n")
    
    return results

