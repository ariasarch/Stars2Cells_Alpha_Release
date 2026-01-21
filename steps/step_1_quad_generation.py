"""
Step 1: Process multiple sessions in parallel with a RAM budget to generate quads
"""

import json
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import glob
import re
import numpy as np
import psutil
import gc
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool

from utilities import *
logger = logging.getLogger("neuron_mapping_parallel")

def find_session_files(config: PipelineConfig) -> List[Dict[str, Any]]:
    """Find all .npy files and extract metadata."""
    a_files = glob.glob(str(config.input_path / "*.npy"))
    pattern = r'(\d+)_(\d+)\.npy'
    
    print(f"\n[DEBUG] Found {len(a_files)} .npy files")
    print(f"[DEBUG] Looking for pattern: {pattern}")
    
    sessions = []
    for file_path in a_files:
        filename = Path(file_path).name
        match = re.search(pattern, filename)
        
        if match:
            animal_id = match.group(1)
            session = match.group(2)
            
            print(f"[DEBUG] Matched: {filename} -> animal={animal_id}, session={session}")
            
            if config.animal_id and animal_id != config.animal_id:
                print(f"[DEBUG] Skipping {filename} - doesn't match requested animal {config.animal_id}")
                continue
            
            sessions.append({
                'file_path': file_path,
                'animal_id': animal_id,
                'session': session,
                'session_name': f"{animal_id}_{session}"
            })
        else:
            print(f"[DEBUG] NO MATCH: {filename}")
    
    sessions.sort(key=lambda x: (x['animal_id'], x['session']))
    
    print(f"\n[DEBUG] Total sessions found: {len(sessions)}")
    return sessions

def process_single_session(session_info: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    """Process a single session to extract centroids and generate quads using sparse approach."""
    session_name = session_info['session_name']
    logger.info(f"Processing {session_name}...")

    step1_dir = Path(config.output_dir) / "step_1_results"
    step1_dir.mkdir(parents=True, exist_ok=True)
    output_file = step1_dir / f"{session_name}_centroids_quads.npz"

    if config.skip_existing and output_file.exists():
        logger.info(f"  SKIPPING: Output file already exists ({output_file})")
        return {
            "session_name": session_name,
            "skipped": True,
            "reason": "already_exists",
            "n_neurons": None,
            "n_quads": None,
        }

    try:
        # HANDLE BOTH FORMATS (same as load_sessions_from_folder)
        raw_data = np.load(session_info["file_path"], allow_pickle=True)
        data = None
        
        # TRY FORMAT 1: Dictionary (0-d array containing dict)
        if isinstance(raw_data, np.ndarray) and raw_data.ndim == 0:
            try:
                data = raw_data.item()
                if not isinstance(data, dict):
                    data = None
            except (ValueError, TypeError):
                data = None
        
        # TRY FORMAT 2: Raw A matrix (3D array)
        if data is None and isinstance(raw_data, np.ndarray) and raw_data.ndim == 3:
            logger.info(f"  Converting A matrix to centroids...")
            
            centroids_x, centroids_y = extract_centroids_from_A(raw_data)
            
            # Create dictionary format
            data = {
                'centroids_x': centroids_x,
                'centroids_y': centroids_y,
                'roi_ids': np.arange(len(centroids_x)),
            }
            logger.info(f"  ✓ Extracted {len(centroids_x)} centroids from A matrix")
        
        # Validate data
        if data is None or not isinstance(data, dict):
            logger.error(f"  SKIPPING: Unrecognized file format")
            return {
                "session_name": session_name,
                "skipped": True,
                "reason": "unrecognized_format",
            }

        # Extract centroids (works for both formats)
        centroids_x = data["centroids_x"]
        centroids_y = data["centroids_y"]
        centroids = np.column_stack([centroids_y, centroids_x]).astype(np.float32)

        neuron_ids = data.get("roi_ids", np.arange(len(centroids_x)))
        n_neurons = len(centroids)

        logger.info(f"  Loaded {n_neurons} centroids")

        if n_neurons < 4:
            logger.warning(f"  SKIPPING: Not enough neurons ({n_neurons} < 4)")
            return {
                "session_name": session_name,
                "skipped": True,
                "reason": "not_enough_neurons",
                "n_neurons": n_neurons,
            }

        logger.info("  Generating quads with SPARSE approach...")
        start_time = time.time()

        final_desc, final_idx = generate_sparse_quads_triangle(centroids, config, logger)

        generation_time = time.time() - start_time
        if final_desc is None or final_idx is None:
            n_quads = 0
        else:
            n_quads = int(final_idx.shape[0])

        logger.info(f"  Generated {n_quads:,} quads in {generation_time:.1f}s")

        np.savez_compressed(
            output_file,
            animal_id=session_info["animal_id"],
            session=session_info["session"],
            session_name=session_name,
            centroids=centroids,
            neuron_ids=neuron_ids,
            quad_desc=final_desc,
            quad_idx=final_idx,
            n_neurons=n_neurons,
            n_quads=n_quads,
            generation_time=generation_time,
            generation_method="sparse",
        )
        logger.info(f"  Saved NPZ to {output_file}")

        return {
            "session_name": session_name,
            "skipped": False,
            "n_neurons": n_neurons,
            "n_quads": n_quads,
            "generation_time": generation_time,
        }

    except Exception as e:
        logger.error(f"  Error processing {session_name}: {str(e)}")
        return {
            "session_name": session_name,
            "skipped": True,
            "reason": "error",
            "error": str(e),
        }
    finally:
        clean_memory()

def compute_max_parallel_sessions(config: PipelineConfig) -> int:
    """Decide how many sessions to process in parallel based on RAM and CPU."""
    mem_info = check_memory_requirements()
    total_gb = mem_info["total_gb"]
    avail_gb = mem_info["available_gb"]

    logger.info(
        f"[PARALLEL] System memory: total={total_gb:.1f} GB, "
        f"available={avail_gb:.1f} GB"
    )

    ram_budget = max(1.0, avail_gb * (1.0 - config.parallel_safety_margin))
    if config.per_session_gb <= 0:
        max_by_ram = 1
    else:
        max_by_ram = max(1, int(ram_budget // config.per_session_gb))

    n_cores = multiprocessing.cpu_count()
    per_proc_threads = max(1, config.n_workers)
    max_by_cpu = max(1, n_cores // per_proc_threads)

    max_parallel = max(1, min(max_by_ram, max_by_cpu))

    logger.info(
        f"[PARALLEL] RAM budget ~{ram_budget:.1f} GB, "
        f"per-session ≈ {config.per_session_gb:.1f} GB → max_by_ram={max_by_ram}"
    )
    logger.info(
        f"[PARALLEL] CPU cores={n_cores}, threads/session={per_proc_threads} "
        f"→ max_by_cpu={max_by_cpu}"
    )
    logger.info(f"[PARALLEL] → Using up to {max_parallel} sessions in parallel.\n")

    return max_parallel

def _process_diagonal(
    d1: int,
    d2: int,
    third_points,
    centroids: np.ndarray,
    config,
    height_tol: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized generation of quads from a single diagonal bucket."""
    if len(third_points) < 2:
        return None, None

    tp = np.asarray(third_points, dtype=np.float64)
    verts = tp[:, 0].astype(np.int32)
    areas = tp[:, 1]
    heights = tp[:, 2]

    M = verts.shape[0]
    if M < 2:
        return None, None

    i_idx, j_idx = np.triu_indices(M, 1)

    h1 = heights[i_idx]
    h2 = heights[j_idx]
    if height_tol > 0.0:
        mask = (np.abs(h1) >= height_tol) | (np.abs(h2) >= height_tol)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        if i_idx.size == 0:
            return None, None

    quad_area = areas[i_idx] + areas[j_idx]
    mask = quad_area >= 1.0
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    if i_idx.size == 0:
        return None, None

    p1 = verts[i_idx]
    p2 = verts[j_idx]

    base = np.column_stack([
        np.full_like(p1, d1, dtype=np.int32),
        np.full_like(p1, d2, dtype=np.int32),
        p1.astype(np.int32),
        p2.astype(np.int32),
    ])

    quad_indices = np.sort(base, axis=1)
    quad_indices = np.unique(quad_indices, axis=0)
    if quad_indices.shape[0] == 0:
        return None, None

    pts = centroids[quad_indices]
    K = quad_indices.shape[0]

    diff = pts[:, :, None, :] - pts[:, None, :, :]
    dist_mats = np.linalg.norm(diff, axis=-1)

    flat = dist_mats.reshape(K, -1)
    max_flat_idx = np.argmax(flat, axis=1)
    A_idx = max_flat_idx // 4
    B_idx = max_flat_idx % 4
    max_dists = flat[np.arange(K), max_flat_idx]

    min_pair = getattr(config, "min_pairwise_distance", 0.0)

    ut_i, ut_j = np.triu_indices(4, 1)
    pair_dists = dist_mats[:, ut_i, ut_j]
    min_pair_d = pair_dists.min(axis=1)

    keep = (max_dists > 1e-9) & (min_pair_d >= min_pair)
    if not np.any(keep):
        return None, None

    A_idx = A_idx[keep]
    B_idx = B_idx[keep]
    max_dists_kept = max_dists[keep]
    pts_keep = pts[keep]
    quad_indices = quad_indices[keep]

    batch = np.arange(quad_indices.shape[0])
    A = pts_keep[batch, A_idx]
    B = pts_keep[batch, B_idx]
    AB = B - A
    distAB = max_dists_kept[:, None]
    ux = AB / distAB
    uy = np.stack([-ux[:, 1], ux[:, 0]], axis=1)

    all_idx = np.array([0, 1, 2, 3])
    C_idx = np.zeros(quad_indices.shape[0], dtype=np.int32)
    D_idx = np.zeros(quad_indices.shape[0], dtype=np.int32)
    for k in range(quad_indices.shape[0]):
        others = all_idx[(all_idx != A_idx[k]) & (all_idx != B_idx[k])]
        C_idx[k], D_idx[k] = others[0], others[1]

    C = pts_keep[batch, C_idx]
    D = pts_keep[batch, D_idx]
    AC = C - A
    AD = D - A

    xC = np.sum(AC * ux, axis=1) / max_dists_kept
    yC = np.sum(AC * uy, axis=1) / max_dists_kept
    xD = np.sum(AD * ux, axis=1) / max_dists_kept
    yD = np.sum(AD * uy, axis=1) / max_dists_kept

    swap = xC > xD
    xC, xD = np.where(swap, xD, xC), np.where(swap, xC, xD)
    yC, yD = np.where(swap, yD, yC), np.where(swap, yC, yD)

    flip = (xC + xD) > 1
    if np.any(flip):
        A_flip = B[flip]
        B_flip = A[flip]
        AB_flip = B_flip - A_flip
        ux_flip = AB_flip / max_dists_kept[flip, None]
        uy_flip = np.stack([-ux_flip[:, 1], ux_flip[:, 0]], axis=1)

        AC_flip = C[flip] - A_flip
        AD_flip = D[flip] - A_flip

        xC_flip = np.sum(AC_flip * ux_flip, axis=1) / max_dists_kept[flip]
        yC_flip = np.sum(AC_flip * uy_flip, axis=1) / max_dists_kept[flip]
        xD_flip = np.sum(AD_flip * ux_flip, axis=1) / max_dists_kept[flip]
        yD_flip = np.sum(AD_flip * uy_flip, axis=1) / max_dists_kept[flip]

        swap_flip = xC_flip > xD_flip
        xC_flip_final = np.where(swap_flip, xD_flip, xC_flip)
        xD_flip_final = np.where(swap_flip, xC_flip, xD_flip)
        yC_flip_final = np.where(swap_flip, yD_flip, yC_flip)
        yD_flip_final = np.where(swap_flip, yC_flip, yD_flip)

        xC[flip] = xC_flip_final
        xD[flip] = xD_flip_final
        yC[flip] = yC_flip_final
        yD[flip] = yD_flip_final

    ok = ~((np.abs(yC) < 1e-4) & (np.abs(yD) < 1e-4))
    if not np.any(ok):
        return None, None

    quad_indices = quad_indices[ok]
    quad_descriptors = np.stack(
        [xC[ok], yC[ok], xD[ok], yD[ok]],
        axis=1
    ).astype(np.float32)

    return quad_descriptors, quad_indices

def compute_height_threshold(
    diagonal_buckets: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    percentile: float = 25.0,
    sample_limit: int = 5_000_000,
) -> float:
    """Compute a global |height| threshold based on a percentile of sampled heights."""
    all_heights = []
    n_diagonals = len(diagonal_buckets)
    if n_diagonals == 0:
        return 0.0

    approx_per_diag = max(1, sample_limit // n_diagonals)

    for v in diagonal_buckets.values():
        h = v["height"]
        m = h.shape[0]
        if m <= approx_per_diag:
            idx = slice(None)
        else:
            idx = np.random.choice(m, size=approx_per_diag, replace=False)
        all_heights.append(np.abs(h[idx]))

    heights = np.concatenate(all_heights, axis=0)
    if heights.size == 0:
        return 0.0

    return float(np.percentile(heights, percentile))

def step1_generate_and_filter_triangles(centroids, min_tri_area, logger):
    """Step 1: Generate all triangles and filter by area."""
    import psutil
    import gc
    from itertools import combinations

    def get_ram():
        return psutil.Process().memory_info().rss / 1e9

    n = len(centroids)
    total_tri = n * (n - 1) * (n - 2) // 6

    logger.info("")
    logger.info("STEP 1: Generating triangles + filtering...")
    logger.info(f"  Total possible triangles: {total_tri:,}")

    t0 = time.time()
    ram0 = get_ram()

    combo_iter = combinations(range(n), 3)
    flat = np.fromiter((x for comb in combo_iter for x in comb),
                       dtype=np.int32, count=total_tri*3)
    triangles = flat.reshape(-1, 3)

    t1 = time.time()
    logger.info(f"  Step 1.1: Generated {len(triangles):,} triangles in {t1-t0:.2f}s "
                f"| RAM: {ram0:.2f} → {get_ram():.2f} GB")

    tri_pts = centroids[triangles]
    t2 = time.time()
    logger.info(f"  Step 1.2: Fetched triangle points in {t2-t1:.2f}s "
                f"| RAM: {get_ram():.2f} GB")

    P0 = tri_pts[:,0]; P1 = tri_pts[:,1]; P2 = tri_pts[:,2]
    areas = 0.5 * np.abs(
        (P1[:,0]-P0[:,0])*(P2[:,1]-P0[:,1]) -
        (P1[:,1]-P0[:,1])*(P2[:,0]-P0[:,0])
    ).astype(np.float32)

    del tri_pts, P0, P1, P2
    gc.collect()

    t3 = time.time()
    logger.info(f"  Step 1.3: Computed areas in {t3-t2:.2f}s | RAM: {get_ram():.2f} GB")

    mask = areas > min_tri_area
    valid_tri = triangles[mask]
    valid_areas = areas[mask]

    t4 = time.time()
    logger.info(f"  Step 1.4: Filtered to {len(valid_tri):,} valid triangles "
                f"({len(valid_tri)/total_tri:.4f} fraction) in {t4-t3:.2f}s")
    logger.info(f"  STEP 1 total: {t4-t0:.2f}s | Final RAM: {get_ram():.2f} GB")

    return valid_tri.astype(np.int32), valid_areas.astype(np.float32)

def step2_build_diagonal_buckets_chunked(
    centroids,
    valid_triangles,
    valid_areas,
    logger,
    chunk_size
):
    """Vectorized + chunked diagonal bucket construction."""
    import psutil
    import gc

    def get_ram():
        return psutil.Process().memory_info().rss / 1e9

    logger.info("")
    logger.info("STEP 2: Building diagonal buckets (chunked, NO global sort)...")

    M = valid_triangles.shape[0]
    if M == 0:
        logger.info("  No valid triangles → empty diagonal buckets.")
        return {}, 0

    num_chunks = (M + chunk_size - 1) // chunk_size
    logger.info(f"  Processing {M:,} triangles in {num_chunks} chunks.")

    buckets = defaultdict(lambda: {
        "third": [], "area": [], "height": []
    })

    t_total = time.time()
    ram0 = get_ram()

    def accumulate(diag, third, area, height):
        if diag.size == 0:
            return
        d1 = diag[:,0].astype(np.int64)
        d2 = diag[:,1].astype(np.int64)
        keys = (d1 << 32) | d2

        order = np.argsort(keys, kind="mergesort")
        keys = keys[order]
        third = third[order]
        area = area[order]
        height = height[order]

        diff = np.diff(keys) != 0
        boundaries = np.where(diff)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(keys)]))

        for s, e in zip(starts, ends):
            key = keys[s]
            d1 = int(key >> 32)
            d2 = int(key & ((1 << 32) - 1))

            buckets[(d1, d2)]["third"].append(third[s:e])
            buckets[(d1, d2)]["area"].append(area[s:e])
            buckets[(d1, d2)]["height"].append(height[s:e])

    for ci, start in enumerate(range(0, M, chunk_size), 1):
        end = min(start+chunk_size, M)
        tri = valid_triangles[start:end]
        area = valid_areas[start:end]

        i = tri[:,0]; j = tri[:,1]; k = tri[:,2]
        Pi = centroids[i]; Pj = centroids[j]; Pk = centroids[k]

        eps = 1e-9

        AB1 = Pj - Pi; base1 = np.sqrt(np.sum(AB1*AB1, axis=1))
        AB2 = Pk - Pi; base2 = np.sqrt(np.sum(AB2*AB2, axis=1))
        AB3 = Pk - Pj; base3 = np.sqrt(np.sum(AB3*AB3, axis=1))

        h1 = np.where(base1>eps, 2.0*area/base1, 0).astype(np.float32)
        h2 = np.where(base2>eps, 2.0*area/base2, 0).astype(np.float32)
        h3 = np.where(base3>eps, 2.0*area/base3, 0).astype(np.float32)

        diag1 = np.stack((np.minimum(i,j), np.maximum(i,j)), axis=1).astype(np.int32)
        diag2 = np.stack((np.minimum(i,k), np.maximum(i,k)), axis=1).astype(np.int32)
        diag3 = np.stack((np.minimum(j,k), np.maximum(j,k)), axis=1).astype(np.int32)

        third1 = k.astype(np.int32)
        third2 = j.astype(np.int32)
        third3 = i.astype(np.int32)

        accumulate(diag1, third1, area.astype(np.float32), h1)
        accumulate(diag2, third2, area.astype(np.float32), h2)
        accumulate(diag3, third3, area.astype(np.float32), h3)

        logger.info(
            f"  Chunk {ci}/{num_chunks} processed "
            f"({end-start:,} triangles) | RAM {get_ram():.2f} GB"
        )

        del tri, area, i, j, k, Pi, Pj, Pk
        del AB1, AB2, AB3, base1, base2, base3
        del h1, h2, h3
        del diag1, diag2, diag3
        del third1, third2, third3
        gc.collect()

    total_edges = 0
    for (d1,d2), data in buckets.items():
        data["third"] = np.concatenate(data["third"]).astype(np.int32)
        data["area"] = np.concatenate(data["area"]).astype(np.float32)
        data["height"] = np.concatenate(data["height"]).astype(np.float32)
        total_edges += len(data["third"])

    logger.info(
        f"STEP 2 done: {len(buckets):,} diagonals, {total_edges:,} edges "
        f"| time {time.time()-t_total:.2f}s "
        f"| RAM {ram0:.2f}→{get_ram():.2f} GB"
    )

    return buckets, total_edges

def step3_generate_quads_from_buckets(
    diagonal_buckets,
    centroids,
    config,
    logger,
    min_height: Optional[float] = None,
):
    """Step 3: For each diagonal, run _process_diagonal in parallel."""
    import psutil

    def get_ram():
        return psutil.Process().memory_info().rss / 1e9

    logger.info("")
    logger.info("STEP 3: Generating quads from diagonal buckets (parallel)...")

    items = list(diagonal_buckets.items())
    n_diag = len(items)
    if n_diag == 0:
        logger.info("  No diagonals to process → no quads.")
        return None, None

    height_thr = 0.0 if min_height is None else float(min_height)

    K = getattr(config, "max_triangles_per_diagonal", 25)
    logger.info(
        f"  Using |height| >= {height_thr:.4f} (if > 0) "
        f"and top-K={K} triangles per diagonal by AREA."
    )

    def worker(item):
        (d1, d2), data = item
        third = data["third"]
        area = data["area"]
        height = data["height"]

        if height_thr > 0.0:
            mask = np.abs(height) >= height_thr
            if not np.any(mask):
                return None, None
            third = third[mask]
            area = area[mask]
            height = height[mask]

        if third.size < 2:
            return None, None

        if third.size > K:
            idx_sel = np.argpartition(area, -K)[-K:]
            third = third[idx_sel]
            area = area[idx_sel]
            height = height[idx_sel]

        if third.size < 2:
            return None, None

        third_points = list(zip(third, area, height))

        desc, idx = _process_diagonal(
            d1, d2, third_points, centroids, config,
            height_tol=0.0,
        )

        if desc is None or idx is None or desc.size == 0:
            return None, None
        return desc.astype(np.float32), idx.astype(np.int32)

    n_workers = getattr(config, "n_workers", 8)
    pool = ThreadPool(n_workers)
    total_quads = 0
    descriptors = []
    indices = []

    t0 = time.time()
    for i, result in enumerate(pool.imap_unordered(worker, items), 1):
        desc, idx = result
        if desc is not None:
            descriptors.append(desc)
            indices.append(idx)
            total_quads += desc.shape[0]

        if i % max(1, n_diag // 50) == 0 or i == n_diag:
            logger.info(
                f"  Processed {i}/{n_diag} diagonals | "
                f"Quads so far: {total_quads:,} | RAM {get_ram():.2f} GB"
            )

    pool.close()
    pool.join()

    if not descriptors:
        logger.info("  No quads generated from any diagonal.")
        return None, None

    final_desc = np.vstack(descriptors)
    final_idx = np.vstack(indices).astype(np.int32)

    logger.info("  Deduplicating quads globally by vertex indices...")
    uniq_idx, uniq_pos = np.unique(final_idx, axis=0, return_index=True)
    if uniq_idx.shape[0] != final_idx.shape[0]:
        logger.info(
            f"  Removed {final_idx.shape[0] - uniq_idx.shape[0]:,} "
            f"duplicate quads across diagonals."
        )

    final_idx = uniq_idx
    final_desc = final_desc[uniq_pos]

    keep_fraction = getattr(config, "quad_keep_fraction", 1.0)
    if 0.0 < keep_fraction < 1.0 and final_desc.shape[0] > 0:
        quality = np.abs(final_desc[:, 1]) + np.abs(final_desc[:, 3])

        cutoff = np.quantile(quality, 1.0 - keep_fraction)
        mask = quality >= cutoff

        n_before = final_desc.shape[0]
        final_desc = final_desc[mask]
        final_idx = final_idx[mask]
        n_after = final_desc.shape[0]

        logger.info(
            f"  Quad quality pruning: kept {n_after:,}/{n_before:,} "
            f"quads (~{100.0 * n_after / n_before:.1f}%) "
            f"using keep_fraction={keep_fraction:.2f}"
        )

    logger.info(
        f"STEP 3 done: {final_desc.shape[0]:,} unique quads "
        f"| time {time.time()-t0:.2f}s | RAM {get_ram():.2f} GB"
    )

    return final_desc, final_idx

def log_diagonal_stats(
    diagonal_buckets: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    logger: logging.Logger,
    sample_limit: int = 5_000_000,
) -> None:
    """Log summary statistics about diagonals and their third points."""
    n_diagonals = len(diagonal_buckets)
    if n_diagonals == 0:
        logger.info("    Diagonal stats: no diagonals found.")
        return

    sizes = np.fromiter(
        (len(v["third"]) for v in diagonal_buckets.values()),
        dtype=np.int32,
        count=n_diagonals
    )

    logger.info(f"    Diagonals: {n_diagonals}")
    logger.info(
        "    Third-points per diagonal: "
        f"min={sizes.min()}, max={sizes.max()}, "
        f"mean={sizes.mean():.1f}, median={np.median(sizes):.1f}, "
        f"95th={np.percentile(sizes, 95):.1f}"
    )

    unique_sizes, counts = np.unique(sizes, return_counts=True)
    if unique_sizes.size <= 10:
        size_str = ", ".join(
            f"{s}: {c}" for s, c in zip(unique_sizes, counts)
        )
        logger.info(f"    Size histogram (exact): {size_str}")
    else:
        bins = [0, 100, 200, 400, 600, 800, 1000, 2000, 5000, 10000]
        hist, edges = np.histogram(sizes, bins=bins)
        bin_str = ", ".join(
            f"[{int(edges[i])},{int(edges[i+1])}): {hist[i]}"
            for i in range(len(hist))
            if hist[i] > 0
        )
        logger.info(f"    Size histogram (binned): {bin_str}")

    sampled_heights = []
    sampled_areas = []
    sampled_proj = []

    approx_per_diag = max(1, sample_limit // n_diagonals)

    has_proj = any("proj" in v for v in diagonal_buckets.values())

    for v in diagonal_buckets.values():
        h = v["height"]
        a = v["area"]
        m = h.shape[0]

        if m <= approx_per_diag:
            idx = slice(None)
        else:
            idx = np.random.choice(m, size=approx_per_diag, replace=False)

        sampled_heights.append(h[idx])
        sampled_areas.append(a[idx])

        if has_proj and "proj" in v:
            p = v["proj"]
            sampled_proj.append(p[idx])

    heights = np.concatenate(sampled_heights, axis=0)
    areas = np.concatenate(sampled_areas, axis=0)
    logger.info(
        f"    Sampled {heights.size:,} (height, area) pairs for global stats"
    )

    def _log_percentiles(name: str, arr: np.ndarray) -> None:
        p = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        logger.info(
            f"    {name} percentiles "
            f"(min,p1,p5,p25,p50,p75,p95,p99,max): "
            f"{p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g}, {p[3]:.4g}, "
            f"{p[4]:.4g}, {p[5]:.4g}, {p[6]:.4g}, {p[7]:.4g}, {p[8]:.4g}"
        )

    _log_percentiles("Triangle HEIGHT", heights)
    _log_percentiles("Triangle AREA", areas)

    if heights.size > 1:
        corr = np.corrcoef(heights, areas)[0, 1]
        logger.info(f"    Corr(height, area): {corr:.4f}")

    if has_proj and sampled_proj:
        proj = np.concatenate(sampled_proj, axis=0)
        _log_percentiles("Projection ALONG diagonal", proj)

        if proj.size == heights.size and proj.size > 1:
            corr_hp = np.corrcoef(heights, proj)[0, 1]
            logger.info(f"    Corr(height, proj): {corr_hp:.4f}")

def _session_worker(args):
    """Thin wrapper around process_single_session for ProcessPoolExecutor."""
    import logging

    session_info, config_dict = args
    cfg = PipelineConfig.from_dict(config_dict)
    
    # Setup logging for this worker process
    log_dir = Path(cfg.output_dir) / "logs_step1"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, verbose=cfg.verbose)
    
    child_logger = logging.getLogger("neuron_mapping_parallel")

    child_logger.info(
        f"[CHILD PID={multiprocessing.current_process().pid}] "
        f"Starting session {session_info['session_name']}..."
    )

    result = process_single_session(session_info, cfg)

    if "session_name" not in result:
        result["session_name"] = session_info.get("session_name", "<unknown>")

    child_logger.info(
        f"[CHILD PID={multiprocessing.current_process().pid}] "
        f"Finished session {result['session_name']} | "
        f"skipped={result.get('skipped', False)} | "
        f"n_quads={result.get('n_quads')}"
    )

    return result

def run_step_1_parallel(
    config: PipelineConfig,
    loaded_sessions: Optional[List[Dict[str, Any]]] = None,
    session_callback: Optional[callable] = None
) -> dict:
    """Run Step 1 in parallel across multiple sessions.
    
    Args:
        config: Pipeline configuration
        loaded_sessions: Optional pre-loaded session list (can be session names, file paths, or dicts)
        session_callback: Optional callback function(current, total, time_per_session)
    """
    # Use loaded sessions if provided, otherwise scan for files
    if loaded_sessions is not None:
        # Convert to proper format if needed
        sessions = []
        
        # Pattern for extracting animal_id and session from various formats
        pattern_with_ext = r'(\d+)_(\d+)(?:_.*)?\.npy'  # Allows optional suffix like _A_final
        pattern_without_ext = r'^(\d+)_(\d+)(?:_.*)?$'   # Same for session names
        
        for sess in loaded_sessions:
            if isinstance(sess, str):
                # It's a string - could be full path, filename, or session name
                import re
                
                # Try to extract filename from path
                filename = Path(sess).name
                
                # Try matching with .npy extension first
                match = re.search(pattern_with_ext, filename)
                
                # If no match, try without extension (just the session name)
                if not match:
                    match = re.search(pattern_without_ext, filename)
                
                if match:
                    animal_id = match.group(1)
                    session_id = match.group(2)
                    
                    # Build the full file path
                    # If sess is already a full path with .npy, use it
                    if sess.endswith('.npy') and Path(sess).exists():
                        file_path = sess
                    # Otherwise, construct it from config.input_dir
                    else:
                        file_path = str(Path(config.input_dir) / f"{animal_id}_{session_id}.npy")
                    
                    sessions.append({
                        'file_path': file_path,
                        'animal_id': animal_id,
                        'session': session_id,
                        'session_name': f"{animal_id}_{session_id}"
                    })
                else:
                    logger.warning(f"Skipping file with invalid pattern: {sess}")
            elif isinstance(sess, dict) and 'session_name' in sess:
                # Already in correct format
                sessions.append(sess)
            else:
                logger.warning(f"Skipping invalid session format: {type(sess)}")
        
        logger.info(f"[PARALLEL] Using {len(sessions)} pre-loaded sessions")
    else:
        sessions = find_session_files(config)
        logger.info(f"[PARALLEL] Found {len(sessions)} session files to process")

    if config.animal_id:
        logger.info(f"[PARALLEL] Processing only animal {config.animal_id}")
        # Filter sessions by animal_id
        sessions = [s for s in sessions if s['animal_id'] == config.animal_id]
        logger.info(f"[PARALLEL] Filtered to {len(sessions)} sessions for animal {config.animal_id}")

    if not sessions:
        logger.warning("[PARALLEL] No sessions found; exiting.")
        return {
            "n_sessions": 0,
            "n_skipped": 0,
            "total_quads": 0,
            "results": [],
        }

    max_parallel = compute_max_parallel_sessions(config)
    config_dict = config.to_dict()

    start = time.time()
    results = []
    total_quads = 0
    n_skipped = 0
    session_times = []

    logger.info(
        f"[PARALLEL] Starting parallel Step 1 with up to "
        f"{max_parallel} worker processes...\n"
    )

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = []
        for sess in sessions:
            futures.append(
                executor.submit(_session_worker, (sess, config_dict))
            )

        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                logger.error(
                    f"[PARALLEL] Error in child process: {e}",
                    exc_info=True,
                )
                continue

            results.append(res)

            if res.get("skipped", False):
                n_skipped += 1
            else:
                total_quads += int(res.get("n_quads") or 0)
                if "generation_time" in res:
                    session_times.append(res["generation_time"])

            # Call progress callback if provided
            if session_callback is not None:
                avg_time = np.mean(session_times) if session_times else 0.0
                session_callback(idx, len(sessions), avg_time)

            logger.info(
                f"[PARALLEL] Session {res.get('session_name')} done | "
                f"skipped={res.get('skipped')} | "
                f"n_quads={res.get('n_quads')}"
            )

    elapsed = time.time() - start
    n_processed = len(results) - n_skipped

    logger.info("\n[PARALLEL] Step 1 Parallel Summary:")
    logger.info(f"  Sessions processed: {n_processed}")
    logger.info(f"  Sessions skipped:   {n_skipped}")
    logger.info(f"  Total quads:        {total_quads:,}")
    logger.info(f"  Wall-clock time:    {elapsed:.1f}s")

    summary = {
        "n_sessions": n_processed,
        "n_skipped": n_skipped,
        "total_quads": total_quads,
        "results": results,
        "wall_time_sec": elapsed,
    }
    return summary

def generate_sparse_quads_triangle(centroids, config, logger):
    """Clean 4-step quad generation pipeline."""
    valid_tri, valid_areas = step1_generate_and_filter_triangles(
        centroids,
        min_tri_area=config.min_tri_area,
        logger=logger,
    )

    diagonal_buckets, total_edges = step2_build_diagonal_buckets_chunked(
        centroids,
        valid_tri,
        valid_areas,
        logger=logger,
        chunk_size=config.triangle_chunk_size,
    )

    logger.info(
        f"STEP 2 (wrapper): {len(diagonal_buckets):,} diagonals, "
        f"{total_edges:,} edges"
    )

    if 0.0 < config.diagonal_drop_percentile < 100.0 and len(diagonal_buckets) > 0:
        logger.info(
            f"    Pruning diagonals by geometric length "
            f"(dropping bottom {config.diagonal_drop_percentile:.1f}%)..."
        )

        diag_keys = list(diagonal_buckets.keys())
        diag_lengths = np.empty(len(diag_keys), dtype=np.float32)

        for idx, (d1, d2) in enumerate(diag_keys):
            p1 = centroids[d1]
            p2 = centroids[d2]
            diff = p2 - p1
            diag_lengths[idx] = np.sqrt(np.dot(diff, diff))

        length_thr = float(np.percentile(diag_lengths, config.diagonal_drop_percentile))
        keep_mask = diag_lengths >= length_thr
        kept_keys = [k for k, keep in zip(diag_keys, keep_mask) if keep]

        logger.info(
            f"    Diagonal length {config.diagonal_drop_percentile:.1f}th percentile: {length_thr:.4f}"
        )
        logger.info(
            f"    Keeping {len(kept_keys):,}/{len(diag_keys):,} diagonals "
            f"(dropped {len(diag_keys) - len(kept_keys):,}) by length."
        )

        diagonal_buckets = {k: diagonal_buckets[k] for k in kept_keys}
    else:
        logger.info("    Skipping diagonal length pruning (disabled or no diagonals).")

    logger.info("    Computing diagonal statistics for pruning design...")
    log_diagonal_stats(diagonal_buckets, logger, sample_limit=5_000_000)

    height_thr = compute_height_threshold(
        diagonal_buckets,
        percentile=config.height_percentile,
        sample_limit=5_000_000,
    )

    logger.info("    Computing M_eff (effective third points per diagonal)...")

    sizes_raw = []
    sizes_kept = []

    for v in diagonal_buckets.values():
        h = np.abs(v["height"])
        sizes_raw.append(len(h))
        sizes_kept.append(np.sum(h >= height_thr))

    M_eff_raw = float(np.mean(sizes_raw)) if sizes_raw else 0.0
    M_eff_kept = float(np.mean(sizes_kept)) if sizes_kept else 0.0

    logger.info(
        f"    M_eff (raw):  {M_eff_raw:.2f} third-points per diagonal on average"
    )
    logger.info(
        f"    M_eff (kept): {M_eff_kept:.2f} after applying |height| >= {height_thr:.4f}"
    )
    logger.info(
        f"    => Expected quads/diagonal ≈ "
        f"{0.5 * M_eff_kept * (M_eff_kept - 1):.1f}"
    )
    logger.info("")
    logger.info(
        f"    Using global |height| >= {height_thr:.4f} "
        f"as triangle height filter in Step 3."
    )
    logger.info("")

    logger.info(
        "    Applying height filter to diagonal buckets "
        "and pruning weak diagonals..."
    )

    kept_buckets = {}
    total_edges_kept = 0

    for key, v in diagonal_buckets.items():
        h = np.abs(v["height"])
        mask = (h >= height_thr)

        if mask.sum() < config.min_third_points_per_diagonal:
            continue

        third_kept = v["third"][mask].astype(np.int32)
        area_kept = v["area"][mask].astype(np.float32)
        height_kept = v["height"][mask].astype(np.float32)

        kept_buckets[key] = {
            "third": third_kept,
            "area": area_kept,
            "height": height_kept,
        }
        total_edges_kept += third_kept.shape[0]

    logger.info(
        f"    Pruned diagonals: kept {len(kept_buckets):,}/"
        f"{len(diagonal_buckets):,} with ≥{config.min_third_points_per_diagonal} good "
        f"third-points after |height| filter. Edges kept: {total_edges_kept:,}"
    )

    diagonal_buckets = kept_buckets

    final_desc, final_idx = step3_generate_quads_from_buckets(
        diagonal_buckets,
        centroids,
        config,
        logger,
        min_height=height_thr,
    )

    return final_desc, final_idx