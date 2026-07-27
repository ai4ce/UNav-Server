"""Server-side task: localize_verbose.
Same as localize_only but adds per-candidate MASt3R/PnP sub-stage stats so we
can identify WHICH stage inside batch_local_matching_and_ransac failed.

Lives at /home/unav/Desktop/UNav_socket/core/tasks/debug/localize_verbose.py.
"""
import os, numpy as np
from core.unav_state import localizer


def _to_list(x):
    if x is None: return None
    if hasattr(x, "tolist"): return x.tolist()
    if isinstance(x, (list, tuple)): return [_to_list(v) for v in x]
    if isinstance(x, dict): return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (float, int, str, bool)): return x
    return str(x)


def _instrumented_mast3r(query_img_path, candidates_data, mast3r_matcher,
                        colmap_models, max_nn_dist=20.0, min_inliers=6,
                        max_candidates=10, early_stop_inliers=50):
    from scipy.spatial import cKDTree
    ref_img_names = list(candidates_data.keys())[:max_candidates]
    grouped = {}
    debug = {"n_cands_tried": 0, "per_cand": []}

    db_paths, db_names = [], []
    for name in ref_img_names:
        c = candidates_data[name]
        place, building, floor = c["map_key"]
        db_paths.append(f"/mnt/data/UNav-IO/temp/{place}/{building}/{floor}/perspectives/{name}")
        db_names.append(name)

    if hasattr(mast3r_matcher, "match_batch") and len(db_paths) > 1:
        batch_results = mast3r_matcher.match_batch(query_img_path, db_paths)
    else:
        batch_results = [mast3r_matcher.match_pair(query_img_path, p) for p in db_paths]

    for idx, name in enumerate(db_names):
        debug["n_cands_tried"] += 1
        cand = candidates_data[name]
        ref_frame = cand["frame"]
        result = batch_results[idx]
        cand_dbg = {"name": name}

        if result is None:
            cand_dbg["fail"] = "mast3r_returned_none"
            debug["per_cand"].append(cand_dbg); continue
        q2d, db2d, conf = result
        n_raw = int(len(q2d)) if q2d is not None else 0
        cand_dbg["n_mast3r_matches"] = n_raw
        if q2d is None or n_raw < min_inliers:
            cand_dbg["fail"] = "too_few_mast3r_matches"
            debug["per_cand"].append(cand_dbg); continue

        colmap_2d = ref_frame["points2D_xy"]
        colmap_3d = ref_frame["points3D_xyz"]
        valid_mask = np.array([p is not None for p in colmap_3d])
        valid_idx = np.where(valid_mask)[0]
        cand_dbg["n_colmap_3d_points"] = int(valid_mask.sum())
        if len(valid_idx) < min_inliers:
            cand_dbg["fail"] = "too_few_colmap_3d"
            debug["per_cand"].append(cand_dbg); continue

        valid_2d = colmap_2d[valid_idx]
        valid_3d = np.array([colmap_3d[i] for i in valid_idx])
        tree = cKDTree(valid_2d)
        dists, nn_idx = tree.query(db2d, k=1)
        close_mask = dists < max_nn_dist
        n_nn = int(close_mask.sum())
        cand_dbg["n_nn_within_20px"] = n_nn
        cand_dbg["nn_dist_median_px"] = float(np.median(dists))
        if n_nn < min_inliers:
            cand_dbg["fail"] = "too_few_nn_close"
            debug["per_cand"].append(cand_dbg); continue

        cand_dbg["success_inliers"] = n_nn
        image_points = q2d[close_mask]
        object_points = valid_3d[nn_idx[close_mask]]

        map_key = cand["map_key"]
        if map_key not in grouped:
            grouped[map_key] = {"ips": [], "ops": [], "res": [], "tot": 0}
        grouped[map_key]["ips"].append(image_points)
        grouped[map_key]["ops"].append(object_points)
        grouped[map_key]["tot"] += n_nn
        grouped[map_key]["res"].append({"ref_image_name": name, "map_key": map_key,
                                        "score": cand.get("score", 0),
                                        "inliers": n_nn})
        debug["per_cand"].append(cand_dbg)

        if grouped[map_key]["tot"] >= early_stop_inliers:
            debug["early_stop"] = True
            break

    if not grouped:
        debug["result"] = "no_candidate_passed"
        return None, {"image_points": np.zeros((0,2)), "object_points": np.zeros((0,3))}, [], debug

    best = max(grouped, key=lambda k: grouped[k]["tot"])
    blk = grouped[best]
    ip = np.concatenate(blk["ips"], axis=0)
    op = np.concatenate(blk["ops"], axis=0)
    debug["result"] = "ok"
    debug["best_map_key"] = list(best)
    debug["total_inliers"] = blk["tot"]
    return best, {"image_points": ip, "object_points": op}, blk["res"], debug


def localize_verbose(inputs):
    import time, cv2, tempfile
    img = inputs.get("image")
    if img is None: return {"success": False, "stage": "input", "reason": "no image"}
    top_k = int(inputs.get("top_k", 10))

    # Mirror localize's stages 1-3 manually to reuse feature/VPR then call our instrumented mast3r
    t0 = time.time()
    timings = {}
    try:
        gf, lfd = localizer.extract_query_features(img)
    except Exception as e:
        return {"success": False, "stage": "extract_query_features", "reason": str(e)}
    timings["extract_query_features"] = time.time() - t0; t0 = time.time()

    effective_topk = min(top_k or 50, 10) if localizer.use_mast3r else top_k
    try:
        top_cands = localizer.vpr_retrieve(gf, top_k=effective_topk)
    except Exception as e:
        return {"success": False, "stage": "vpr_retrieve", "reason": str(e)}
    timings["vpr_retrieve"] = time.time() - t0; t0 = time.time()
    if not top_cands:
        return {"success": False, "stage": "vpr_retrieve", "reason": "empty"}

    try:
        cands_data = localizer.get_candidates_data(top_cands)
    except Exception as e:
        return {"success": False, "stage": "get_candidates_data", "reason": str(e)}
    timings["get_candidates_data"] = time.time() - t0; t0 = time.time()

    # Write query image to temp file for MASt3R
    _tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(_tmp.name, img)
    try:
        mast3r_cfg = localizer.config.feature_extraction_config["local_extractor_config"].get("mast3r", {})
        bmk, pnp_pairs, results, debug = _instrumented_mast3r(
            _tmp.name, cands_data, localizer.local_matcher, localizer.all_colmap_models,
            max_nn_dist=mast3r_cfg.get("max_nn_dist", 20.0),
            min_inliers=localizer.config.localization_config.get("min_inliers", 6),
        )
    finally:
        try: os.unlink(_tmp.name)
        except: pass
    timings["batch_local_matching_and_ransac"] = time.time() - t0

    return {
        "success": bmk is not None,
        "stage": "batch_local_matching_and_ransac" if bmk is None else None,
        "reason": None if bmk is not None else "No candidates passed",
        "mast3r_debug": _to_list(debug),
        "top_candidates": _to_list(top_cands),
        "timings": timings,
    }


DEBUG_TASKS = {"localize_verbose": localize_verbose}
