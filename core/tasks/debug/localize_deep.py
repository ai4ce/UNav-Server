"""Server-side task: localize_deep (copy to /home/unav/Desktop/UNav_socket/core/tasks/debug/).

Runs localizer.localize() and returns the FULL internal pipeline state needed to
diagnose PnP drift:

  3-stage match counts per candidate:
    n_mast3r_raw_matches   — total MASt3R 2D-2D matches before any filter
    n_post_ransac_filter   — survived essential/fundamental-matrix RANSAC (2D-2D)
    n_post_3d_validity     — further kept: DB keypoint has a valid 3D point
    img_pts / obj_pts      — the Stage-B' inliers (input to PnP RANSAC)

  Stage-C: PnP RANSAC inliers (top-level, from pnp_debug):
    pnp_n_input_pairs, pnp_n_inliers_ransac, pnp_inlier_ratio
    pnp_inlier_image_points, pnp_inlier_object_points

All numpy -> list so FastAPI can JSON-serialize.
"""
from core.unav_state import localizer
import numpy as np
import cv2, time, tempfile, os


def _to_list(x):
    if x is None: return None
    if hasattr(x, "tolist"): return x.tolist()
    if isinstance(x, (list, tuple)): return [_to_list(v) for v in x]
    if isinstance(x, dict): return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (float, int, str, bool)): return x
    return str(x)


def localize_deep(inputs):
    img = inputs.get("image")
    if img is None or not isinstance(img, np.ndarray):
        return {"success": False, "stage": "input", "reason": "missing or invalid 'image'"}
    top_k = int(inputs.get("top_k", 10))
    H, W = img.shape[:2]

    t_total = time.time()
    resp = localizer.localize(img, {}, top_k=top_k)

    # Extract per-candidate match stats from resp["results"]
    per_candidate = []
    results = resp.get("results") or []
    for c in results:
        ip = c.get("image_points")
        op = c.get("object_points")
        debug = c.get("debug") or {}
        rec = {
            "ref_image_name": c.get("ref_image_name"),
            "map_key": _to_list(c.get("map_key")),
            "score": float(c.get("score")) if c.get("score") is not None else None,
            # Legacy fields
            "inliers": int(c.get("inliers")) if c.get("inliers") is not None else None,
            # Stage counts from MASt3R pipeline:
            #   raw -> NN lookup against COLMAP 2D -> distance filter -> 3D valid
            "n_mast3r_raw_matches":  debug.get("n_mast3r_raw_matches"),
            "n_colmap_3d_valid":     debug.get("n_colmap_3d_valid"),
            "n_nn_close":            debug.get("n_nn_close"),
            "n_post_3d_validity":    debug.get("n_post_3d_validity"),
            # Legacy fields from batch_local path (may also be present)
            "n_post_ransac_filter":  debug.get("n_post_ransac_filter"),
        }
        if ip is not None and len(ip) > 0:
            ip = np.asarray(ip)
            rec["img_pts"] = _to_list(ip)
            rec["img_pts_bbox"] = [float(ip[:, 0].min()), float(ip[:, 1].min()),
                                    float(ip[:, 0].max()), float(ip[:, 1].max())]
            rec["img_pts_mean"] = [float(ip[:, 0].mean()), float(ip[:, 1].mean())]
            rec["img_pts_std"]  = [float(ip[:, 0].std()),  float(ip[:, 1].std())]
        if op is not None and len(op) > 0:
            op = np.asarray(op)
            rec["obj_pts"] = _to_list(op)
            rec["obj_depth_mean"] = float(np.linalg.norm(op, axis=1).mean())
            rec["obj_depth_std"]  = float(np.linalg.norm(op, axis=1).std())
            rec["obj_z_mean"] = float(op[:, 2].mean())
            rec["obj_z_std"]  = float(op[:, 2].std())
        per_candidate.append(rec)

    # Stage C: PnP RANSAC inliers (if resp carries it through)
    # refine_pose_from_queue populates "pnp_debug", which localize() doesn't currently
    # forward. So we also introspect via refinement_queue if present.
    pnp_debug = None
    if resp.get("pnp_debug"):
        pnp_debug = resp["pnp_debug"]
    else:
        # Reach into refinement queue: last entry's pairs were set to PnP inliers
        rq = resp.get("refinement_queue") or {}
        # refinement_queue is {map_key: {pairs:[...]}} where pairs[-1] is current frame
        if resp.get("best_map_key"):
            mq = rq.get(resp["best_map_key"]) or {}
            prs = mq.get("pairs") or []
            if prs:
                cur = prs[-1]
                ip = cur.get("image_points"); op = cur.get("object_points")
                if ip is not None:
                    pnp_debug = {
                        "n_pnp_inliers_ransac": int(len(ip)),
                        "pnp_inlier_image_points": _to_list(ip),
                        "pnp_inlier_object_points": _to_list(op),
                    }

    return {
        "success": bool(resp.get("success", False)),
        "stage": resp.get("stage"),
        "reason": resp.get("reason"),
        "qvec": _to_list(resp.get("qvec")),
        "tvec": _to_list(resp.get("tvec")),
        "floorplan_pose": _to_list(resp.get("floorplan_pose")),
        "best_map_key": _to_list(resp.get("best_map_key")),
        "top_candidates": _to_list(resp.get("top_candidates")),
        "n_frames": resp.get("n_frames"),
        "timings": _to_list(resp.get("timings")),
        "query_hw": [int(H), int(W)],
        "per_candidate": per_candidate,
        "pnp_debug": _to_list(pnp_debug),
        "total_time_s": time.time() - t_total,
    }


DEBUG_TASKS = {"localize_deep": localize_deep}
