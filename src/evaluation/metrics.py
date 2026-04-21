import gzip
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# ============================================================
# Basic COLMAP parsing
# ============================================================

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)


def parse_all_cameras(camera_file):
    cams = {}

    with open(camera_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params
                fx = fy = f
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ], dtype=np.float64)

            cams[camera_id] = {
                "K": K,
                "width": width,
                "height": height,
                "model": model,
            }

    return cams


def parse_all_images(image_file):
    result = {}

    with open(image_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]

            qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
            tvec = np.array([tx, ty, tz], dtype=np.float64)
            R = qvec2rotmat(qvec)

            result[name] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "R": R,
                "camera_id": camera_id,
            }

            i += 2
        else:
            i += 1

    return result


# ============================================================
# CO3D parsing
# ============================================================

def load_co3d_annotations(jgz_path):
    with gzip.open(jgz_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_co3d_frame_dict(frame_data):
    frames = {}

    for item in frame_data:
        image_name = os.path.basename(item["image"]["path"])

        frames[image_name] = {
            "image_rel_path": item["image"]["path"],
            "image_size": item["image"]["size"],
            "mask_rel_path": item["mask"]["path"],
            "depth_rel_path": item["depth"]["path"],
            "depth_mask_rel_path": item["depth"]["mask_path"],
            "depth_scale_adjustment": float(item["depth"]["scale_adjustment"]),
        }

    return frames


# ============================================================
# Loading from flattened subset layout
# ============================================================

def load_rgb_image(dataset_root, rel_path):
    filename = os.path.basename(rel_path)
    path = dataset_root / "images" / filename
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_mask_soft(dataset_root, rel_path):
    filename = os.path.basename(rel_path)
    path = dataset_root / "masks" / filename
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask.astype(np.float32) / 255.0


def load_depth(dataset_root, rel_path, scale_adjustment=1.0):
    filename = os.path.basename(rel_path)
    path = dataset_root / "depths" / filename
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth: {path}")

    depth = depth.astype(np.float32)
    depth *= scale_adjustment
    return depth


def load_depth_mask_soft(dataset_root, rel_path):
    filename = os.path.basename(rel_path)
    path = dataset_root / "depth_masks" / filename
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read depth mask: {path}")
    return mask.astype(np.float32) / 255.0


# ============================================================
# Resize helpers
# ============================================================

def resize_rgb_to_shape(img, H, W):
    if img.shape[0] == H and img.shape[1] == W:
        return img
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


def resize_soft_mask_to_shape(mask, H, W):
    if mask.shape[0] == H and mask.shape[1] == W:
        return mask.astype(np.float32)
    return cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def resize_depth_to_shape(depth, H, W):
    if depth.shape[0] == H and depth.shape[1] == W:
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def binarize_soft_mask(mask_soft, threshold=0.5):
    return (mask_soft >= threshold).astype(np.uint8)


# ============================================================
# Rendering
# ============================================================

def voxel_downsample_points(points, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down.points)


def render_depth_and_mask(points_world, K, R, t, H, W, point_radius_px=1):
    pts_cam = (R @ points_world.T + t.reshape(3, 1)).T
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]

    if pts_cam.shape[0] == 0:
        depth = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.uint8)
        return depth, mask

    proj = (K @ pts_cam.T).T
    uv = proj[:, :2] / pts_cam[:, 2:3]
    z = pts_cam[:, 2]

    depth = np.full((H, W), np.inf, dtype=np.float32)

    xs = np.round(uv[:, 0]).astype(int)
    ys = np.round(uv[:, 1]).astype(int)

    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs = xs[inside]
    ys = ys[inside]
    z = z[inside]

    radius = max(0, int(point_radius_px))

    for x, y, zz in zip(xs, ys, z):
        x0 = max(0, x - radius)
        x1 = min(W - 1, x + radius)
        y0 = max(0, y - radius)
        y1 = min(H - 1, y + radius)

        patch = depth[y0:y1 + 1, x0:x1 + 1]
        np.minimum(patch, zz, out=patch)

    mask = np.isfinite(depth).astype(np.uint8)
    depth[~np.isfinite(depth)] = 0.0

    return depth, mask


def clean_mask(mask, dilate_kernel=3, dilate_iterations=1):
    out = (mask > 0).astype(np.uint8)

    if dilate_kernel and dilate_kernel > 1 and dilate_iterations > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        out = cv2.dilate(out, kernel, iterations=dilate_iterations)

    return out


# ============================================================
# Metrics
# ============================================================

def compute_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def fit_depth_scale(pred_depth, gt_depth, valid_mask):
    valid = valid_mask.astype(bool)
    valid &= pred_depth > 0
    valid &= gt_depth > 0
    valid &= np.isfinite(gt_depth)

    if valid.sum() == 0:
        return np.nan

    p = pred_depth[valid]
    g = gt_depth[valid]

    denom = np.sum(p * p)
    if denom <= 1e-12:
        return np.nan

    alpha = np.sum(p * g) / denom
    return float(alpha)


def compute_depth_mae_scaled(pred_depth, gt_depth, depth_valid_mask):
    alpha = fit_depth_scale(pred_depth, gt_depth, depth_valid_mask)

    if np.isnan(alpha):
        return np.nan, np.nan, pred_depth.copy()

    pred_scaled = pred_depth * alpha

    valid = depth_valid_mask.astype(bool)
    valid &= pred_scaled > 0
    valid &= gt_depth > 0
    valid &= np.isfinite(gt_depth)

    if valid.sum() == 0:
        return np.nan, alpha, pred_scaled

    mae = np.mean(np.abs(pred_scaled[valid] - gt_depth[valid]))
    return float(mae), float(alpha), pred_scaled


# ============================================================
# Visualization helpers
# ============================================================

def normalize_depth_for_vis(depth, valid_mask=None):
    out = np.zeros_like(depth, dtype=np.uint8)

    if valid_mask is None:
        valid = depth > 0
    else:
        valid = valid_mask.astype(bool) & (depth > 0)

    if np.any(valid):
        vals = depth[valid]
        dmin = vals.min()
        dmax = vals.max()
        if dmax > dmin:
            norm = np.zeros_like(depth, dtype=np.float32)
            norm[valid] = (depth[valid] - dmin) / (dmax - dmin)
            out = (norm * 255).astype(np.uint8)

    return out


def make_color_depth(depth, valid_mask=None):
    gray = normalize_depth_for_vis(depth, valid_mask)
    return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)


def make_depth_error_map(pred_depth, gt_depth, valid_mask):
    err = np.zeros_like(pred_depth, dtype=np.float32)
    valid = valid_mask.astype(bool)
    valid &= pred_depth > 0
    valid &= gt_depth > 0
    valid &= np.isfinite(gt_depth)

    if np.any(valid):
        err[valid] = np.abs(pred_depth[valid] - gt_depth[valid])

    if np.any(valid):
        vmax = np.percentile(err[valid], 95)
        if vmax <= 0:
            vmax = err[valid].max() if err[valid].max() > 0 else 1.0
    else:
        vmax = 1.0

    vis = np.clip(err / vmax, 0, 1)
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    vis[~valid] = 0
    return vis


def draw_mask_contours(rgb, pred_mask, gt_mask):
    out = rgb.copy()

    pred_u8 = (pred_mask > 0).astype(np.uint8) * 255
    gt_u8 = (gt_mask > 0).astype(np.uint8) * 255

    pred_contours, _ = cv2.findContours(pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(out, gt_contours, -1, (0, 0, 255), 2)
    cv2.drawContours(out, pred_contours, -1, (0, 255, 0), 2)

    return out


def make_overlay(rgb, pred_mask, gt_mask=None):
    out = rgb.copy()

    pred = pred_mask.astype(bool)
    out[pred] = (0.65 * out[pred] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)

    if gt_mask is not None:
        gt = gt_mask.astype(bool)
        out[gt] = (0.65 * out[gt] + 0.35 * np.array([0, 0, 255])).astype(np.uint8)

    return out


def add_title_bar(img, title, height=36):
    h, w = img.shape[:2]
    bar = np.full((height, w, 3), 245, dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (10, int(height * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, img])


def stack_h(images):
    return np.hstack(images)


def stack_v(images):
    return np.vstack(images)


def save_debug_outputs(
    out_dir,
    frame_name,
    rgb,
    pred_mask_raw,
    pred_mask_clean,
    gt_mask,
    pred_depth,
    pred_depth_scaled,
    gt_depth,
    gt_depth_mask,
    mask_iou,
    depth_mae,
    depth_scale,
):
    frame_dir = out_dir / frame_name.replace(".jpg", "")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    overlay_pred = make_overlay(rgb, pred_mask_clean, None)
    overlay_both = make_overlay(rgb, pred_mask_clean, gt_mask)
    contour_compare = draw_mask_contours(rgb, pred_mask_clean, gt_mask)

    pred_depth_color = make_color_depth(pred_depth_scaled, gt_depth_mask)
    gt_depth_color = make_color_depth(gt_depth, gt_depth_mask)
    depth_err_color = make_depth_error_map(pred_depth_scaled, gt_depth, gt_depth_mask)

    pred_mask_vis = cv2.cvtColor(pred_mask_clean * 255, cv2.COLOR_GRAY2BGR)
    gt_mask_vis = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(frame_dir / "rgb.png"), rgb)
    cv2.imwrite(str(frame_dir / "pred_mask_raw.png"), pred_mask_raw * 255)
    cv2.imwrite(str(frame_dir / "pred_mask_clean.png"), pred_mask_clean * 255)
    cv2.imwrite(str(frame_dir / "gt_mask.png"), gt_mask * 255)
    cv2.imwrite(str(frame_dir / "overlay_pred.png"), overlay_pred)
    cv2.imwrite(str(frame_dir / "overlay_pred_gt.png"), overlay_both)
    cv2.imwrite(str(frame_dir / "contours_pred_vs_gt.png"), contour_compare)
    cv2.imwrite(str(frame_dir / "pred_depth_color.png"), pred_depth_color)
    cv2.imwrite(str(frame_dir / "gt_depth_color.png"), gt_depth_color)
    cv2.imwrite(str(frame_dir / "depth_error_map.png"), depth_err_color)

    fig1 = stack_h([
        add_title_bar(rgb, "Original RGB"),
        add_title_bar(overlay_pred, "Projected Prediction Overlay"),
    ])
    cv2.imwrite(str(frame_dir / "figure1_overlay.png"), fig1)

    fig2 = stack_h([
        add_title_bar(pred_mask_vis, "Predicted Mask"),
        add_title_bar(gt_mask_vis, "Ground-Truth Mask"),
        add_title_bar(contour_compare, f"Contours (IoU = {mask_iou:.4f})"),
    ])
    cv2.imwrite(str(frame_dir / "figure2_masks.png"), fig2)

    depth_title = f"Depth Error (MAE = {depth_mae:.4f}, scale = {depth_scale:.4f})" if not np.isnan(depth_mae) and not np.isnan(depth_scale) else "Depth Error"
    fig3 = stack_h([
        add_title_bar(pred_depth_color, "Predicted Depth (Scaled)"),
        add_title_bar(gt_depth_color, "Ground-Truth Depth"),
        add_title_bar(depth_err_color, depth_title),
    ])
    cv2.imwrite(str(frame_dir / "figure3_depth.png"), fig3)

    top = stack_h([
        add_title_bar(rgb, "RGB"),
        add_title_bar(overlay_both, "Pred + GT Overlay"),
    ])
    bottom = stack_h([
        add_title_bar(contour_compare, f"Mask Comparison (IoU = {mask_iou:.4f})"),
        add_title_bar(depth_err_color, depth_title),
    ])
    fig4 = stack_v([top, bottom])
    cv2.imwrite(str(frame_dir / "figure4_summary.png"), fig4)


# ============================================================
# Main evaluation
# ============================================================

def run_co3d_reprojection_metrics(config, project_root):
    dataset_cfg = config["dataset"]
    pred_cfg = config["prediction"]
    eval_cfg = config["evaluation"]

    dataset_root = project_root / dataset_cfg["root_dir"]
    annotations_file = project_root / dataset_cfg["annotations_file"]

    point_cloud_path = project_root / pred_cfg["point_cloud_path"]
    images_txt = project_root / pred_cfg["images_txt"]
    cameras_txt = project_root / pred_cfg["cameras_txt"]

    out_dir = project_root / eval_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_frame = eval_cfg.get("debug_frame")
    voxel_size = eval_cfg.get("voxel_size", 0.0)
    point_radius_px = eval_cfg.get("point_radius_px", 1)
    dilate_kernel = eval_cfg.get("dilate_kernel", 3)
    dilate_iterations = eval_cfg.get("dilate_iterations", 1)
    compute_mask_iou_flag = bool(eval_cfg.get("compute_mask_iou", True))
    compute_depth_mae_flag = bool(eval_cfg.get("compute_depth_mae", True))
    save_debug_images = bool(eval_cfg.get("save_debug_images", True))
    max_frames = eval_cfg.get("max_frames", None)

    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    points = np.asarray(pcd.points)
    print(f"Original points: {points.shape[0]}")

    points = voxel_downsample_points(points, voxel_size)
    print(f"Downsampled points: {points.shape[0]}")

    print("Loading COLMAP cameras/images...")
    cameras = parse_all_cameras(cameras_txt)
    images = parse_all_images(images_txt)

    print("Loading CO3D annotations...")
    frame_data = load_co3d_annotations(annotations_file)
    co3d_frames = build_co3d_frame_dict(frame_data)

    common_frames = sorted(set(images.keys()) & set(co3d_frames.keys()))
    if max_frames is not None:
        common_frames = common_frames[:int(max_frames)]

    print(f"Frames available for evaluation: {len(common_frames)}")

    results = []

    for idx, frame_name in enumerate(common_frames):
        pred_img = images[frame_name]
        co3d = co3d_frames[frame_name]

        camera_id = pred_img["camera_id"]
        cam = cameras[camera_id]

        K = cam["K"]
        W = cam["width"]
        H = cam["height"]
        R = pred_img["R"]
        t = pred_img["tvec"]

        try:
            rgb = load_rgb_image(dataset_root, co3d["image_rel_path"])
            gt_mask_soft = load_mask_soft(dataset_root, co3d["mask_rel_path"])
            gt_depth = load_depth(
                dataset_root,
                co3d["depth_rel_path"],
                co3d["depth_scale_adjustment"],
            )
            gt_depth_mask_soft = load_depth_mask_soft(dataset_root, co3d["depth_mask_rel_path"])
        except FileNotFoundError as e:
            print(f"[skip] {frame_name}: {e}")
            continue

        rgb = resize_rgb_to_shape(rgb, H, W)
        gt_mask_soft = resize_soft_mask_to_shape(gt_mask_soft, H, W)
        gt_depth = resize_depth_to_shape(gt_depth, H, W)
        gt_depth_mask_soft = resize_soft_mask_to_shape(gt_depth_mask_soft, H, W)

        gt_mask = binarize_soft_mask(gt_mask_soft, threshold=0.5)
        gt_depth_mask = binarize_soft_mask(gt_depth_mask_soft, threshold=0.5)

        pred_depth, pred_mask_raw = render_depth_and_mask(
            points_world=points,
            K=K,
            R=R,
            t=t,
            H=H,
            W=W,
            point_radius_px=point_radius_px,
        )

        pred_mask = clean_mask(
            pred_mask_raw,
            dilate_kernel=dilate_kernel,
            dilate_iterations=dilate_iterations,
        )

        row = {"frame": frame_name}

        if compute_mask_iou_flag:
            row["mask_iou"] = compute_iou(pred_mask, gt_mask)
        else:
            row["mask_iou"] = np.nan

        if compute_depth_mae_flag:
            depth_mae, depth_scale, pred_depth_scaled = compute_depth_mae_scaled(
                pred_depth=pred_depth,
                gt_depth=gt_depth,
                depth_valid_mask=gt_depth_mask,
            )
            row["depth_mae"] = depth_mae
            row["depth_scale"] = depth_scale
        else:
            pred_depth_scaled = pred_depth.copy()
            row["depth_mae"] = np.nan
            row["depth_scale"] = np.nan

        results.append(row)

        if save_debug_images and (frame_name == debug_frame):
            print(f"Debug frame metrics for {frame_name}:")
            print(f"  mask_iou   = {row['mask_iou']}")
            print(f"  depth_mae  = {row['depth_mae']}")
            print(f"  depth_scale= {row['depth_scale']}")

            save_debug_outputs(
                out_dir=out_dir,
                frame_name=frame_name,
                rgb=rgb,
                pred_mask_raw=pred_mask_raw,
                pred_mask_clean=pred_mask,
                gt_mask=gt_mask,
                pred_depth=pred_depth,
                pred_depth_scaled=pred_depth_scaled,
                gt_depth=gt_depth,
                gt_depth_mask=gt_depth_mask,
                mask_iou=row["mask_iou"],
                depth_mae=row["depth_mae"],
                depth_scale=row["depth_scale"],
            )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(common_frames):
            print(f"Processed {idx + 1}/{len(common_frames)}")

    if not results:
        print("No valid frames were evaluated.")
        return

    csv_path = out_dir / "per_frame_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,mask_iou,depth_mae,depth_scale\n")
        for row in results:
            f.write(f"{row['frame']},{row['mask_iou']},{row['depth_mae']},{row['depth_scale']}\n")

    mask_ious = [r["mask_iou"] for r in results if not np.isnan(r["mask_iou"])]
    depth_maes = [r["depth_mae"] for r in results if not np.isnan(r["depth_mae"])]
    depth_scales = [r["depth_scale"] for r in results if not np.isnan(r["depth_scale"])]

    summary = {
        "num_frames": len(results),
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else np.nan,
        "median_mask_iou": float(np.median(mask_ious)) if mask_ious else np.nan,
        "mean_depth_mae": float(np.mean(depth_maes)) if depth_maes else np.nan,
        "median_depth_mae": float(np.median(depth_maes)) if depth_maes else np.nan,
        "mean_depth_scale": float(np.mean(depth_scales)) if depth_scales else np.nan,
        "median_depth_scale": float(np.median(depth_scales)) if depth_scales else np.nan,
    }

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\nDone.")
    print(f"Saved per-frame metrics to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")