import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from matplotlib import cm
from matplotlib.colors import ListedColormap

# ───────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────
LOWE_RATIO               = 0.7
ACCUMULATOR_THRESHOLD    = 1       # we threshold “> 0” for raw_centers
VOTE_RADIUS              = 100
RANSAC_REPROJ_THRESHOLD  = 5.0
MIN_DISTANCE             = 300

GLOBAL_MIN_MATCH_PERCENT = 0.25
GLOBAL_MIN_MATCH_ABS     = 70

LOCAL_MIN_MATCH_PERCENT  = 0.05
LOCAL_MIN_MATCH_ABS      = 18      # already lowered for weak local matches

CLUSTER_RADIUS           = 20      # pixels for grouping raw_centers
MIN_CLUSTER_SIZE         = 5       # raw_centers needed to form a valid cluster

# ───────────────────────────────────────────────────────────────────────────────
def merge_barycenters(centers, min_distance=MIN_DISTANCE):
    """
    Merge barycenters closer than min_distance into one,
    keeping distinct barycenters separated.
    """
    merged = []
    for c in centers:
        if all(np.linalg.norm(np.array(c) - np.array(m)) > min_distance for m in merged):
            merged.append(c)
    return merged

def cluster_raw_centers(raw_centers, radius=CLUSTER_RADIUS, min_size=MIN_CLUSTER_SIZE, log_f=None):
    """
    Group raw_centers (each a (cy, cx) tuple) into clusters via a simple
    seed+neighborhood approach. Any cluster with ≥ min_size points whose
    pairwise distance ≤ radius is kept.
    Returns a list of (cy, cx) centroids for each surviving cluster.
    """
    pts = raw_centers.copy()
    clusters = []

    while pts:
        seed = pts[0]
        seed_y, seed_x = seed

        # Find all indices in pts within 'radius' of the seed
        group_indices = [
            i for i, (yy, xx) in enumerate(pts)
            if np.hypot(yy - seed_y, xx - seed_x) <= radius
        ]

        if len(group_indices) >= min_size:
            # Compute centroid of these group points
            group_pts = [pts[i] for i in group_indices]
            cy_cluster = float(np.mean([yy for yy, xx in group_pts]))
            cx_cluster = float(np.mean([xx for yy, xx in group_pts]))
            clusters.append((cy_cluster, cx_cluster))

            if log_f is not None:
                print(f"\t[CLUSTER] seed=({seed_x},{seed_y}), size={len(group_indices)} "
                      f"→ centroid=({int(cx_cluster)},{int(cy_cluster)})", file=log_f)

            # Remove all group members from pts
            for idx in sorted(group_indices, reverse=True):
                pts.pop(idx)
        else:
            # Drop this seed (too few neighbors)
            if log_f is not None:
                print(f"\t[DROP-CLUSTER] Raw center ({seed_x},{seed_y}) alone: "
                      f"only {len(group_indices)} < {min_size}", file=log_f)
            pts.pop(0)

    return clusters

# ───────────────────────────────────────────────────────────────────────────────
def detect_model_in_scene(model_img_path, scene_img_path, log_f):
    """
    1) Load model and scene
    2) SIFT match → accumulator votes
    3) Connected‐component finding of accumulator>0 → raw_centers
    4) Proximity‐clustering of raw_centers → merged centers
    5) For each center: local‐match + RANSAC check
    6) If model is blue‐dominant, run an HSV blue‐fraction check on each ROI
    7) Draw boxes for surviving candidates
    8) Plot accumulator heatmap + final detections
    """
    print(f"\n🔎  Detecting model: {model_img_path} in scene: {scene_img_path}", file=log_f)

    # 1) LOAD & RESIZE
    model_img = cv2.imread(model_img_path)
    if model_img is None:
        print(f"[ERROR] Failed to load model image {model_img_path}", file=log_f)
        return
    scene_img = cv2.imread(scene_img_path)
    if scene_img is None:
        print(f"[ERROR] Failed to load scene image {scene_img_path}", file=log_f)
        return

    # Resize model to a fixed size (180×240) for consistent voting
    model_img = cv2.resize(model_img, (180, 240))
    model_img_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
    scene_img_rgb = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
    model_gray = cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1.a) DETERMINE IF MODEL IS “BLUE‐DOMINANT” (for conditional color check)
    # ─────────────────────────────────────────────────────────────────────────────
    # Convert model to HSV
    model_hsv = cv2.cvtColor(model_img, cv2.COLOR_BGR2HSV)
    # Mask out low‐saturation pixels (white/cream/highlight)
    S_THRESHOLD_MODEL = 50
    mask_sat_model = model_hsv[:, :, 1] >= S_THRESHOLD_MODEL
    sat_hues_model = model_hsv[:, :, 0][mask_sat_model]
    if sat_hues_model.size == 0:
        # If model has no sufficiently saturated pixels, definitely not “blue”
        do_color_check = False
        print(f"\t[MODEL‐COLOR] Model has no saturated pixels → skipping color check", file=log_f)
    else:
        # Count fraction of saturated pixels whose hue ∈ [100,135] (blue range)
        HUE_BLUE_MIN = 100
        HUE_BLUE_MAX = 135
        num_blue_model = np.count_nonzero((sat_hues_model >= HUE_BLUE_MIN) & (sat_hues_model <= HUE_BLUE_MAX))
        frac_blue_model = float(num_blue_model) / float(sat_hues_model.size)
        MODEL_BLUE_FRACTION_THRESHOLD = 0.30
        do_color_check = (frac_blue_model >= MODEL_BLUE_FRACTION_THRESHOLD)
        print(f"\t[MODEL‐COLOR] frac_blue_model={frac_blue_model:.2f} "
              f"(threshold={MODEL_BLUE_FRACTION_THRESHOLD:.2f}) → do_color_check={do_color_check}", file=log_f)

    # 2) SIFT FEATURE DETECTION & DESCRIPTION
    sift = cv2.SIFT_create()
    kp_model, des_model = sift.detectAndCompute(model_gray, None)
    kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)
    if des_model is None or des_scene is None:
        print("[ERROR] No descriptors found in model or scene", file=log_f)
        return

    # 3) FLANN MATCHER + LOWE’S RATIO TEST
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )
    matches = flann.knnMatch(des_model, des_scene, k=2)
    good_matches = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    # 4) ADAPTIVE GLOBAL‐MATCH THRESHOLD
    unique_scene_kp_indices = set(m.trainIdx for m in good_matches)
    num_scene_kps_matched = len(unique_scene_kp_indices)
    adaptive_global_min_matches = max(
        GLOBAL_MIN_MATCH_ABS,
        int(num_scene_kps_matched * GLOBAL_MIN_MATCH_PERCENT)
    )
    if len(good_matches) < adaptive_global_min_matches:
        print(f"\t❌ Not enough good matches: have {len(good_matches)}, need ≥ {adaptive_global_min_matches}", file=log_f)
        return

    # 5) COMPUTE MODEL BARYCENTER & OFFSET VECTORS
    model_pts = np.array([kp.pt for kp in kp_model])
    barycenter = np.mean(model_pts, axis=0)
    vectors = model_pts - barycenter

    votes = []
    accumulator = np.zeros(scene_gray.shape, dtype=np.uint16)

    # 6) CAST VOTES INTO THE ACCUMULATOR
    for match in good_matches:
        i, j = match.queryIdx, match.trainIdx
        pt_scene = np.array(kp_scene[j].pt)
        scale = kp_scene[j].size / kp_model[i].size
        vote = pt_scene - scale * vectors[i]
        x, y = np.round(vote).astype(int)
        if 0 <= x < accumulator.shape[1] and 0 <= y < accumulator.shape[0]:
            accumulator[y, x] += 1
            votes.append((x, y))

    # ─────────────────────────────────────────────────────────────────────────────
    # 7) CONNECTED‐COMPONENTS → raw_centers (any pixel with ≥1 vote)
    # ─────────────────────────────────────────────────────────────────────────────
    binary_mask = (accumulator > 0).astype(np.uint8)
    labeled_mask, num_objects = label(binary_mask)
    raw_centers = center_of_mass(accumulator, labeled_mask, range(1, num_objects + 1))

    # DEBUG: print all raw_centers + their vote counts
    for idx, (cy, cx) in enumerate(raw_centers, start=1):
        cy_i, cx_i = int(cy), int(cx)
        val = int(accumulator[cy_i, cx_i])
        print(f"\t[DEBUG] RAW CENTER #{idx}: (x={cx_i}, y={cy_i}) → accumulator[{cy_i},{cx_i}] = {val}", file=log_f)

    # 8) CLUSTER raw_centers BY PROXIMITY
    filtered_centers = cluster_raw_centers(raw_centers,
                                           radius=CLUSTER_RADIUS,
                                           min_size=MIN_CLUSTER_SIZE,
                                           log_f=log_f)

    # 9) MERGE CLUSTER CENTROIDS → final centers
    centers = merge_barycenters(filtered_centers)

    # DEBUG: print merged centers
    for idx, (cy, cx) in enumerate(centers, start=1):
        cy_i, cx_i = int(cy), int(cx)
        print(f"\t[DEBUG] MERGED CENTER #{idx}: (x={cx_i}, y={cy_i})", file=log_f)

    # Prepare for drawing
    h_model, w_model = model_gray.shape
    model_corners = np.array([[0, 0],
                              [w_model, 0],
                              [w_model, h_model],
                              [0, h_model]],
                             dtype=np.float32).reshape(-1, 1, 2)
    scene_with_boxes = scene_img_rgb.copy()

    # 10) FOR EACH MERGED CENTER: LOCAL‐MATCH + RANSAC + CONDITIONAL COLOR CHECK
    for (cy, cx) in centers:
        cx_i, cy_i = int(cx), int(cy)

        # 10.1) Gather vote indices within VOTE_RADIUS
        nearby_indices = [
            idx for idx, (x0, y0) in enumerate(votes)
            if abs(x0 - cx_i) < VOTE_RADIUS and abs(y0 - cy_i) < VOTE_RADIUS
        ]
        if not nearby_indices:
            print(f"\t[REJECT] Center at (x={cx_i},y={cy_i}): no nearby votes (|nearby_indices|=0).", file=log_f)
            continue

        # 10.2) Build local_matches & check LOCAL thresholds
        local_matches = [good_matches[idx] for idx in nearby_indices if idx < len(good_matches)]
        adaptive_local_min_matches = max(
            LOCAL_MIN_MATCH_ABS,
            int(len(nearby_indices) * LOCAL_MIN_MATCH_PERCENT)
        )
        print(f"\t[CHECK] Center (x={cx_i},y={cy_i}): local_matches={len(local_matches)}, "
              f"threshold={adaptive_local_min_matches}", file=log_f)
        if len(local_matches) < adaptive_local_min_matches:
            print(f"\t[REJECT] Center at (x={cx_i},y={cy_i}): only {len(local_matches)} local matches < {adaptive_local_min_matches}", file=log_f)
            continue

        # 10.3) Attempt homography & count inliers
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in local_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in local_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if H is None or mask is None:
            print(f"\t[REJECT] Center at (x={cx_i},y={cy_i}): homography failed (H is None)", file=log_f)
            continue

        inliers = int(np.sum(mask))
        print(f"\t[CHECK] Center (x={cx_i},y={cy_i}): homography inliers = {inliers}", file=log_f)
        if inliers < 8:
            print(f"\t[REJECT] Center at (x={cx_i},y={cy_i}): only {inliers} inliers < 8", file=log_f)
            continue

        # 10.4) Project corners → axis-aligned bounding box
        projected_corners = cv2.perspectiveTransform(model_corners, H).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(projected_corners.astype(np.float32))

        # Clip bounding box to image boundaries
        h_scene, w_scene = scene_gray.shape
        x = np.clip(x, 0, w_scene - 1)
        y = np.clip(y, 0, h_scene - 1)
        w = min(w, w_scene - x)
        h = min(h, h_scene - y)

        # ─────────────────────────────────────────────────────────────────────────
        # 10.5) CONDITIONAL HSV “BLUE‐VS‐ORANGE” CHECK
        # Only run if the model was blue‐dominant
        if do_color_check:
            # Extract ROI in BGR (because that’s what cv2.cvtColor expects)
            roi_bgr = scene_img[y : y + h, x : x + w]
            if roi_bgr.size == 0:
                # If bounding box is off‐image, skip
                print(f"\t[REJECT] Box at (x={x},y={y}) fell outside image bounds.", file=log_f)
                continue

            # Convert ROI → HSV
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            # Mask out low‐saturation pixels (white / cream / specular)
            S_THRESHOLD = 50
            mask_sat = (roi_hsv[:, :, 1] >= S_THRESHOLD)
            sat_hues = roi_hsv[:, :, 0][mask_sat]

            if sat_hues.size == 0:
                # No pixel is sufficiently colored → reject
                print(f"\t[REJECT‐COLOR] ROI at (x={x},y={y}) has no saturated pixels; likely not blue.", file=log_f)
                continue

            # Count fraction of saturated pixels whose Hue ∈ [100,135] (blue range)
            HUE_BLUE_MIN = 100
            HUE_BLUE_MAX = 135
            num_blue_pixels = np.count_nonzero((sat_hues >= HUE_BLUE_MIN) & (sat_hues <= HUE_BLUE_MAX))
            frac_blue = float(num_blue_pixels) / float(sat_hues.size)

            BLUE_FRACTION_THRESHOLD = 0.30
            if frac_blue < BLUE_FRACTION_THRESHOLD:
                print(f"\t[REJECT‐COLOR] Box at (x={x},y={y}): frac_blue={frac_blue:.2f} "
                      f"< {BLUE_FRACTION_THRESHOLD:.2f}", file=log_f)
                continue

            # If we reach here, this candidate passed the blue‐fraction test:
            print(f"\t[PASS‐COLOR] Box at (x={x},y={y}): frac_blue={frac_blue:.2f}", file=log_f)

        # Otherwise, either do_color_check is False, or it passed → accept box
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(scene_with_boxes, top_left, bottom_right, (0, 255, 0), 3)
        cv2.circle(scene_with_boxes, (cx_i, cy_i), 5, (255, 0, 0), -1)

        print(f"\t☑️  Detected model {os.path.basename(model_img_path)} at (x={cx_i},y={cy_i}), "
              f"box=[{top_left}→{bottom_right}], inliers={inliers}", file=log_f)

    # 11) Plot accumulator heatmap + final detections
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.title("Accumulator Heatmap with Barycenters")
    accum_norm = accumulator.astype(np.float32)
    if accum_norm.max() > 0:
        accum_norm /= accum_norm.max()
    base_cmap = cm.get_cmap('viridis')
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])  # paint zero‐votes white
    new_cmap = ListedColormap(colors)

    plt.imshow(accum_norm, cmap=new_cmap)
    for (cy, cx) in centers:
        plt.scatter(cx, cy, s=150, c='red', edgecolors='black', linewidth=1.5)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.title(f"Detected Instances\nModel: {os.path.basename(model_img_path)}\n"
              f"Scene: {os.path.basename(scene_img_path)}")
    plt.imshow(scene_with_boxes)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ───────────────────────────────────────────────────────────────────────────────
def find_images(folder, extensions={'jpg','png','jpeg','bmp','tiff'}):
    img_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().split('.')[-1] in extensions:
                img_paths.append(os.path.join(root, f))
    return img_paths
