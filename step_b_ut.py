import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter

def compute_model_vectors(keypoints):
    """
    From each keypoint in a model, compute vector to the barycenter.
    """
    pts = np.array([kp.pt for kp in keypoints])
    if len(pts) == 0:
        return [], np.array([0, 0])
    barycenter = np.mean(pts, axis=0)
    vectors = barycenter - pts
    return vectors, barycenter

from typing import Dict, List, Tuple

def accumulate_votes(
    matches_dict: Dict[str, List[cv2.DMatch]],
    kp_query_dict: Dict[str, List[cv2.KeyPoint]],
    kp_train_dict: Dict[str, List[cv2.KeyPoint]],
    model_vectors: Dict[str, np.ndarray],
    image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:

    accumulator = np.zeros(image_shape[:2], dtype=np.float32)
    vote_locations = []

    for channel in ['R', 'G', 'B']:
        matches = matches_dict[channel]
        model_kps = kp_query_dict[channel]
        scene_kps = kp_train_dict[channel]
        join_vectors = model_vectors[channel]

        for m in matches:
            model_pt = model_kps[m.queryIdx].pt
            scene_pt = scene_kps[m.trainIdx].pt
            model_size = model_kps[m.queryIdx].size
            scene_size = scene_kps[m.trainIdx].size

            if model_size == 0:
                continue
            scale_ratio = scene_size / model_size

            joining_vector = join_vectors[m.queryIdx]
            estimated_barycenter = np.array(scene_pt) + joining_vector * scale_ratio
            x, y = int(round(estimated_barycenter[0])), int(round(estimated_barycenter[1]))

            if 0 <= x < accumulator.shape[1] and 0 <= y < accumulator.shape[0]:
                accumulator[y, x] += 1
                vote_locations.append((x, y))

    return accumulator, vote_locations


def detect_peaks(accumulator, threshold=0.5, distance=25):
    """
    Non-maximum suppression + thresholding to find peak locations.
    """
    peaks = []
    acc_max = accumulator.max()
    if acc_max == 0:
        return []

    acc_copy = accumulator.copy()

    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(acc_copy)
        if max_val < threshold * acc_max:
            break
        x, y = max_loc
        peaks.append((x, y))
        cv2.circle(acc_copy, (x, y), distance, 0, -1)

    return peaks



def cluster_votes_around_peaks(vote_locations, peaks, radius=50):
    """
    For each peak, collect matches within a radius.
    Returns: list of clustered_matches = dict[color] = [cv2.DMatch]
    """
    clusters = []

    for peak in peaks:
        clustered = {'R': [], 'G': [], 'B': []}
        px, py = np.array(peak)

        for vote_pt, q_idx, s_idx, color in vote_locations:
            dist = np.linalg.norm(vote_pt - [px, py])
            if dist < radius:
                clustered[color].append(cv2.DMatch(q_idx, s_idx, 0))

        total = sum(len(v) for v in clustered.values())
        if total > 10:  # keep non-trivial clusters
            clusters.append((peak, clustered))

    return clusters


def compute_homography_from_matches(kp_query, kp_scene, matches_dict):
    """
    Compute homography using 2D-2D correspondences across channels.
    """
    src_pts, dst_pts = [], []

    for color in ['R', 'G', 'B']:
        q_kps = kp_query[color]
        s_kps = kp_scene[color]
        matches = matches_dict[color]

        for m in matches:
            if m.queryIdx < len(q_kps) and m.trainIdx < len(s_kps):
                src_pts.append(q_kps[m.queryIdx].pt)
                dst_pts.append(s_kps[m.trainIdx].pt)

    if len(src_pts) >= 4:
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    return None

def estimate_local_homography_from_votes(matches_dict, kp_query_dict, kp_train_dict, peak, radius=30):
    """
    Extract local matches near a peak and compute homography using them.
    """
    src_pts, dst_pts = [], []
    px, py = peak

    for color in ['R', 'G', 'B']:
        matches = matches_dict[color]
        q_kps = kp_query_dict[color]
        s_kps = kp_train_dict[color]

        for m in matches:
            s_pt = s_kps[m.trainIdx].pt
            if np.linalg.norm(np.array(s_pt) - np.array([px, py])) < radius:
                src_pts.append(q_kps[m.queryIdx].pt)
                dst_pts.append(s_pt)

    if len(src_pts) >= 4:
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    return None
