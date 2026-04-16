from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition


ImageInput = Union[str, Path, np.ndarray]


# =========================================================
# Basic image utilities
# =========================================================

def load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
    else:
        path = Path(image)
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not load image.")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def save_image(path: Union[str, Path], image: np.ndarray) -> str:
    path = str(path)
    ext = Path(path).suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode image for saving: {path}")
    Path(path).write_bytes(buf.tobytes())
    return path


# =========================================================
# Geometry helpers
# =========================================================

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("order_quad_points expects exactly 4 points.")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]     # top-left
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[2] = pts[np.argmax(s)]     # bottom-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def clip_points_to_image(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
    return pts


def quad_to_rect(pts: np.ndarray) -> Tuple[int, int, int, int]:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    x_min = int(np.floor(np.min(pts[:, 0])))
    y_min = int(np.floor(np.min(pts[:, 1])))
    x_max = int(np.ceil(np.max(pts[:, 0])))
    y_max = int(np.ceil(np.max(pts[:, 1])))
    return x_min, y_min, x_max, y_max


def rect_to_quad(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def polygon_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_center(pts: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def normalize_polygon_to_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    if pts.shape[0] < 4:
        raise ValueError("Polygon must have at least 4 points.")

    if pts.shape[0] == 4:
        return order_quad_points(pts)

    x_min = np.min(pts[:, 0])
    y_min = np.min(pts[:, 1])
    x_max = np.max(pts[:, 0])
    y_max = np.max(pts[:, 1])

    rect = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )
    return order_quad_points(rect)


def build_region_from_polygon(pts: np.ndarray) -> Dict[str, Any]:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x1, y1, x2, y2 = quad_to_rect(pts)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    cx, cy = polygon_center(pts)

    return {
        "polygon": pts.tolist(),
        "bbox": [x1, y1, x2, y2],
        "width": width,
        "height": height,
        "area": polygon_area(pts),
        "center_x": cx,
        "center_y": cy,
    }


def crop_quad_with_perspective(
    image: np.ndarray,
    polygon: List[List[float]],
    pad_ratio: float = 0.05,
) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("crop_quad_with_perspective expects a 4-point polygon.")

    pts = order_quad_points(pts)
    h, w = image.shape[:2]
    pts = clip_points_to_image(pts, w, h)

    tl, tr, br, bl = pts

    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    max_width = max(width_top, width_bottom)

    height_right = float(np.linalg.norm(br - tr))
    height_left = float(np.linalg.norm(bl - tl))
    max_height = max(height_right, height_left)

    adaptive_pad = pad_ratio
    if max_width < 180:
        adaptive_pad = max(adaptive_pad, 0.08)
    if max_height < 32:
        adaptive_pad = max(adaptive_pad, 0.08)

    pad_x = max_width * adaptive_pad
    pad_y = max_height * adaptive_pad

    src = np.array(
        [
            [tl[0] - pad_x, tl[1] - pad_y],
            [tr[0] + pad_x, tr[1] - pad_y],
            [br[0] + pad_x, br[1] + pad_y],
            [bl[0] - pad_x, bl[1] + pad_y],
        ],
        dtype=np.float32,
    )
    src = clip_points_to_image(src, w, h)

    out_w = int(max(8, round(max_width + 2 * pad_x)))
    out_h = int(max(8, round(max_height + 2 * pad_y)))

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def crop_rect(
    image: np.ndarray,
    bbox: List[int],
    pad_x_ratio: float = 0.03,
    pad_y_ratio: float = 0.12,
) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = int(round(bw * pad_x_ratio))
    py = int(round(bh * pad_y_ratio))

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    return image[y1:y2, x1:x2].copy()


def rotate_if_vertical(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h > 2.8 * max(1, w):
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


# =========================================================
# Full-page preprocessing for detection
# Keep same size so coordinates stay aligned.
# =========================================================

def preprocess_full_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Mild full-page preprocessing for detection.
    Same output size as input.
    This improves contrast and edge clarity without destroying text shapes.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    merged = cv2.merge([l2, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)

    return sharpened


# =========================================================
# Detection helpers
# =========================================================

def get_result_boxes(result: Any) -> Optional[Any]:
    candidate_keys = ["dt_polys", "polys", "boxes"]

    if isinstance(result, dict):
        for key in candidate_keys:
            value = result.get(key, None)
            if value is not None:
                return value
        return None

    for key in candidate_keys:
        value = getattr(result, key, None)
        if value is not None:
            return value

    return None


def bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def intersection_over_smaller(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    smaller = max(1, min(area_a, area_b))

    return float(inter / smaller)


def vertical_overlap_ratio(a: List[int], b: List[int]) -> float:
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]

    inter = max(0, min(ay2, by2) - max(ay1, by1))
    min_h = max(1, min(ay2 - ay1, by2 - by1))
    return float(inter / min_h)


def horizontal_gap(a: List[int], b: List[int]) -> int:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b

    if bx1 >= ax2:
        return bx1 - ax2
    if ax1 >= bx2:
        return ax1 - bx2
    return 0


def valid_region_dims(region: Dict[str, Any]) -> bool:
    bbox = region["bbox"]
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    return width > 0 and height > 0


def deduplicate_regions(
    regions: List[Dict[str, Any]],
    iou_threshold: float = 0.80,
    overlap_smaller_threshold: float = 0.88,
) -> List[Dict[str, Any]]:
    if not regions:
        return []

    valid_regions = [r for r in regions if valid_region_dims(r)]
    if not valid_regions:
        return []

    valid_regions.sort(key=lambda r: (-float(r["area"]), r["bbox"][1], r["bbox"][0]))

    kept: List[Dict[str, Any]] = []
    for region in valid_regions:
        keep = True
        for other in kept:
            iou = bbox_iou(region["bbox"], other["bbox"])
            ios = intersection_over_smaller(region["bbox"], other["bbox"])
            if iou >= iou_threshold or ios >= overlap_smaller_threshold:
                keep = False
                break
        if keep:
            kept.append(region)

    kept.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return kept


def should_merge_regions_into_line(
    a: Dict[str, Any],
    b: Dict[str, Any],
    y_overlap_threshold: float = 0.84,
    x_gap_ratio: float = 0.30,
    max_height_ratio: float = 1.6,
    max_width_expansion_ratio: float = 2.2,
) -> bool:
    bbox_a = a["bbox"]
    bbox_b = b["bbox"]

    overlap = vertical_overlap_ratio(bbox_a, bbox_b)
    if overlap < y_overlap_threshold:
        return False

    ha = max(1, bbox_a[3] - bbox_a[1])
    hb = max(1, bbox_b[3] - bbox_b[1])
    taller = max(ha, hb)
    shorter = max(1, min(ha, hb))

    if taller / shorter > max_height_ratio:
        return False

    gap = horizontal_gap(bbox_a, bbox_b)
    allowed_gap = int(round(max(ha, hb) * x_gap_ratio))
    if gap > allowed_gap:
        return False

    width_a = max(1, bbox_a[2] - bbox_a[0])
    width_b = max(1, bbox_b[2] - bbox_b[0])
    merged_width = max(bbox_a[2], bbox_b[2]) - min(bbox_a[0], bbox_b[0])
    wider = max(width_a, width_b)

    if gap > 0 and (merged_width / wider) > max_width_expansion_ratio:
        return False

    return True


def merge_region_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    x1 = min(a["bbox"][0], b["bbox"][0])
    y1 = min(a["bbox"][1], b["bbox"][1])
    x2 = max(a["bbox"][2], b["bbox"][2])
    y2 = max(a["bbox"][3], b["bbox"][3])

    pts = rect_to_quad(x1, y1, x2, y2)
    return build_region_from_polygon(pts)


def merge_boxes_into_lines(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not regions:
        return []

    regions = [r for r in regions if valid_region_dims(r)]
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    merged_any = True
    while merged_any:
        merged_any = False
        used = [False] * len(regions)
        new_regions: List[Dict[str, Any]] = []

        for i, region in enumerate(regions):
            if used[i]:
                continue

            current = region
            used[i] = True

            changed = True
            while changed:
                changed = False
                best_index: Optional[int] = None
                best_gap: Optional[int] = None

                for j, cand in enumerate(regions):
                    if used[j]:
                        continue

                    if should_merge_regions_into_line(current, cand):
                        gap = horizontal_gap(current["bbox"], cand["bbox"])
                        if best_gap is None or gap < best_gap:
                            best_gap = gap
                            best_index = j

                if best_index is not None:
                    current = merge_region_pair(current, regions[best_index])
                    used[best_index] = True
                    changed = True
                    merged_any = True

            new_regions.append(current)

        regions = sorted(new_regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    return regions


def split_overly_tall_regions(
    regions: List[Dict[str, Any]],
    max_height_multiplier: float = 2.8,
    max_region_height_px: int = 260,
) -> List[Dict[str, Any]]:
    if not regions:
        return []

    heights = [max(1, int(r["height"])) for r in regions if valid_region_dims(r)]
    median_h = float(np.median(heights)) if heights else 1.0

    output: List[Dict[str, Any]] = []

    for r in regions:
        if not valid_region_dims(r):
            continue

        x1, y1, x2, y2 = r["bbox"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        is_too_tall = h > (median_h * max_height_multiplier)
        is_absolutely_tall = h > max_region_height_px
        looks_like_block = h > (w * 0.30)

        if is_too_tall and is_absolutely_tall and looks_like_block:
            mid_y = (y1 + y2) // 2
            top = build_region_from_polygon(rect_to_quad(x1, y1, x2, mid_y))
            bottom = build_region_from_polygon(rect_to_quad(x1, mid_y, x2, y2))
            output.extend([top, bottom])
        else:
            output.append(r)

    return output


def sort_regions_reading_order(
    regions: List[Dict[str, Any]],
    y_tolerance: float = 0.6,
) -> List[Dict[str, Any]]:
    if not regions:
        return []

    heights = [max(1, r["bbox"][3] - r["bbox"][1]) for r in regions]
    median_h = float(np.median(heights)) if heights else 20.0
    row_thresh = max(10.0, median_h * y_tolerance)

    regions = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    rows: List[List[Dict[str, Any]]] = []

    for r in regions:
        placed = False
        cy = (r["bbox"][1] + r["bbox"][3]) / 2.0

        for row in rows:
            row_cy = np.mean([(x["bbox"][1] + x["bbox"][3]) / 2.0 for x in row])
            if abs(cy - row_cy) <= row_thresh:
                row.append(r)
                placed = True
                break

        if not placed:
            rows.append([r])

    rows.sort(key=lambda row: min(x["bbox"][1] for x in row))

    ordered: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda x: x["bbox"][0])
        ordered.extend(row)

    return ordered


# =========================================================
# Word segmentation on PREPROCESSED line crop
# =========================================================

def preprocess_line_crop_for_words(crop: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Returns binary mask for one line crop.
    text = 255, background = 0

    This is the important part:
    word segmentation happens on this preprocessed line crop,
    not on the raw crop.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    original_h, original_w = gray.shape[:2]
    scale = 1.0

    if original_h < 56:
        scale = 64.0 / max(1.0, float(original_h))
        gray = cv2.resize(
            gray,
            (int(round(original_w * scale)), int(round(original_h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return bw, scale, scale


def remove_tiny_components(mask: np.ndarray, min_component_area: int = 8) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_component_area:
            continue
        if w <= 1 or h <= 1:
            continue
        clean[labels == i] = 255

    return clean


def tighten_box_to_foreground(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[List[int]]:
    sub = mask[y1:y2, x1:x2]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [
        x1 + int(xs.min()),
        y1 + int(ys.min()),
        x1 + int(xs.max()) + 1,
        y1 + int(ys.max()) + 1,
    ]


def get_foreground_columns(mask: np.ndarray, min_ink_per_col: int = 1) -> np.ndarray:
    proj = np.sum(mask > 0, axis=0)
    return proj >= min_ink_per_col


def runs_from_binary_vector(vec: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for i, v in enumerate(vec.tolist()):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None

    if start is not None:
        runs.append((start, len(vec)))

    return runs


def merge_small_gaps(
    fg_runs: List[Tuple[int, int]],
    max_gap: int,
) -> List[Tuple[int, int]]:
    if not fg_runs:
        return []

    merged: List[Tuple[int, int]] = [fg_runs[0]]

    for start, end in fg_runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end

        if gap <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def estimate_word_gap_threshold(mask: np.ndarray) -> int:
    fg_cols = get_foreground_columns(mask, min_ink_per_col=1)
    fg_runs = runs_from_binary_vector(fg_cols)

    if len(fg_runs) <= 1:
        return 10

    gaps = []
    for i in range(1, len(fg_runs)):
        gap = fg_runs[i][0] - fg_runs[i - 1][1]
        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return 10

    gaps = np.array(gaps, dtype=np.int32)
    median_gap = int(np.median(gaps))
    perc75 = int(np.percentile(gaps, 75))

    return max(6, min(24, max(median_gap + 2, perc75)))


def vertical_bounds_for_xrange(mask: np.ndarray, x1: int, x2: int) -> Optional[Tuple[int, int]]:
    sub = mask[:, x1:x2]
    ys, xs = np.where(sub > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1


def split_words_by_projection(mask: np.ndarray) -> List[List[int]]:
    """
    mask: binary text mask for a single line crop, text=255
    returns local boxes [x1, y1, x2, y2]
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return []

    fg_cols = get_foreground_columns(mask, min_ink_per_col=1)
    fg_runs = runs_from_binary_vector(fg_cols)
    if not fg_runs:
        return []

    gap_threshold = estimate_word_gap_threshold(mask)

    # IMPORTANT:
    # we only merge tiny inner gaps, not real word gaps
    merged_runs = merge_small_gaps(fg_runs, max_gap=max(2, gap_threshold // 2))

    boxes: List[List[int]] = []
    for x1, x2 in merged_runs:
        if x2 - x1 < 3:
            continue

        y_bounds = vertical_bounds_for_xrange(mask, x1, x2)
        if y_bounds is None:
            continue

        y1, y2 = y_bounds
        box = tighten_box_to_foreground(mask, x1, y1, x2, y2)
        if box is None:
            continue

        bx1, by1, bx2, by2 = box
        bw = bx2 - bx1
        bh = by2 - by1
        area = bw * bh

        if bw < 5 or bh < 6:
            continue
        if area < 60:
            continue

        boxes.append([bx1, by1, bx2, by2])

    return boxes


def attach_isolated_punctuation(mask: np.ndarray, boxes: List[List[int]]) -> List[List[int]]:
    if not boxes:
        return boxes

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    extra_boxes: List[List[int]] = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 3:
            continue
        if w > 10 or h > 10:
            continue

        tiny = [x, y, x + w, y + h]

        overlaps_any = False
        for bx in boxes:
            if bbox_iou(tiny, bx) > 0 or intersection_over_smaller(tiny, bx) > 0:
                overlaps_any = True
                break
        if overlaps_any:
            continue

        best_idx = None
        best_dist = None

        tx = (tiny[0] + tiny[2]) / 2.0
        ty = (tiny[1] + tiny[3]) / 2.0

        for idx, bx in enumerate(boxes):
            by_center = (bx[1] + bx[3]) / 2.0
            vdist = abs(ty - by_center)
            if vdist > max(10, (bx[3] - bx[1]) * 0.8):
                continue

            if tx < bx[0]:
                hdist = bx[0] - tx
            elif tx > bx[2]:
                hdist = tx - bx[2]
            else:
                hdist = 0

            dist = hdist + 0.5 * vdist
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None and best_dist is not None and best_dist <= 12:
            bx = boxes[best_idx]
            boxes[best_idx] = [
                min(bx[0], tiny[0]),
                min(bx[1], tiny[1]),
                max(bx[2], tiny[2]),
                max(bx[3], tiny[3]),
            ]
        else:
            extra_boxes.append(tiny)

    boxes.extend(extra_boxes)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def split_touching_wide_word_boxes(mask: np.ndarray, boxes: List[List[int]]) -> List[List[int]]:
    """
    Extra pass:
    if one detected box is too wide, try to split it at deep low-ink valleys.
    This helps recover real word boundaries when two words got fused.
    """
    if not boxes:
        return boxes

    out: List[List[int]] = []

    widths = [b[2] - b[0] for b in boxes]
    median_w = float(np.median(widths)) if widths else 1.0

    for box in boxes:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        if bw < max(30, median_w * 1.8):
            out.append(box)
            continue

        sub = mask[y1:y2, x1:x2]
        if sub.size == 0:
            out.append(box)
            continue

        proj = np.sum(sub > 0, axis=0).astype(np.float32)
        if len(proj) < 8:
            out.append(box)
            continue

        smooth = cv2.GaussianBlur(proj.reshape(1, -1), (1, 0), 0).reshape(-1)
        maxv = float(np.max(smooth)) if len(smooth) else 0.0
        if maxv <= 0:
            out.append(box)
            continue

        low = smooth <= (0.18 * maxv)
        valley_runs = runs_from_binary_vector(low)

        candidate_splits = []
        for sx1, sx2 in valley_runs:
            valley_w = sx2 - sx1
            center = (sx1 + sx2) / 2.0
            if valley_w < 2:
                continue
            if center < bw * 0.18 or center > bw * 0.82:
                continue
            candidate_splits.append((sx1, sx2))

        if not candidate_splits:
            out.append(box)
            continue

        # split at widest central valley
        best = max(candidate_splits, key=lambda t: t[1] - t[0])
        sx1, sx2 = best
        left = [x1, y1, x1 + sx1, y2]
        right = [x1 + sx2, y1, x2, y2]

        left = tighten_box_to_foreground(mask, left[0], left[1], left[2], left[3])
        right = tighten_box_to_foreground(mask, right[0], right[1], right[2], right[3])

        pieces = []
        for p in [left, right]:
            if p is None:
                continue
            pw = p[2] - p[0]
            ph = p[3] - p[1]
            if pw >= 5 and ph >= 6:
                pieces.append(p)

        if len(pieces) >= 2:
            out.extend(pieces)
        else:
            out.append(box)

    out.sort(key=lambda b: (b[1], b[0]))
    return out


def segment_words_in_line_crop(
    crop: np.ndarray,
    min_component_area: int = 8,
) -> List[List[int]]:
    """
    Better word segmentation:
    - preprocess line crop
    - clean mask
    - light horizontal joining for broken letters
    - projection-based split
    - split overly-wide fused boxes
    - punctuation attachment
    Returns crop-local boxes [x1, y1, x2, y2]
    """
    if crop.size == 0:
        return []

    original_h, original_w = crop.shape[:2]
    mask, _, _ = preprocess_line_crop_for_words(crop)
    mask = remove_tiny_components(mask, min_component_area=min_component_area)

    if np.count_nonzero(mask) == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    mask_joined = cv2.dilate(mask, kernel, iterations=1)

    boxes = split_words_by_projection(mask_joined)
    boxes = split_touching_wide_word_boxes(mask_joined, boxes)
    boxes = attach_isolated_punctuation(mask, boxes)

    if not boxes:
        return []

    sx = original_w / float(mask.shape[1])
    sy = original_h / float(mask.shape[0])

    scaled_boxes: List[List[int]] = []
    for x1, y1, x2, y2 in boxes:
        x1 = int(round(x1 * sx))
        y1 = int(round(y1 * sy))
        x2 = int(round(x2 * sx))
        y2 = int(round(y2 * sy))

        x1 = max(0, min(original_w - 1, x1))
        y1 = max(0, min(original_h - 1, y1))
        x2 = max(x1 + 1, min(original_w, x2))
        y2 = max(y1 + 1, min(original_h, y2))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 5:
            continue

        scaled_boxes.append([x1, y1, x2, y2])

    scaled_boxes.sort(key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = []
    for box in scaled_boxes:
        if not merged:
            merged.append(box)
            continue

        prev = merged[-1]
        if bbox_iou(prev, box) > 0.05 or intersection_over_smaller(prev, box) > 0.2:
            merged[-1] = [
                min(prev[0], box[0]),
                min(prev[1], box[1]),
                max(prev[2], box[2]),
                max(prev[3], box[3]),
            ]
        else:
            merged.append(box)

    return merged


def segment_words_for_regions(
    processed_image_for_segmentation: np.ndarray,
    regions: List[Dict[str, Any]],
    pad_x: int = 4,
    pad_y: int = 3,
) -> List[Dict[str, Any]]:
    """
    IMPORTANT:
    Uses the PREPROCESSED image for word segmentation,
    so boundaries are found from the cleaned crop.
    """
    img = processed_image_for_segmentation
    h, w = img.shape[:2]

    enriched: List[Dict[str, Any]] = []

    for idx, region in enumerate(regions):
        x1, y1, x2, y2 = region["bbox"]

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        region_copy = dict(region)
        region_copy["line_index"] = idx
        region_copy["words"] = []

        if cx2 <= cx1 or cy2 <= cy1:
            enriched.append(region_copy)
            continue

        crop = img[cy1:cy2, cx1:cx2]
        local_boxes = segment_words_in_line_crop(crop)

        words: List[Dict[str, Any]] = []
        for wx1, wy1, wx2, wy2 in local_boxes:
            gx1 = cx1 + wx1
            gy1 = cy1 + wy1
            gx2 = cx1 + wx2
            gy2 = cy1 + wy2

            words.append(
                {
                    "bbox": [int(gx1), int(gy1), int(gx2), int(gy2)],
                    "width": int(gx2 - gx1),
                    "height": int(gy2 - gy1),
                }
            )

        region_copy["words"] = words
        enriched.append(region_copy)

    return enriched


# =========================================================
# OCR recognition helpers
# =========================================================

def resize_for_recognition(img: np.ndarray, target_h: int = 64) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = target_h / float(h)
    new_w = max(8, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)


def enhance_crop_variants(crop: np.ndarray, enable_otsu: bool = False) -> Dict[str, np.ndarray]:
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        base = crop
    else:
        gray = crop.copy()
        base = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    variants: Dict[str, np.ndarray] = {
        "base": resize_for_recognition(base),
    }

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    variants["clahe"] = resize_for_recognition(cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR))

    sharp = cv2.GaussianBlur(clahe_gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe_gray, 1.35, sharp, -0.35, 0)
    variants["sharp_clahe"] = resize_for_recognition(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

    if enable_otsu:
        otsu = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants["otsu"] = resize_for_recognition(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    return variants


def get_result_text_and_score(result: Any) -> Tuple[str, float]:
    if isinstance(result, dict):
        text = result.get("rec_text", result.get("text", ""))
        score = result.get("rec_score", result.get("score", 0.0))
        return str(text or "").strip(), float(score or 0.0)

    text = getattr(result, "rec_text", getattr(result, "text", ""))
    score = getattr(result, "rec_score", getattr(result, "score", 0.0))
    return str(text or "").strip(), float(score or 0.0)


def is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF or
        0x0750 <= code <= 0x077F or
        0x08A0 <= code <= 0x08FF or
        0xFB50 <= code <= 0xFDFF or
        0xFE70 <= code <= 0xFEFF
    )


def is_latin_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0041 <= code <= 0x005A or
        0x0061 <= code <= 0x007A or
        0x00C0 <= code <= 0x024F
    )


def contains_arabic_chars(text: str) -> bool:
    return any(is_arabic_char(ch) for ch in text)


def contains_latin_chars(text: str) -> bool:
    return any(is_latin_char(ch) for ch in text)


def count_arabic_chars(text: str) -> int:
    return sum(1 for ch in text if is_arabic_char(ch))


def count_latin_chars(text: str) -> int:
    return sum(1 for ch in text if is_latin_char(ch))


def count_digit_chars(text: str) -> int:
    return sum(1 for ch in text if ch.isdigit())


def count_symbol_chars(text: str) -> int:
    return sum(
        1 for ch in text
        if not (ch.isalnum() or ch.isspace() or is_arabic_char(ch) or is_latin_char(ch))
    )


def detect_script_hint(text: str) -> str:
    if not text:
        return "unknown"

    arabic = count_arabic_chars(text)
    latin = count_latin_chars(text)
    digits = count_digit_chars(text)

    if arabic > 0 and latin > 0:
        return "mixed"
    if arabic > 0 and latin == 0:
        return "arabic"
    if latin > 0 and arabic == 0:
        return "latin"
    if digits > 0:
        return "numeric"
    return "unknown"


def normalize_ocr_text(text: str) -> str:
    text = str(text or "")
    replacements = {
        "\u00a0": " ",
        "\ufeff": "",
        "ـ": "",
        "\u200f": "",
        "\u200e": "",
        "\u202a": "",
        "\u202b": "",
        "\u202c": "",
        "\u202d": "",
        "\u202e": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = " ".join(text.split())
    return text.strip()


def score_text_quality(text: str) -> float:
    text = normalize_ocr_text(text)
    if not text:
        return 0.0

    compact = "".join(text.split())
    if not compact:
        return 0.0

    total = len(compact)
    alnum = sum(ch.isalnum() for ch in compact)
    useful_ratio = alnum / max(1, total)
    length_bonus = min(1.0, total / 55.0)

    arabic_count = count_arabic_chars(compact)
    latin_count = count_latin_chars(compact)
    digit_count = count_digit_chars(compact)
    symbol_count = count_symbol_chars(compact)

    symbol_ratio = symbol_count / max(1, total)

    script_bonus = 0.0
    if arabic_count > 0:
        script_bonus += 0.12
    if latin_count > 0:
        script_bonus += 0.05
    if digit_count > 0:
        script_bonus += 0.03

    mixed_penalty = 0.0
    if arabic_count > 0 and latin_count > 0 and symbol_ratio > 0.20 and digit_count == 0:
        mixed_penalty += 0.06

    symbol_penalty = 0.0
    if symbol_ratio > 0.30:
        symbol_penalty += 0.12
    elif symbol_ratio > 0.18:
        symbol_penalty += 0.05

    score = (
        0.50 * useful_ratio +
        0.28 * length_bonus +
        script_bonus -
        mixed_penalty -
        symbol_penalty
    )
    return max(0.0, min(1.0, score))


def is_garbage(text: str, score: float = 0.0) -> bool:
    text = normalize_ocr_text(text)
    if not text:
        return True

    compact = "".join(text.split())
    if len(compact) < 1:
        return True

    arabic_count = count_arabic_chars(compact)
    latin_count = count_latin_chars(compact)
    digit_count = count_digit_chars(compact)
    symbol_count = count_symbol_chars(compact)

    if len(compact) <= 2:
        if arabic_count > 0 or latin_count > 0 or digit_count > 0:
            return score < 0.18
        return True

    alnum_ratio = sum(ch.isalnum() for ch in compact) / max(1, len(compact))
    symbol_ratio = symbol_count / max(1, len(compact))

    if score < 0.20:
        return True
    if alnum_ratio < 0.38 and digit_count == 0:
        return True

    weird_patterns = ["///", "\\\\", "|||", "___", "===", ">>>", "<<<"]
    if any(p in text for p in weird_patterns):
        return True

    if symbol_ratio > 0.45 and arabic_count == 0 and latin_count == 0:
        return True

    if arabic_count > 0 and latin_count > 0 and symbol_ratio > 0.35 and digit_count == 0:
        return True

    return False


def median_height(items: List[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    heights = [max(1.0, float(r["bbox"][3] - r["bbox"][1])) for r in items]
    return float(np.median(heights))


def should_insert_space_between(
    prev_item: Dict[str, Any],
    curr_item: Dict[str, Any],
    line_direction: str,
    median_h: float,
) -> bool:
    prev_text = normalize_ocr_text(prev_item.get("text", ""))
    curr_text = normalize_ocr_text(curr_item.get("text", ""))

    if not prev_text or not curr_text:
        return False

    prev_x1, _, prev_x2, _ = prev_item["bbox"]
    curr_x1, _, curr_x2, _ = curr_item["bbox"]

    if line_direction == "rtl":
        gap = max(0.0, float(prev_x1 - curr_x2))
    else:
        gap = max(0.0, float(curr_x1 - prev_x2))

    threshold = max(1.5, 0.12 * median_h)

    prev_last = prev_text[-1]
    curr_first = curr_text[0]

    if prev_last in "([{/«“" or curr_first in ".,;:!?)\\]}%»،”":
        return gap > threshold * 1.6

    prev_is_num = prev_text.replace(",", "").replace(".", "").isdigit()
    curr_is_num = curr_text.replace(",", "").replace(".", "").isdigit()
    if prev_is_num != curr_is_num and gap > threshold * 0.55:
        return True

    if (
        (contains_arabic_chars(prev_text) != contains_arabic_chars(curr_text) or
         contains_latin_chars(prev_text) != contains_latin_chars(curr_text))
        and gap > threshold * 0.60
    ):
        return True

    if gap > threshold:
        return True

    return False


def choose_model_order_from_geometry(crop: np.ndarray) -> List[str]:
    h, w = crop.shape[:2]
    aspect = w / max(1.0, h)

    if aspect >= 7.5:
        return ["general", "arabic", "latin"]
    if aspect >= 3.8:
        return ["arabic", "general", "latin"]
    return ["arabic", "general", "latin"]


# =========================================================
# Detector
# =========================================================

class BestTextDetector:
    def __init__(
        self,
        model_name: str = "PP-OCRv5_server_det",
        limit_side_len: int = 1792,
        limit_type: str = "max",
        thresh: float = 0.22,
        box_thresh: float = 0.45,
        unclip_ratio: float = 1.7,
        merge_lines: bool = False,
        min_width: int = 8,
        min_height: int = 8,
        min_area: float = 24.0,
    ) -> None:
        self.detector = TextDetection(model_name=model_name)
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.merge_lines = merge_lines
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area

    def _detect_raw_regions(self, img: np.ndarray) -> List[Dict[str, Any]]:
        h, w = img.shape[:2]

        results = self.detector.predict(
            input=img,
            limit_side_len=self.limit_side_len,
            limit_type=self.limit_type,
            thresh=self.thresh,
            box_thresh=self.box_thresh,
            unclip_ratio=self.unclip_ratio,
        )

        regions: List[Dict[str, Any]] = []

        for result in results:
            boxes = get_result_boxes(result)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                    pts = normalize_polygon_to_quad(pts)
                    pts = clip_points_to_image(pts, w, h)

                    region = build_region_from_polygon(pts)

                    if region["width"] < self.min_width:
                        continue
                    if region["height"] < self.min_height:
                        continue
                    if region["area"] < self.min_area:
                        continue

                    regions.append(region)
                except Exception:
                    continue

        return regions

    def detect(self, image_or_processed: ImageInput) -> List[Dict[str, Any]]:
        img = load_image(image_or_processed)

        regions = self._detect_raw_regions(img)
        regions = deduplicate_regions(regions)

        if self.merge_lines:
            regions = merge_boxes_into_lines(regions)
            regions = split_overly_tall_regions(regions)
            regions = deduplicate_regions(regions)

        regions = sort_regions_reading_order(regions)
        return regions


# =========================================================
# Recognizer
# =========================================================

class MultiRecognizer:
    def __init__(self, enable_otsu_variant: bool = False) -> None:
        self.enable_otsu_variant = enable_otsu_variant
        self.models: Dict[str, TextRecognition] = {
            "arabic": TextRecognition(model_name="arabic_PP-OCRv5_mobile_rec"),
            "general": TextRecognition(model_name="PP-OCRv5_server_rec"),
            "latin": TextRecognition(model_name="latin_PP-OCRv3_mobile_rec"),
        }

    def _run_model(self, model_key: str, img: np.ndarray) -> Tuple[str, float]:
        model = self.models[model_key]
        results = model.predict(input=img)

        for result in results:
            text, score = get_result_text_and_score(result)
            return normalize_ocr_text(text), float(score)

        return "", 0.0

    @staticmethod
    def _rank_candidate(candidate: Dict[str, Any]) -> float:
        text = candidate["text"]
        score = float(candidate["score"])
        quality = float(candidate["quality"])
        model = candidate["model"]
        hint = candidate["script_hint"]
        variant = candidate["variant"]

        arabic_present = contains_arabic_chars(text)
        latin_present = contains_latin_chars(text)
        digit_present = any(ch.isdigit() for ch in text)
        symbol_ratio = count_symbol_chars(text) / max(1, len("".join(text.split())))

        model_bonus = 0.0
        variant_bonus = 0.0

        if hint == "arabic":
            if model == "arabic":
                model_bonus += 0.18
            elif model == "general":
                model_bonus += 0.08
            elif model == "latin":
                model_bonus -= 0.25
        elif hint == "latin":
            if model == "latin":
                model_bonus += 0.18
            elif model == "general":
                model_bonus += 0.10
            elif model == "arabic":
                model_bonus -= 0.12
        elif hint == "mixed":
            if model == "general":
                model_bonus += 0.24
            elif model == "arabic":
                model_bonus += 0.04
        elif hint == "numeric":
            if model == "general":
                model_bonus += 0.12
            elif model == "latin":
                model_bonus += 0.04
        else:
            if model == "general":
                model_bonus += 0.05

        if digit_present and arabic_present and latin_present and model == "general":
            model_bonus += 0.10
        if digit_present and arabic_present and model == "general":
            model_bonus += 0.05

        if variant == "base":
            variant_bonus += 0.06
        elif variant == "clahe":
            variant_bonus += 0.02
        elif variant == "sharp_clahe":
            variant_bonus += 0.04
        elif variant == "otsu":
            variant_bonus -= 0.04

        mixed_penalty = 0.0
        if arabic_present and latin_present and hint != "mixed":
            mixed_penalty += 0.04

        short_penalty = 0.06 if len("".join(text.split())) <= 2 and not digit_present else 0.0
        symbol_penalty = 0.10 if symbol_ratio > 0.28 else 0.0

        return (
            (0.56 * score) +
            (0.44 * quality) +
            model_bonus +
            variant_bonus -
            mixed_penalty -
            short_penalty -
            symbol_penalty
        )

    def recognize_best(self, crop: np.ndarray) -> Dict[str, Any]:
        crop = rotate_if_vertical(crop)
        variants = enhance_crop_variants(crop, enable_otsu=self.enable_otsu_variant)

        candidates: List[Dict[str, Any]] = []
        base_order = choose_model_order_from_geometry(crop)

        for variant_name, variant_img in variants.items():
            for model_key in base_order:
                try:
                    text, score = self._run_model(model_key, variant_img)
                    candidate = {
                        "text": text,
                        "score": float(score),
                        "model": model_key,
                        "variant": variant_name,
                        "script_hint": detect_script_hint(text),
                        "quality": score_text_quality(text),
                    }

                    candidate["quality"] = max(0.0, min(1.0, float(candidate["quality"])))

                    if not is_garbage(text, float(score)):
                        candidates.append(candidate)
                except Exception:
                    continue

        if not candidates:
            return {
                "text": "",
                "score": 0.0,
                "model": "none",
                "variant": "none",
                "script_hint": "unknown",
                "quality": 0.0,
            }

        best = max(candidates, key=self._rank_candidate)
        return best


class BestTextRecognizer:
    def __init__(self, enable_otsu_variant: bool = False) -> None:
        self.router = MultiRecognizer(enable_otsu_variant=enable_otsu_variant)

    def recognize_crop(self, crop: ImageInput) -> Dict[str, Any]:
        img = load_image(crop)
        return self.router.recognize_best(img)

    def recognize_word(
        self,
        image: np.ndarray,
        word_box: Dict[str, Any],
        save_dir: Optional[Path] = None,
        line_index: int = 0,
        word_index: int = 0,
    ) -> Dict[str, Any]:
        bbox = word_box["bbox"]
        crop = crop_rect(image, bbox, pad_x_ratio=0.05, pad_y_ratio=0.18)
        crop = rotate_if_vertical(crop)

        rec = self.recognize_crop(crop)

        crop_path: Optional[str] = None
        if save_dir is not None:
            crop_file = save_dir / f"line_{line_index:04d}_word_{word_index:04d}.png"
            save_image(crop_file, crop)
            crop_path = str(crop_file)

        out = dict(word_box)
        out.update(
            {
                "text": rec["text"],
                "score": rec["score"],
                "rec_model": rec["model"],
                "variant_used": rec["variant"],
                "script_hint": rec["script_hint"],
                "quality": rec.get("quality", 0.0),
                "crop_path": crop_path,
            }
        )
        return out

    def recognize_line_region(
        self,
        original_image: np.ndarray,
        region: Dict[str, Any],
        save_dir: Optional[Path] = None,
        line_index: int = 0,
    ) -> Dict[str, Any]:
        polygon = region["polygon"]

        try:
            line_crop = crop_quad_with_perspective(original_image, polygon, pad_ratio=0.05)
        except Exception:
            line_crop = crop_rect(original_image, region["bbox"], pad_x_ratio=0.04, pad_y_ratio=0.18)

        line_crop = rotate_if_vertical(line_crop)
        line_rec = self.recognize_crop(line_crop)

        line_crop_path: Optional[str] = None
        if save_dir is not None:
            crop_file = save_dir / f"line_{line_index:04d}.png"
            save_image(crop_file, line_crop)
            line_crop_path = str(crop_file)

        recognized_words: List[Dict[str, Any]] = []
        for word_index, word_box in enumerate(region.get("words", [])):
            try:
                recognized_word = self.recognize_word(
                    image=original_image,
                    word_box=word_box,
                    save_dir=save_dir,
                    line_index=line_index,
                    word_index=word_index,
                )
                recognized_words.append(recognized_word)
            except Exception:
                failed = dict(word_box)
                failed.update(
                    {
                        "text": "",
                        "score": 0.0,
                        "rec_model": "none",
                        "variant_used": "none",
                        "script_hint": "unknown",
                        "quality": 0.0,
                        "crop_path": None,
                    }
                )
                recognized_words.append(failed)

        merged = dict(region)
        merged.update(
            {
                "region_index": line_index,
                "text": line_rec["text"],
                "score": line_rec["score"],
                "crop_path": line_crop_path,
                "rec_model": line_rec["model"],
                "variant_used": line_rec["variant"],
                "script_hint": line_rec["script_hint"],
                "quality": line_rec.get("quality", 0.0),
                "recognized_words": recognized_words,
            }
        )
        return merged

    def recognize_regions(
        self,
        original_image: ImageInput,
        regions: List[Dict[str, Any]],
        save_crops_dir: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        img = load_image(original_image)

        save_dir: Optional[Path] = None
        if save_crops_dir is not None:
            save_dir = Path(save_crops_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        outputs: List[Dict[str, Any]] = []
        for idx, region in enumerate(regions):
            try:
                out = self.recognize_line_region(
                    original_image=img,
                    region=region,
                    save_dir=save_dir,
                    line_index=idx,
                )
            except Exception:
                out = dict(region)
                out.update(
                    {
                        "region_index": idx,
                        "text": "",
                        "score": 0.0,
                        "crop_path": None,
                        "rec_model": "none",
                        "variant_used": "none",
                        "script_hint": "unknown",
                        "quality": 0.0,
                        "recognized_words": [],
                    }
                )
            outputs.append(out)

        return outputs


# =========================================================
# Final text merge
# =========================================================

def choose_line_direction_from_words_or_line(region: Dict[str, Any]) -> str:
    words = region.get("recognized_words", [])
    if words:
        joined = " ".join(normalize_ocr_text(w.get("text", "")) for w in words)
    else:
        joined = normalize_ocr_text(region.get("text", ""))

    arabic = count_arabic_chars(joined)
    latin = count_latin_chars(joined)
    return "rtl" if arabic > latin else "ltr"


def score_line_candidate(text: str, recognizer_score: float = 0.0) -> float:
    text_q = score_text_quality(text)
    visible_len = len("".join(normalize_ocr_text(text).split()))
    len_bonus = min(1.0, visible_len / 60.0) * 0.08
    return (0.72 * text_q) + (0.20 * recognizer_score) + len_bonus


def merge_word_texts_for_region(region: Dict[str, Any]) -> str:
    words = region.get("recognized_words", [])
    if not words:
        return ""

    usable = []
    for w in words:
        text = normalize_ocr_text(w.get("text", ""))
        if not text:
            continue
        cloned = dict(w)
        cloned["text"] = text
        usable.append(cloned)

    if not usable:
        return ""

    direction = choose_line_direction_from_words_or_line(region)
    med_h = median_height(usable)

    if direction == "rtl":
        usable.sort(key=lambda x: x["bbox"][0], reverse=True)
    else:
        usable.sort(key=lambda x: x["bbox"][0])

    parts: List[str] = []
    prev_item: Optional[Dict[str, Any]] = None

    for item in usable:
        text = item["text"]
        if not text:
            continue

        if prev_item is not None and should_insert_space_between(prev_item, item, direction, med_h):
            parts.append(" ")

        parts.append(text)
        prev_item = item

    merged = "".join(parts)
    merged = " ".join(merged.split())
    return merged.strip()


def pick_best_line_text(region: Dict[str, Any]) -> str:
    line_text = normalize_ocr_text(region.get("text", ""))
    line_score = score_line_candidate(line_text, float(region.get("score", 0.0) or 0.0))

    word_text = merge_word_texts_for_region(region)
    word_scores = [
        float(w.get("score", 0.0) or 0.0)
        for w in region.get("recognized_words", [])
        if normalize_ocr_text(w.get("text", ""))
    ]
    word_mean_score = float(np.mean(word_scores)) if word_scores else 0.0
    word_score = score_line_candidate(word_text, word_mean_score)

    if word_text and word_score > line_score + 0.02:
        return word_text
    if line_text:
        return line_text
    return word_text


def merge_lines_to_text(recognized_regions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for region in recognized_regions:
        chosen = pick_best_line_text(region)
        chosen = normalize_ocr_text(chosen)
        if chosen:
            lines.append(chosen)

    return "\n".join(lines).strip()


# =========================================================
# Visualization
# =========================================================

def draw_detections(
    image: ImageInput,
    regions: List[Dict[str, Any]],
    out_path: Union[str, Path],
    draw_index: bool = False,
    draw_words: bool = True,
    draw_word_index: bool = False,
) -> str:
    img = load_image(image).copy()

    for idx, region in enumerate(regions):
        pts = np.array(region["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        if draw_index:
            x1, y1, _, _ = region["bbox"]
            cv2.putText(
                img,
                f"L{idx}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        if draw_words:
            words = region.get("words", [])
            for widx, word in enumerate(words):
                x1, y1, x2, y2 = word["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

                if draw_word_index:
                    cv2.putText(
                        img,
                        f"W{widx}",
                        (x1, max(12, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (255, 80, 0),
                        1,
                        cv2.LINE_AA,
                    )

    return save_image(out_path, img)


# =========================================================
# Full pipeline
# =========================================================

class OCRPipeline:
    def __init__(
        self,
        merge_lines: bool = False,
        enable_otsu_variant: bool = False,
        detector_model_name: str = "PP-OCRv5_server_det",
    ) -> None:
        self.detector = BestTextDetector(
            model_name=detector_model_name,
            merge_lines=merge_lines,
        )
        self.recognizer = BestTextRecognizer(enable_otsu_variant=enable_otsu_variant)

    def run(
        self,
        image: ImageInput,
        save_crops_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        original = load_image(image)
        processed = preprocess_full_image_for_detection(original)

        regions = self.detector.detect(processed)
        regions = segment_words_for_regions(processed, regions)

        recognized = self.recognizer.recognize_regions(
            original_image=original,
            regions=regions,
            save_crops_dir=save_crops_dir,
        )

        final_text = merge_lines_to_text(recognized)

        return {
            "regions": recognized,
            "text": final_text,
            "processed_image": processed,
        }


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Full OCR pipeline: preprocess -> detect -> word boundaries -> recognize -> merge text"
    )
    parser.add_argument("image", help="Path to full page/image")
    parser.add_argument("--out_json", default="recognized.json")
    parser.add_argument("--out_txt", default="recognized.txt")
    parser.add_argument("--out_vis", default="detections_vis.png")
    parser.add_argument("--out_preprocessed", default="preprocessed.png")
    parser.add_argument("--save_crops_dir", default="rec_crops")
    parser.add_argument("--enable_otsu_variant", action="store_true")
    parser.add_argument("--merge_lines", action="store_true")
    parser.add_argument("--draw_index", action="store_true")
    parser.add_argument("--draw_word_index", action="store_true")
    args = parser.parse_args()

    pipeline = OCRPipeline(
        merge_lines=args.merge_lines,
        enable_otsu_variant=args.enable_otsu_variant,
    )

    result = pipeline.run(
        image=args.image,
        save_crops_dir=args.save_crops_dir,
    )

    recognized_regions = result["regions"]
    final_text = result["text"]
    processed_img = result["processed_image"]

    Path(args.out_json).write_text(
        json.dumps(recognized_regions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.out_txt).write_text(final_text, encoding="utf-8")
    save_image(args.out_preprocessed, processed_img)

    draw_detections(
        image=args.image,
        regions=recognized_regions,
        out_path=args.out_vis,
        draw_index=args.draw_index,
        draw_words=True,
        draw_word_index=args.draw_word_index,
    )

    total_words = sum(len(r.get("words", [])) for r in recognized_regions)

    print(f"Saved recognized regions to {args.out_json}")
    print(f"Saved merged text to {args.out_txt}")
    print(f"Saved preprocessed image to {args.out_preprocessed}")
    print(f"Saved visualization to {args.out_vis}")
    print(f"Detected {len(recognized_regions)} lines/regions")
    print(f"Detected {total_words} word boxes")
    