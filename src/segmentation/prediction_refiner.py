import cv2
import numpy as np


TERM_END_CHARS = set("0123456789)")
TERM_START_CHARS = set("0123456789(")
OPERATORS = set("+-*/")


def _safe_mean(arr):
    return float(arr.mean()) if arr.size else 0.0


def _content_crop(roi):
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        h, w = roi.shape[:2]
        return roi.astype(np.float32), (0, 0, w, h)

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return roi[y0:y1, x0:x1].astype(np.float32), (x0, y0, x1 - x0, y1 - y0)


def _extract_features(roi, rect):
    crop, content_box = _content_crop(roi)
    binary = (crop > 0).astype(np.float32)
    binary_u8 = (binary * 255).astype(np.uint8)
    ch, cw = binary.shape[:2]

    half_h = max(1, ch // 2)
    half_w = max(1, cw // 2)
    third_h = max(1, ch // 3)
    third_w = max(1, cw // 3)

    tl = binary[:half_h, :half_w]
    tr = binary[:half_h, cw - half_w:]
    bl = binary[ch - half_h:, :half_w]
    br = binary[ch - half_h:, cw - half_w:]
    center = binary[third_h:ch - third_h, third_w:cw - third_w]
    center_row = binary[max(0, ch // 2 - 1):min(ch, ch // 2 + 2), :]
    center_col = binary[:, max(0, cw // 2 - 1):min(cw, cw // 2 + 2)]

    holes = 0
    contours, hierarchy = cv2.findContours(binary_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for node in hierarchy[0]:
            if node[3] != -1:
                holes += 1

    return {
        "rect_area": float(rect[2] * rect[3]),
        "rect_w": float(rect[2]),
        "rect_h": float(rect[3]),
        "y_center": float(rect[1]) + float(rect[3]) / 2.0,
        "content_box": content_box,
        "content_w": float(cw),
        "content_h": float(ch),
        "aspect": float(cw) / max(1.0, float(ch)),
        "holes": int(holes),
        "fill": _safe_mean(binary),
        "left": _safe_mean(binary[:, :half_w]),
        "right": _safe_mean(binary[:, cw - half_w:]),
        "top": _safe_mean(binary[:half_h, :]),
        "bottom": _safe_mean(binary[ch - half_h:, :]),
        "center": _safe_mean(center),
        "center_row": _safe_mean(center_row),
        "center_col": _safe_mean(center_col),
        "diag_primary": _safe_mean(tl) + _safe_mean(br),
        "diag_secondary": _safe_mean(tr) + _safe_mean(bl),
    }


def _looks_like_minus(features):
    return (
        features["aspect"] >= 2.0
        and features["content_h"] <= 6
        and features["center_row"] >= 0.45
    )


def _looks_like_slash(features):
    return (
        features["content_h"] >= 12
        and features["aspect"] <= 0.9
        and features["fill"] <= 0.28
        and features["diag_secondary"] >= 0.45
        and features["diag_primary"] <= 0.35
    )


def _looks_like_lparen(features):
    return (
        features["content_h"] >= 14
        and features["aspect"] <= 0.55
        and features["fill"] <= 0.34
        and features["center_row"] <= 0.35
        and features["center_col"] <= 0.3
        and features["left"] >= features["right"] * 1.45
    )


def _looks_like_rparen(features):
    return (
        features["content_h"] >= 14
        and features["aspect"] <= 0.55
        and features["fill"] <= 0.34
        and features["center_row"] <= 0.35
        and features["center_col"] <= 0.3
        and features["right"] >= features["left"] * 1.45
    )


def _looks_like_star(features):
    return (
        0.75 <= features["aspect"] <= 1.35
        and features["center"] >= 0.5
        and features["center_row"] >= 0.65
        and features["diag_primary"] >= 0.95
        and features["diag_secondary"] >= 0.95
        and features["center_col"] <= 0.55
    )


def _is_open_paren_context(prev_char, next_char):
    prev_ok = prev_char is None or prev_char in OPERATORS or prev_char == "("
    next_ok = next_char in TERM_START_CHARS or next_char == "-"
    return prev_ok and next_ok


def _is_close_paren_context(prev_char, next_char):
    prev_ok = prev_char in TERM_END_CHARS
    next_ok = next_char is None or next_char in OPERATORS or next_char == ")"
    return prev_ok and next_ok


def _should_drop_noise(item, median_area, median_height):
    features = item["features"]
    if item["conf"] < 0.55 and features["rect_area"] < max(25.0, median_area * 0.1):
        return True
    if (
        features["rect_area"] < max(120.0, median_area * 0.02)
        and features["rect_h"] < max(14.0, median_height * 0.2)
    ):
        return True
    if features["content_h"] < 5 and item["conf"] < 0.98:
        return True
    if (
        item["char"] == "-"
        and item["conf"] < 0.995
        and features["rect_area"] < max(32.0, median_area * 0.22)
        and features["rect_h"] < max(10.0, median_height * 0.35)
    ):
        return True
    return False


def _cluster_lines(items, median_height):
    tolerance = max(28.0, median_height * 0.65)
    groups = []

    for item in sorted(items, key=lambda candidate: candidate["features"]["y_center"]):
        y_center = item["features"]["y_center"]
        placed = False

        for group in groups:
            if abs(y_center - group["center"]) <= tolerance:
                group["items"].append(item)
                group["center"] = float(np.mean([
                    member["features"]["y_center"] for member in group["items"]
                ]))
                placed = True
                break

        if not placed:
            groups.append({"center": y_center, "items": [item]})

    return [group["items"] for group in groups]


def _filter_main_line(items, median_area, median_height):
    core_items = [
        item for item in items
        if item["features"]["rect_area"] >= max(150.0, median_area * 0.2)
    ]
    if len(core_items) < 3:
        return sorted(items, key=lambda item: item["rect"][0])

    y_centers = [item["features"]["y_center"] for item in core_items]
    line_spread = max(y_centers) - min(y_centers)
    if line_spread <= max(120.0, median_height * 1.65):
        return sorted(items, key=lambda item: item["rect"][0])

    line_groups = [group for group in _cluster_lines(core_items, median_height) if len(group) >= 3]
    if len(line_groups) <= 1:
        return sorted(items, key=lambda item: item["rect"][0])

    def _line_score(group):
        count_score = len(group) * 1000.0
        area_score = sum(item["features"]["rect_area"] for item in group)
        y_score = -min(item["features"]["y_center"] for item in group)
        return count_score + area_score + y_score

    selected_group = max(line_groups, key=_line_score)
    selected_center = float(np.mean([item["features"]["y_center"] for item in selected_group]))
    tolerance = max(40.0, median_height * 0.85)

    filtered = [
        item for item in items
        if abs(item["features"]["y_center"] - selected_center) <= tolerance
    ]
    return sorted(filtered or items, key=lambda item: item["rect"][0])


def _looks_like_numeric_line(items):
    if len(items) < 5:
        return False

    strong_ops = 0
    for item in items:
        char = item["char"]
        conf = item["conf"]
        features = item["features"]

        if char in {"+", "*"} and conf >= 0.9:
            strong_ops += 1
        elif char == "-" and conf >= 0.98 and _looks_like_minus(features):
            strong_ops += 1
        elif char == "/" and conf >= 0.8 and _looks_like_slash(features):
            strong_ops += 1

    digitish = sum(
        item["char"].isdigit() or item["char"] in {"(", ")", "/"}
        for item in items
    )
    return strong_ops == 0 and digitish >= len(items) - 1


def _map_numeric_char(item):
    raw_char = item["char"]
    features = item["features"]

    if features["holes"] >= 2:
        return "8"

    if (
        raw_char in {"7", "0", ")"}
        and features["holes"] == 1
        and features["center"] >= 0.18
        and features["right"] >= features["left"] * 1.2
        and features["top"] >= features["bottom"] * 0.95
    ):
        return "9"

    if (
        raw_char == "/"
        and features["holes"] == 0
        and features["aspect"] <= 0.62
        and features["content_w"] <= 10
    ):
        return "1"

    if raw_char == "/" and features["aspect"] >= 0.75:
        if features["bottom"] > features["top"] * 1.12:
            return "2"
        return "3"

    if raw_char in {"(", ")", "6"}:
        if features["holes"] >= 2:
            return "8"
        if features["left"] >= features["right"] * 1.15:
            return "6"
        if (
            features["right"] >= features["left"] * 1.15
            and features["center"] >= 0.18
        ):
            return "9"

    return raw_char


def _override_char(item, prev_char=None, next_char=None):
    features = item["features"]
    raw_char = item["char"]

    if (
        raw_char in {"4", "*", "+"}
        and _looks_like_star(features)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "*"

    if (
        raw_char in {"1", "7", "/"}
        and _looks_like_slash(features)
        and prev_char in TERM_END_CHARS
        and next_char in TERM_START_CHARS
    ):
        return "/"

    if (
        raw_char in {"(", "1", "6"}
        and _looks_like_lparen(features)
        and _is_open_paren_context(prev_char, next_char)
    ):
        return "("

    if (
        raw_char in {")", "1", "7"}
        and _looks_like_rparen(features)
        and _is_close_paren_context(prev_char, next_char)
    ):
        return ")"

    return raw_char


def _should_merge_as_star(left_item, right_item, prev_char, next_char, median_height):
    if prev_char not in TERM_END_CHARS or next_char not in TERM_START_CHARS:
        return False

    if {left_item["char"], right_item["char"]} - {"(", ")", "1", "/"}:
        return False

    lx, ly, lw, lh = left_item["rect"]
    rx, ry, rw, rh = right_item["rect"]
    gap = rx - (lx + lw)
    x_overlap = min(lx + lw, rx + rw) - max(lx, rx)
    y_overlap = min(ly + lh, ry + rh) - max(ly, ry)
    combined_w = (rx + rw) - lx
    combined_h = max(ly + lh, ry + rh) - min(ly, ry)
    combined_aspect = combined_w / max(1.0, float(combined_h))

    return (
        gap <= max(6.0, median_height * 0.12)
        and x_overlap >= -max(6.0, median_height * 0.05)
        and y_overlap >= min(lh, rh) * 0.45
        and 0.45 <= combined_aspect <= 1.2
    )


def _merge_items_as_star(left_item, right_item):
    lx, ly, lw, lh = left_item["rect"]
    rx, ry, rw, rh = right_item["rect"]
    merged_rect = (
        min(lx, rx),
        min(ly, ry),
        max(lx + lw, rx + rw) - min(lx, rx),
        max(ly + lh, ry + rh) - min(ly, ry),
    )
    return {
        "char": "*",
        "conf": max(left_item["conf"], right_item["conf"]),
        "raw_char": f"{left_item['char']}{right_item['char']}",
        "raw_conf": max(left_item["conf"], right_item["conf"]),
        "rect": merged_rect,
        "adjusted": True,
    }


def _drop_edge_operator_noise(items, median_area, median_height):
    trimmed = list(items)

    while len(trimmed) >= 2:
        current = trimmed[0]
        nxt = trimmed[1]
        gap = nxt["rect"][0] - (current["rect"][0] + current["rect"][2])
        area = current["features"]["rect_area"]
        if (
            current["char"] in OPERATORS
            and _looks_like_minus(current["features"])
            and area < max(45.0, median_area * 0.25)
            and gap > median_height * 0.45
        ):
            trimmed.pop(0)
            continue
        break

    while len(trimmed) >= 2:
        current = trimmed[-1]
        prev = trimmed[-2]
        gap = current["rect"][0] - (prev["rect"][0] + prev["rect"][2])
        area = current["features"]["rect_area"]
        if (
            current["char"] in OPERATORS
            and _looks_like_minus(current["features"])
            and area < max(45.0, median_area * 0.25)
            and gap > median_height * 0.45
        ):
            trimmed.pop()
            continue
        break

    return trimmed


def refine_predictions(rects, roi_images, raw_predictions):
    if not rects or not roi_images or not raw_predictions:
        return []

    items = []
    for rect, roi, raw_pred in zip(rects, roi_images, raw_predictions):
        if isinstance(raw_pred, dict):
            char = raw_pred["char"]
            conf = float(raw_pred["conf"])
        else:
            char, conf = raw_pred
            conf = float(conf)

        items.append({
            "char": char,
            "conf": conf,
            "raw_char": char,
            "raw_conf": conf,
            "rect": tuple(int(v) for v in rect),
            "features": _extract_features(roi, rect),
            "adjusted": False,
        })

    median_area = float(np.median([item["features"]["rect_area"] for item in items]))
    median_height = float(np.median([item["features"]["rect_h"] for item in items]))

    items = _filter_main_line(items, median_area, median_height)
    if not items:
        return []

    median_area = float(np.median([item["features"]["rect_area"] for item in items]))
    median_height = float(np.median([item["features"]["rect_h"] for item in items]))

    items = [item for item in items if not _should_drop_noise(item, median_area, median_height)]
    if not items:
        return []

    numeric_mode = _looks_like_numeric_line(items)

    if numeric_mode:
        merged = items
    else:
        merged = []
        i = 0
        while i < len(items):
            current = items[i]
            if i + 1 < len(items):
                prev_char = items[i - 1]["char"] if i > 0 else None
                next_char = items[i + 2]["char"] if i + 2 < len(items) else None
                if _should_merge_as_star(current, items[i + 1], prev_char, next_char, median_height):
                    merged.append(_merge_items_as_star(current, items[i + 1]))
                    i += 2
                    continue
            merged.append(current)
            i += 1

    final_items = []
    if numeric_mode:
        for item in merged:
            refined_char = _map_numeric_char(item)
            if refined_char != item["char"]:
                item = dict(item)
                item["char"] = refined_char
                item["adjusted"] = True
            final_items.append(item)
    else:
        for idx, item in enumerate(merged):
            prev_char = final_items[-1]["char"] if final_items else None
            next_char = merged[idx + 1]["char"] if idx + 1 < len(merged) else None
            if "features" in item:
                refined_char = _override_char(item, prev_char, next_char)
                if refined_char != item["char"]:
                    item = dict(item)
                    item["char"] = refined_char
                    item["adjusted"] = True
            final_items.append(item)

        final_items = _drop_edge_operator_noise(final_items, median_area, median_height)

    return [{
        "char": item["char"],
        "conf": round(float(item["conf"]), 3),
        "raw_char": item["raw_char"],
        "raw_conf": round(float(item["raw_conf"]), 3),
        "rect": item["rect"],
        "adjusted": bool(item["adjusted"]),
    } for item in final_items]


def refine_predictions_by_line(rects, roi_images, raw_predictions):
    if not rects or not roi_images or not raw_predictions:
        return []

    entries = []
    for rect, roi, raw_pred in zip(rects, roi_images, raw_predictions):
        if isinstance(raw_pred, dict):
            conf = float(raw_pred["conf"])
        else:
            _, conf = raw_pred
            conf = float(conf)

        x, y, w, h = (int(v) for v in rect)
        entries.append({
            "rect": (x, y, w, h),
            "roi": roi,
            "raw_pred": raw_pred,
            "area": float(w * h),
            "y_center": y + (h / 2.0),
            "conf": conf,
        })

    heights = [entry["rect"][3] for entry in entries]
    median_height = float(np.median(heights)) if heights else 0.0
    tolerance = max(60.0, median_height * 1.25)

    groups = []
    for entry in sorted(entries, key=lambda item: item["y_center"]):
        placed = False
        for group in groups:
            if abs(entry["y_center"] - group["center"]) <= tolerance:
                group["items"].append(entry)
                group["center"] = float(np.mean([
                    item["y_center"] for item in group["items"]
                ]))
                placed = True
                break
        if not placed:
            groups.append({
                "center": entry["y_center"],
                "items": [entry],
            })

    if len(groups) == 1:
        refined = refine_predictions(rects, roi_images, raw_predictions)
        if not refined:
            return []
        return [{
            "line_index": 0,
            "rect": (
                min(item["rect"][0] for item in refined),
                min(item["rect"][1] for item in refined),
                max(item["rect"][0] + item["rect"][2] for item in refined) - min(item["rect"][0] for item in refined),
                max(item["rect"][1] + item["rect"][3] for item in refined) - min(item["rect"][1] for item in refined),
            ),
            "characters": refined,
        }]

    max_area = max(sum(item["area"] for item in group["items"]) for group in groups)
    significant_groups = []
    for group in groups:
        group_items = group["items"]
        group_area = sum(item["area"] for item in group_items)
        if group_area >= max_area * 0.08 or len(group_items) >= 3:
            significant_groups.append(group)

    if not significant_groups:
        significant_groups = groups

    line_results = []
    for line_index, group in enumerate(sorted(significant_groups, key=lambda item: min(v["rect"][1] for v in item["items"]))):
        sorted_items = sorted(group["items"], key=lambda item: item["rect"][0])
        line_rects = [item["rect"] for item in sorted_items]
        line_rois = [item["roi"] for item in sorted_items]
        line_raw = [item["raw_pred"] for item in sorted_items]
        refined = refine_predictions(line_rects, line_rois, line_raw)
        if not refined:
            continue

        min_x = min(item["rect"][0] for item in refined)
        min_y = min(item["rect"][1] for item in refined)
        max_x = max(item["rect"][0] + item["rect"][2] for item in refined)
        max_y = max(item["rect"][1] + item["rect"][3] for item in refined)

        line_results.append({
            "line_index": line_index,
            "rect": (min_x, min_y, max_x - min_x, max_y - min_y),
            "characters": refined,
        })

    return line_results
