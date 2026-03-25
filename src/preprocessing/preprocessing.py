import cv2
import numpy as np
import torch


TARGET_SIZE = 28
INNER_SIZE = 20


def _to_grayscale(image_input):
    if image_input is None:
        raise ValueError("image_input must not be None")

    if len(image_input.shape) == 3 and image_input.shape[2] == 3:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image_input.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def _binary_score(binary):
    foreground = int(np.count_nonzero(binary))
    if foreground == 0:
        return float("-inf")

    ratio = foreground / float(binary.size)
    edges = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    edge_ratio = np.count_nonzero(edges) / max(1.0, float(edges.size))

    # A handwritten character should occupy a minority of the canvas and
    # should not flood the image borders.
    return -abs(ratio - 0.14) - (edge_ratio * 0.8)


def _binarize_character(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary_inv = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    candidates = [binary_inv, binary]
    chosen = max(candidates, key=_binary_score)

    # Keep thin strokes like '/' and '-' while removing isolated specks.
    chosen = cv2.medianBlur(chosen, 3)
    return chosen


def _center_on_canvas(binary):
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]

    if h >= w:
        new_h = INNER_SIZE
        new_w = max(1, int(round(w * INNER_SIZE / float(h))))
    else:
        new_w = INNER_SIZE
        new_h = max(1, int(round(h * INNER_SIZE / float(w))))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    x_offset = (TARGET_SIZE - new_w) // 2
    y_offset = (TARGET_SIZE - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        shift_x = int(round((TARGET_SIZE / 2.0) - center_x))
        shift_y = int(round((TARGET_SIZE / 2.0) - center_y))
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, matrix, (TARGET_SIZE, TARGET_SIZE))

    return canvas


def preprocess(image_input):
    """
    Normalize a character image to the same 28x28 white-on-black format used
    by the classifier.
    """
    gray = _to_grayscale(image_input)
    binary = _binarize_character(gray)
    return _center_on_canvas(binary)


def preprocess_for_torch(image_input):
    processed = preprocess(image_input)
    tensor = torch.tensor(processed, dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


def preprocess_for_keras(image_input):
    processed = preprocess(image_input)
    normalized = processed.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))
