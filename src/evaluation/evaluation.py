from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    average: str
    labels: List[Any]
    confusion_matrix: List[List[int]]
    per_class: List[Dict[str, Any]]
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "average": self.average,
            "labels": self.labels,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
            "total_samples": self.total_samples,
        }


def _to_label_vector(values: Sequence[Any], name: str) -> np.ndarray:
    array = np.asarray(values)

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if array.ndim == 1:
        return array

    if array.ndim == 2:
        if array.shape[1] == 1:
            return array.reshape(-1)
        return np.argmax(array, axis=1)

    raise ValueError(
        f"{name} must be a 1D label vector or a 2D score matrix. Received shape {array.shape}."
    )


def _resolve_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Iterable[Any]],
) -> List[Any]:
    if labels is not None:
        return list(labels)

    merged = np.concatenate([y_true, y_pred])
    return list(np.unique(merged))


def evaluate_results(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Iterable[Any]] = None,
    average: str = "macro",
    zero_division: int = 0,
) -> Dict[str, Any]:
    """
    Compute the four main evaluation metrics for a classifier.

    Parameters
    ----------
    y_true:
        Ground-truth labels. Can be a 1D label vector or one-hot matrix.
    y_pred:
        Predicted labels, one-hot matrix, or probability/logit matrix.
    labels:
        Optional explicit label order for confusion matrix and per-class metrics.
    average:
        Aggregation for precision, recall, and F1. Default is 'macro', which is
        usually the best summary for multi-class classification.
    zero_division:
        Value returned when a class has no predicted positives or no true positives.
    """
    true_labels = _to_label_vector(y_true, "y_true")
    pred_labels = _to_label_vector(y_pred, "y_pred")

    if len(true_labels) != len(pred_labels):
        raise ValueError(
            "y_true and y_pred must have the same number of samples. "
            f"Received {len(true_labels)} and {len(pred_labels)}."
        )

    label_list = _resolve_labels(true_labels, pred_labels, labels)

    accuracy = float(accuracy_score(true_labels, pred_labels))
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=label_list,
        average=average,
        zero_division=zero_division,
    )

    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=label_list,
        average=None,
        zero_division=zero_division,
    )

    matrix = confusion_matrix(true_labels, pred_labels, labels=label_list)
    per_class = []
    for index, label in enumerate(label_list):
        per_class.append(
            {
                "label": label,
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1_score": float(per_class_f1[index]),
                "support": int(support[index]),
            }
        )

    result = EvaluationResult(
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        average=average,
        labels=label_list,
        confusion_matrix=matrix.astype(int).tolist(),
        per_class=per_class,
        total_samples=int(len(true_labels)),
    )
    return result.to_dict()


def format_evaluation(result: Dict[str, Any], decimals: int = 4) -> str:
    lines = [
        "Evaluation metrics",
        f"- Accuracy : {result['accuracy']:.{decimals}f}",
        f"- Precision: {result['precision']:.{decimals}f} ({result['average']} average)",
        f"- Recall   : {result['recall']:.{decimals}f} ({result['average']} average)",
        f"- F1-score : {result['f1_score']:.{decimals}f} ({result['average']} average)",
        f"- Samples  : {result['total_samples']}",
        "",
        "Per-class metrics",
    ]

    for item in result["per_class"]:
        lines.append(
            "- "
            f"{item['label']}: "
            f"precision={item['precision']:.{decimals}f}, "
            f"recall={item['recall']:.{decimals}f}, "
            f"f1={item['f1_score']:.{decimals}f}, "
            f"support={item['support']}"
        )

    return "\n".join(lines)


def evaluate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Iterable[Any]] = None,
    average: str = "macro",
    zero_division: int = 0,
) -> Dict[str, Any]:
    return evaluate_results(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        average=average,
        zero_division=zero_division,
    )


if __name__ == "__main__":
    sample_true = [0, 1, 2, 2, 1, 0, 3, 3]
    sample_pred = [0, 1, 2, 1, 1, 0, 3, 2]

    metrics = evaluate_results(sample_true, sample_pred)
    print(format_evaluation(metrics))
import os
import sys
import numpy as np
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ================================================================
# EVALUATION 1 — Keras model (bestmodel.keras) trên test set MNIST
# ================================================================

def evaluate_keras_model(
    model_path='src/model/bestmodel.keras',
    x_test_path='src/model/x_test_full.npy',
    y_test_path='src/model/y_test_full.npy'
):
    """
    Đánh giá bestmodel.keras trên toàn bộ test set (MNIST + EMNIST).
    In ra: Loss, Accuracy, Confusion Matrix (10 class: 0-9).
    """
    import keras
    from sklearn.metrics import confusion_matrix, classification_report

    print("=" * 60)
    print("EVALUATION 1 — Keras Model (bestmodel.keras)")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"[ERROR] Không tìm thấy model tại {model_path}")
        return
    model = keras.models.load_model(model_path)

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print("[ERROR] Không tìm thấy test data. Chạy merge_minst_emnist_data.py trước.")
        return

    x_test = np.load(x_test_path).astype('float32') / 255.0
    y_test = np.load(y_test_path)
    x_test = np.expand_dims(x_test, -1)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\n  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=[str(i) for i in range(10)]))
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}")
    return acc


# ================================================================
# EVALUATION 2 — PyTorch model (model_combined.pth) trên output_digit
# ================================================================

def evaluate_pytorch_model(
    image_folder='src/segmentation/output_digit',
    expected_chars=None
):
    """
    Đánh giá model_combined.pth trên các ROI trong output_digit/.

    expected_chars: list ký tự đúng theo thứ tự, ví dụ ['3','+','5'].
    Nếu None → chỉ in kết quả, không tính accuracy.
    """
    from segmentation.operator_classifier import predict_character

    print("=" * 60)
    print("EVALUATION 2 — PyTorch Model (model_combined.pth)")
    print("=" * 60)

    if not os.path.exists(image_folder):
        print(f"[ERROR] Không tìm thấy folder {image_folder}")
        print("[INFO] Chạy pipeline với 1 ảnh trước để tạo output_digit/")
        return

    image_list = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    if not image_list:
        print("[ERROR] Không có ảnh nào trong folder!")
        return

    print(f"\n  {'File':<20} | {'Predicted':<10} | {'Confidence':<12} | {'Expected':<10} | Result")
    print("  " + "-" * 72)

    correct = 0
    for i, filename in enumerate(image_list):
        img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        char, conf = predict_character(img)
        expected   = expected_chars[i] if (expected_chars and i < len(expected_chars)) else "?"
        is_correct = "✅" if expected != "?" and char == expected else ("❌" if expected != "?" else "-")
        if expected != "?" and char == expected:
            correct += 1

        print(f"  {filename:<20} | {char:<10} | {conf*100:>8.2f}%   | {expected:<10} | {is_correct}")

    if expected_chars:
        total = len(image_list)
        print(f"\n  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")


# ================================================================
# EVALUATION 3 — Full pipeline (ảnh → expression → result)
# ================================================================

def evaluate_pipeline(test_cases):
    """
    Test toàn bộ pipeline với list ảnh và kết quả mong đợi.

    test_cases = [
        {"image": "input_image/test1.jpg", "expected_expr": "3+5", "expected_result": "8"},
    ]
    """
    from segmentation.segmentation import segment_image
    from segmentation.operator_classifier import predict_character
    from segmentation.expression_parser import build_and_evaluate

    print("=" * 60)
    print("EVALUATION 3 — Full Pipeline")
    print("=" * 60)

    correct_expr = 0
    correct_result = 0
    total = len(test_cases)

    for tc in test_cases:
        image_path    = tc['image']
        expected_expr = tc.get('expected_expr', '?')
        expected_res  = tc.get('expected_result', '?')

        roi_images, _, _, _ = segment_image(image_path)
        if not roi_images:
            print(f"  [{image_path}] ❌ No characters detected")
            continue

        raw_chars = [predict_character(roi)[0] for roi in roi_images]
        expr_str, result_str, error = build_and_evaluate(raw_chars)

        expr_ok   = "✅" if expr_str   == expected_expr else "❌"
        result_ok = "✅" if result_str == expected_res  else "❌"
        if expr_str   == expected_expr: correct_expr   += 1
        if result_str == expected_res:  correct_result += 1

        print(f"\n  Image  : {image_path}")
        print(f"  Got    : expr={expr_str!r}  result={result_str!r}")
        print(f"  Expect : expr={expected_expr!r}  result={expected_res!r}")
        print(f"  Status : expr={expr_ok}  result={result_ok}")
        if error:
            print(f"  Error  : {error}")

    print(f"\n  Expression Accuracy : {correct_expr}/{total} = {correct_expr/total*100:.1f}%")
    print(f"  Result Accuracy     : {correct_result}/{total} = {correct_result/total*100:.1f}%")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # Evaluation 1: Keras model accuracy
    evaluate_keras_model()

    # Evaluation 2: PyTorch model trên output_digit
    # Truyền expected_chars nếu muốn tính accuracy
    # Ví dụ: expected_chars=['0','9','1','3','2','0','0','0','6','8']
    evaluate_pytorch_model(expected_chars=None)

    # Evaluation 3: Full pipeline — thêm ảnh test vào list
    # evaluate_pipeline([
    #     {"image": "input_image/test.jpg", "expected_expr": "3+5", "expected_result": "8"},
    # ])
