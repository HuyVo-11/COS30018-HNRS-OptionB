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
