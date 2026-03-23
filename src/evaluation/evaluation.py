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