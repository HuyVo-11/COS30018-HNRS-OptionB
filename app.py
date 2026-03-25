import os
import sys
import cv2
import base64
from flask import Flask, render_template, request, jsonify

# Add src to the module search path so we can import project files
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from segmentation.segmentation import segment_image
from segmentation.operator_classifier import predict_character
from segmentation.expression_parser import build_and_evaluate
from segmentation.prediction_refiner import refine_predictions_by_line

app = Flask(__name__)

# Đảm bảo thư mục lưu ảnh upload tồn tại
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LINE_COLORS = [
    (15, 118, 110),
    (180, 83, 9),
    (29, 78, 216),
    (153, 27, 27),
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "Vui lòng chọn hoặc kéo thả ảnh!"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File ảnh không hợp lệ!"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, 'temp_upload.jpg')
    file.save(file_path)

    try:
        # Bước 1: Phân đoạn
        roi_images, rects, thresh, img_display = segment_image(file_path)
        if not roi_images:
            return jsonify({"error": "Không phát hiện ký tự nào trong ảnh."}), 400

        # Bước 2: Nhận diện bằng PyTorch
        raw_predictions = []
        for i, roi in enumerate(roi_images):
            char, conf = predict_character(roi)
            raw_predictions.append({"char": char, "conf": round(conf, 3)})

        line_predictions = refine_predictions_by_line(rects, roi_images, raw_predictions)
        if not line_predictions:
            return jsonify({"error": "Không thể ghép các ký tự thành dòng biểu thức hợp lệ."}), 400

        flattened_predictions = []
        lines_response = []

        for line_idx, line in enumerate(line_predictions):
            color = LINE_COLORS[line_idx % len(LINE_COLORS)]
            characters = line["characters"]
            raw_chars = [item["char"] for item in characters]
            expression_str, result_str, error = build_and_evaluate(raw_chars)

            for item in characters:
                x, y, w, h = item["rect"]
                label = f"L{line_idx + 1}: {item['char']} ({item['conf']:.2f})"
                cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    img_display,
                    label,
                    (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                flattened_predictions.append({
                    **item,
                    "line_index": line_idx,
                })

            lx, ly, lw, lh = line["rect"]
            cv2.rectangle(img_display, (lx, ly), (lx + lw, ly + lh), color, 3)
            cv2.putText(
                img_display,
                f"Line {line_idx + 1}",
                (lx, max(30, ly - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            lines_response.append({
                "line_index": line_idx,
                "expression": expression_str,
                "result": result_str,
                "error": error,
                "characters": characters,
                "rect": line["rect"],
            })

        # Bước 3: Đánh giá biểu thức tổng hợp cho front-end cũ
        if len(lines_response) == 1:
            expression_str = lines_response[0]["expression"]
            result_str = lines_response[0]["result"]
            error = lines_response[0]["error"]
        else:
            success_count = sum(1 for item in lines_response if not item["error"])
            expression_str = f"Đã phát hiện {len(lines_response)} dòng"
            result_str = f"{success_count}/{len(lines_response)} dòng hợp lệ"
            error = None

        # Chuyển ảnh sang dạng Base64 để trả về front-end
        _, buffer1 = cv2.imencode('.png', thresh)
        thresh_b64 = base64.b64encode(buffer1).decode('utf-8')
        
        _, buffer2 = cv2.imencode('.png', img_display)
        display_b64 = base64.b64encode(buffer2).decode('utf-8')

        return jsonify({
            "expression": expression_str,
            "result": result_str,
            "error": error,
            "characters": flattened_predictions,
            "lines": lines_response,
            "display_image": f"data:image/png;base64,{display_b64}",
            "thresh_image": f"data:image/png;base64,{thresh_b64}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Đã xảy ra lỗi hệ thống: {str(e)}"}), 500

if __name__ == '__main__':
    # Chạy Web trên port 5000
    print("Khởi động Máy chủ Web tại địa chỉ: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
