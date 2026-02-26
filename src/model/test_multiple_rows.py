import cv2
import numpy as np
import keras
import os

# --- CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Nếu file model nằm cùng folder với code, dùng join(BASE_DIR, ...)
MODEL_PATH = os.path.join(BASE_DIR, 'bestmodel.keras')
INPUT_IMAGE = os.path.join(BASE_DIR, '..', 'segmentation', 'input_image', 'digits.png') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'test_results')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. LOAD MODEL
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy model tại: {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
print("✅ Đã load model thành công!")

# --- HÀM GỘP BOX BỊ ĐỨT (Của bà) ---
def merge_broken_parts(rects, threshold=50):
    if len(rects) < 2: return rects
    rects.sort(key=lambda r: r[0])
    merged = []
    skip = False
    for i in range(len(rects)):
        if skip:
            skip = False
            continue
        x1, y1, w1, h1 = rects[i]
        if i == len(rects) - 1:
            merged.append((x1, y1, w1, h1))
            break
        x2, y2, w2, h2 = rects[i+1]
        
        # Nếu khoảng cách tâm theo trục X nhỏ -> Gộp (Số bị đứt dọc)
        center1 = x1 + w1 // 2
        center2 = x2 + w2 // 2
        if abs(center1 - center2) < threshold:
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1+w1, x2+w2) - new_x
            new_h = max(y1+h1, y2+h2) - new_y
            merged.append((new_x, new_y, new_w, new_h))
            skip = True
        else:
            merged.append((x1, y1, w1, h1))
    return merged

# --- HÀM CHÍNH ---
def process_full_grid(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Không tìm thấy ảnh tại: {img_path}")
        return

    # Chuẩn hóa nhị phân: Chữ trắng nền đen
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 1. CẮT HÀNG
    row_sums = np.sum(thresh, axis=1)
    rows = []
    is_row = False
    start_r = 0
    for i, val in enumerate(row_sums):
        if val > 0 and not is_row:
            start_r = i
            is_row = True
        elif val == 0 and is_row:
            if i - start_r > 5: rows.append(thresh[start_r:i, :])
            is_row = False

    print(f"📂 Tìm thấy {len(rows)} hàng số.")

    # 2. XỬ LÝ TỪNG HÀNG
    for r_idx, row_img in enumerate(rows):
        # Làm dày nét nhẹ để tìm contour chính xác hơn
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(row_img, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours]
        
        # Gộp mảnh vỡ & Sắp xếp
        rects = merge_broken_parts(rects)
        rects.sort(key=lambda x: x[0])

        row_predictions = []
        for c_idx, (x, y, w, h) in enumerate(rects):
            if w < 5 or h < 5: continue 

            # Cắt ảnh từ row_img gốc (không lấy ảnh dilated)
            digit_raw = row_img[y:y+h, x:x+w]

            # --- CHIÊU QUAN TRỌNG: CĂN GIỮA (CENTERING) ---
            # Resize số về 20x20 nhưng giữ tỷ lệ
            h_d, w_d = digit_raw.shape
            ratio = 20.0 / max(h_d, w_d)
            new_size = (int(w_d * ratio), int(h_d * ratio))
            digit_resized = cv2.resize(digit_raw, new_size, interpolation=cv2.INTER_AREA)

            # Đặt vào giữa khung 28x28
            final_digit = np.zeros((28, 28), dtype=np.uint8)
            ox = (28 - new_size[0]) // 2
            oy = (28 - new_size[1]) // 2
            final_digit[oy:oy+new_size[1], ox:ox+new_size[0]] = digit_resized

            # Dự đoán
            input_data = final_digit.astype('float32') / 255.0
            input_data = np.expand_dims(input_data, axis=(0, -1))

            pred = model.predict(input_data, verbose=0)
            digit_num = np.argmax(pred)
            row_predictions.append(str(digit_num))
            
            # Lưu lại để kiểm tra nếu cần
            # cv2.imwrite(f"{OUTPUT_DIR}/r{r_idx}_c{c_idx}_p{digit_num}.png", final_digit)

        print(f"Row {r_idx:02d}: {' '.join(row_predictions)}")

if __name__ == "__main__":
    process_full_grid(INPUT_IMAGE)