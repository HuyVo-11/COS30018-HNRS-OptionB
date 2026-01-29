import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CẤU HÌNH ---
# Bà nhớ check đúng tên file hình nha
INPUT_FILE = 'input_image/test.jpg'  
OUTPUT_DIR = 'output_digit'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- HÀM 1: ĐỌC ẢNH AN TOÀN ---
def read_image_safe(path):
    try:
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

# --- HÀM 2: GỘP CÁC BOX BỊ ĐỨT (Logic mới thêm) ---
def merge_broken_parts(rects):
    # Nếu ít hơn 2 box thì khỏi gộp
    if len(rects) < 2:
        return rects

    # Sắp xếp theo trục X (từ trái qua phải)
    rects.sort(key=lambda r: r[0])
    
    merged_rects = []
    skip_next = False
    
    for i in range(len(rects)):
        if skip_next:
            skip_next = False
            continue
            
        # Lấy box hiện tại
        x1, y1, w1, h1 = rects[i]
        
        # Nếu đây là box cuối cùng rồi thì thêm vào luôn
        if i == len(rects) - 1:
            merged_rects.append((x1, y1, w1, h1))
            break
            
        # Lấy box kế tiếp để so sánh
        x2, y2, w2, h2 = rects[i+1]
        
        # Tính tâm của 2 box
        center1 = x1 + w1 // 2
        center2 = x2 + w2 // 2
        
        # ĐIỀU KIỆN GỘP:
        # 1. Hai box phải nằm gần nhau theo chiều ngang (thẳng hàng dọc)
        #    (Khoảng cách giữa 2 tâm nhỏ hơn 20 pixel)
        # 2. Hai box phải gần nhau (đừng gộp số 1 và số 3 nếu tụi nó xa quá)
        dist_centers = abs(center1 - center2)
        
        # Nếu thẳng hàng (lệch nhau xíu xiu < 20px) -> GỘP!
        if dist_centers < 20:
            # Tạo box mới bao trùm cả 2
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1+w1, x2+w2) - new_x
            new_h = max(y1+h1, y2+h2) - new_y
            
            merged_rects.append((new_x, new_y, new_w, new_h))
            skip_next = True # Bỏ qua box kế tiếp vì đã gộp rồi
            print(f"[LOGIC] Da gop 2 manh vo cua so tai vi tri X={new_x}")
        else:
            merged_rects.append((x1, y1, w1, h1))
            
    return merged_rects

# --- HÀM CHÍNH ---
def process_image(image_path):
    print(f"[INFO] Dang xu ly anh: {image_path}")
    img = read_image_safe(image_path)
    
    if img is None:
        print("[ERROR] Khong tim thay file anh!")
        return

    img_display = img.copy()
    if len(img.shape) == 3 and img.shape[2] == 4:
         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
         img_display = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- PRE-PROCESSING ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)

    # Vẫn giữ bước làm sạch rác (Opening)
    kernel_clean = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    # Bước Closing (Hàn gắn nhẹ)
    kernel_heal = np.ones((5,3), np.uint8) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_heal)

    # --- TÌM CONTOUR ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- LỌC BAN ĐẦU ---
    initial_rects = [] 
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 300 and h < 300: # Bỏ khung quá to
            # Logic: Lấy hết các mảnh vỡ, miễn là không phải bụi quá nhỏ
            # Số 5 bị gãy thì mỗi mảnh có thể lùn (h>5) hoặc ngắn, cứ lấy hết vào
            if (h > 5 and w > 5): 
                initial_rects.append((x, y, w, h))

    print(f"[INFO] Tim thay {len(initial_rects)} manh vo ban dau.")

    # --- BƯỚC QUAN TRỌNG: GỘP BOX BẰNG LOGIC ---
    final_rects = merge_broken_parts(initial_rects)
    
    # Chạy thêm 1 lần nữa cho chắc (đề phòng số bị gãy làm 3 khúc)
    final_rects = merge_broken_parts(final_rects)

    print(f"[INFO] Sau khi gop manh vo, con lai {len(final_rects)} so hoan chinh.")

    # --- CẮT & LƯU ẢNH ---
    final_digits = []
    for i, (x, y, w, h) in enumerate(final_rects):
        # Chỉ lưu những box nào đủ tiêu chuẩn là số (sau khi gộp)
        # Cao > 15 hoặc Rộng > 15
        if h < 15 and w < 15: continue 

        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_display, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        pad = 5
        roi = thresh[max(0, y-pad):min(thresh.shape[0], y+h+pad),
                     max(0, x-pad):min(thresh.shape[1], x+w+pad)]

        if roi.size == 0: continue

        # Square Padding
        h_roi, w_roi = roi.shape
        max_dim = max(h_roi, w_roi)
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        start_x = (max_dim - w_roi) // 2
        start_y = (max_dim - h_roi) // 2
        square_img[start_y:start_y+h_roi, start_x:start_x+w_roi] = roi

        final_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{OUTPUT_DIR}/digit_{i}.png', final_img)
        final_digits.append(final_img)

    # --- SHOW KẾT QUẢ ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Final Result (Merged Boxes)")
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("Binary Input")
    plt.imshow(thresh, cmap='gray')
    plt.show()

# --- RUN ---
if __name__ == "__main__":
    process_image(INPUT_FILE)
