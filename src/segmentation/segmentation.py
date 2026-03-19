import cv2  # LIbrary OpenCV for read, cut, and filter image color
import numpy as np # Numpy library for calculations
import os #OS library for file handling within system directories   

# Create Output path
OUTPUT_DIR = 'output_digit'

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR) 

# --- HÀM 1: ĐỌC ẢNH AN TOÀN ---
def read_image_safe(path):
    try: 
        stream = open(path, "rb") 
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception: 
        return None

# --- HÀM 2: Merge the image boxes ---
def merge_broken_parts(rects):
    if len(rects) < 2:
        return rects

    rects.sort(key=lambda r: r[0])
    
    used = [False] * len(rects)
    merged_rects = []
    
    for i in range(len(rects)):
        if used[i]: continue
        
        x1, y1, w1, h1 = rects[i]
        
        merged_in_this_pass = True
        while merged_in_this_pass:
            merged_in_this_pass = False
            
            for j in range(i + 1, len(rects)):
                if used[j]: continue
                
                x2, y2, w2, h2 = rects[j]
                
                dist_centers_x = abs((x1 + w1 // 2) - (x2 + w2 // 2))
                gap_x = max(0, max(x1, x2) - min(x1+w1, x2+w2))
                
                x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                y_overlap = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                gap_y = max(0, max(y1, y2) - min(y1+h1, y2+h2))
                
                cond1 = (dist_centers_x < 15) and (gap_y <= 12)
                cond2 = (y_overlap < 20) and (gap_x <= 15) and (gap_y <= 12)
                cond3 = (x_overlap > 0) and (y_overlap > min(h1, h2) * 0.3)
                cond4 = (w1 <= 15 and w2 <= 15) and (gap_x <= 10) and (y_overlap > min(h1, h2) * 0.5)
                
                if cond1 or cond2 or cond3 or cond4:
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1+w1, x2+w2) - new_x
                    new_h = max(y1+h1, y2+h2) - new_y
                    
                    x1, y1, w1, h1 = new_x, new_y, new_w, new_h
                    used[j] = True
                    merged_in_this_pass = True 
                    print(f"[LOGIC] Radar da gop manh tai X={x2} vao X={x1}")
                    break 
                    
        merged_rects.append((x1, y1, w1, h1))
            
    return merged_rects 

#  HÀM MỚI 3: CÁ LỚN NUỐT CÁ BÉ
def remove_inside_boxes(rects):
    valid_rects = []
    for i in range(len(rects)):
        x1, y1, w1, h1 = rects[i]
        is_inside = False
        
        for j in range(len(rects)):
            if i == j: continue 
            x2, y2, w2, h2 = rects[j]
            
            if x1 >= x2 and y1 >= y2 and (x1+w1) <= (x2+w2) and (y1+h1) <= (y2+h2):
                is_inside = True
                print(f"[LOGIC] Phat hien hop nho tai X={x1} nam trong hop to -> XOA!")
                break 
                
        if not is_inside:
            valid_rects.append((x1, y1, w1, h1))
            
    return valid_rects


def segment_image(image_path):
    print(f"[INFO] Dang xu ly anh: {image_path}") 
    print("[INFO] Dang don dep folder cu...")
    
    if os.path.exists(OUTPUT_DIR):
        for file_name in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                pass
                
    img = read_image_safe(image_path)
    if img is None: 
        print("[ERROR] Cannot read the image.")
        return [], [], None, None 

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
    
    kernel_clean = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    kernel_heal = np.ones((5,3), np.uint8) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_heal)
    
    # --- TÌM CONTOUR ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- LỌC BAN ĐẦU ---
    initial_rects = [] 
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 300 and h < 300: 
            # Ép h >= 5 để chém rác cám, cho w thả ga về 2 để cứu nét mỏng
            if (h >= 5 and w >= 2): 
                initial_rects.append((x, y, w, h)) 

    print(f"[INFO] Tim thay {len(initial_rects)} manh vo ban dau.")
    
    # --- GỘP BOX BẰNG LOGIC ---
    final_rects = merge_broken_parts(initial_rects)
    final_rects = remove_inside_boxes(final_rects)
    print(f"[INFO] Sau khi gop manh vo, con lai {len(final_rects)} so hoan chinh.")

    # --- CẮT & XỬ LÝ ẢNH ---
    roi_images = []
    valid_rects = []
    valid_count = 0 
    
    for (x, y, w, h) in final_rects:
        
        # 🔥 MÀNG LỌC NANO MỚI: Chỉ chém mụn thật sự (h<10, hoặc diện tích bé tí)
        if (h < 10) or (w * h < 25): 
            print(f"[-] Vut rac nano tai X={x}")
            continue

        # Thẻ VIP Dấu Trừ: Nét ngang dài, dẹt
        is_minus = (w >= 10 and h < 20 and (w / max(1, float(h))) >= 1.5)
        
        # 🔥 THẺ VIP ĐỘC QUYỀN CHO DẤU XUYỆT / GẠCH THẲNG CỦA ÔNG KHANG
        # Nét mỏng dính, cao vừa đủ (>= 14).
        # Phải ốm nhom (h lớn hơn w ít nhất 1.5 lần) để phân biệt với cục bụi vuông vức
        is_slash = (h >= 14 and (float(h) / max(1, w)) >= 1.5)
        
        if not (is_minus or is_slash):
            # LƯỚI LỌC THÉP TỰ ĐỘNG CHÉM CỤC MỤN BỰ
            if (w * h < 150) or (h < 22 and w < 22): 
                print(f"[-] Vut 1 manh vun (rac) tai X={x}")
                continue 

        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_display, str(valid_count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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

        # BÍ KÍP VÕ CÔNG CHUẨN HÓA MNIST
        resized_20 = cv2.resize(square_img, (20, 20), interpolation=cv2.INTER_AREA)
        final_img = np.zeros((28, 28), dtype=np.uint8)
        final_img[4:24, 4:24] = resized_20
        
        cv2.imwrite(f'{OUTPUT_DIR}/digit_{valid_count}.png', final_img)
        
        roi_images.append(final_img)
        valid_rects.append((x, y, w, h))
        
        valid_count += 1 
        
    return roi_images, valid_rects, thresh, img_display