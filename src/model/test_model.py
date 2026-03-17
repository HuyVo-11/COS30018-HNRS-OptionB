import cv2
import numpy as np
import os
import glob
from keras.models import load_model

# 1. Cấu hình đường dẫn
MODEL_PATH = 'src/model/bestmodel.keras'  # Đường dẫn file model
IMAGE_FOLDER = 'src/segmentation/output_digit'  # THAY ĐỔI ĐƯỜNG DẪN FOLDER CHỨA ẢNH Ở ĐÂY


# 2. Load model
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file model tại {MODEL_PATH}")
    exit()

model = load_model(MODEL_PATH)

def preprocess_for_mnist(img_path):
    """
    Hàm tiền xử lý 'vạn năng' để biến ảnh thực tế thành chuẩn MNIST
    """
    # Đọc ảnh grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # Tự động đảo màu nếu là nền trắng chữ đen (MNIST cần nền đen chữ trắng)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Khử nhiễu và nhị phân hóa
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cắt bỏ lề thừa và căn giữa chữ số (Centering)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit_roi = img[y:y+h, x:x+w]
        
        # Tạo khung vuông có đệm (padding) để resize không bị méo
        size = max(w, h) + 14
        pad_img = np.zeros((size, size), np.uint8)
        dx, dy = (size - w) // 2, (size - h) // 2
        pad_img[dy:dy+h, dx:dx+w] = digit_roi
        img = pad_img

    # Resize về 28x28 chuẩn MNIST
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalization & Reshape
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1)) # Shape thành (1, 28, 28, 1)
    return img

# 3. Quét folder và dự đoán
print(f"Đang quét folder: {IMAGE_FOLDER}...")
print("-" * 50)
print(f"{'Tên file':<25} | {'Dự đoán':<10} | {'Độ tin cậy'}")
print("-" * 50)

# Lấy tất cả các file có định dạng ảnh trong folder
image_list = [f for f in os.listdir(IMAGE_FOLDER)]

if not image_list:
    print("Không tìm thấy file ảnh nào trong folder này!")
else:
    for filename in image_list:
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        input_data = preprocess_for_mnist(file_path)
        
        if input_data is not None:
            preds = model.predict(input_data, verbose=0)
            digit = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            print(f"{filename:<25} | {digit:<10} | {confidence:>8.2f}%")
        else:
            print(f"{filename:<25} | Lỗi xử lý ảnh")

print("-" * 50)