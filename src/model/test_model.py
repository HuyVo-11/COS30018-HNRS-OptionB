import os
import cv2
import numpy as np
import keras
from keras.models import load_model

# 1. Đường dẫn tới model và folder ảnh
MODEL_PATH = 'D:\\COS30018-HNRS-OptionB\\src\\model\\bestmodel.keras' # Hoặc đường dẫn tuyệt đối của bạn
IMAGE_FOLDER = 'D:\\COS30018-HNRS-OptionB\\src\\segmentation\\output_image'

# 2. Load model đã train
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file model tại {MODEL_PATH}")
else:
    model = load_model(MODEL_PATH)
    print("Đã load model thành công!")

# 3. Hàm tiền xử lý ảnh để khớp với MNIST (28x28, 1 channel, scale 0-1)
def preprocess_image(img_path):
    # Đọc ảnh (dạng grayscale)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Resize về 28x28 như dataset MNIST
    img = cv2.resize(img, (28, 28))
    
    # Đảo ngược màu nếu cần (MNIST là nền đen chữ trắng)
    # Nếu ảnh của bạn là chữ đen nền trắng, hãy uncomment dòng dưới:
    # img = cv2.bitwise_not(img)
    
    # Normalize về [0, 1]
    img = img.astype('float32') / 255.0
    
    # Reshape về (1, 28, 28, 1) để đưa vào model predict
    img = np.expand_dims(img, axis=(0, -1))
    return img

# 4. Duyệt qua folder và predict
print(f"{'File Name':<25} | {'Predicted Number'}")
print("-" * 45)

for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        
        processed_img = preprocess_image(img_path)
        
        if processed_img is not None:
            # Predict
            prediction = model.predict(processed_img, verbose=0)
            
            # Lấy index của class có xác suất cao nhất
            predicted_class = np.argmax(prediction)
            
            print(f"{filename:<25} | {predicted_class}")
        else:
            print(f"Không thể đọc file: {filename}")