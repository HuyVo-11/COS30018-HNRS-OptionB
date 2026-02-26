import os
import cv2
import numpy as np

def load_custom_data(base_path, limit_per_folder=5000):
    images = []
    labels = []
    
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            print(f"Đang load folder {folder_name}...")
            count = 0 
            
            for filename in os.listdir(folder_path):
                if count >= limit_per_folder: 
                    break
                
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Sửa đoạn này trong file load_custom_data.py của bạn:
                if img is not None:
                    img = cv2.resize(img, (28, 28))
                    img = cv2.medianBlur(img, 3) 
                    
                    # THAY THẾ OTSU BẰNG ADAPTIVE THRESHOLD:
                    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2)
                    
                    # Xóa nhiễu nhỏ
                    kernel = np.ones((2,2), np.uint8)
                    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                    
                    images.append(img)
                    labels.append(int(folder_name))
                    count += 1
                    
    return np.array(images), np.array(labels)