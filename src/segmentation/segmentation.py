from glob import glob
import cv2  # LIbrary OpenCV for read, cut, and filter image color
import numpy as np # Numpy library for calculations
import matplotlib.pyplot as plt
import os #OS library for file handling within system directories   

# Create Input and Output paths
INPUT_FILE = 'src/segmentation/input_image/test2.jpg'  
OUTPUT_DIR = 'src/segmentation/output_digit'

if not os.path.exists(OUTPUT_DIR): #Check if the output directory (path) exists in the system
    os.makedirs(OUTPUT_DIR) #If not, create the output directory

# --- HÀM 1: ĐỌC ẢNH AN TOÀN ---
def read_image_safe(path):
    try: 
        stream = open(path, "rb") #Open the image file in binary mode to get the raw data
        bytes = bytearray(stream.read())
        #Read the byte data into a numpy array and decode it into an image format using OpenCV
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        # Turn Python byte data into image into Numpy array for OpenCV
        # np.uint8: Số nguyên dương 8-bit (0-255) để biểu diễn giá trị màu pixel trong ảnh (mỗi pixel có giá trị từ 0 đến 255)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        #Use OpenCV to decoded the byte data into an image format to read
        # Dùng hàm IMREAD_UNCHANGED để giữ nguyên định dạng ảnh gốc không đổi màu 
    except Exception: #If there is any error during the process
        return None
        # Return None means that the image could not be read successfully
# --- HÀM 2: Merge the image boxes ---
def merge_broken_parts(rects):
    # Nếu ít hơn 2 box thì khỏi gộp
    if len(rects) < 2:
        return rects
    # If only found less than 2 boxes near, no need to merge, return the original boxes 

    # Order the list of rectangles from left to right base on the position X - r[0]
    rects.sort(key=lambda r: r[0])
    
    merged_rects = []
    #Create an empty list to store the merged rectangles after processing
    
    skip_next = False
    #Create a flag variable to skip the next rectangle if it has been merged with the current one
    
    # Loop through each rectangle in the sorted list
    for i in range(len(rects)):
        # If the previous rectangle was merged with the current one, skip this iteration
        if skip_next:
            skip_next = False
            continue
        
        # Take the position X,Y and the size W,H of the current rectangle
        x1, y1, w1, h1 = rects[i]
        
        # Check if this is the last item in the list
        if i == len(rects) - 1:
            merged_rects.append((x1, y1, w1, h1)) #Append it to the merged list as is
            break # Exit the loop since there are no more rectangles to process
            
        # Take the position and size of the next rectangle
        x2, y2, w2, h2 = rects[i+1]
        
        # Calculate the center X of both rectangles (Current and Next)
        center1 = x1 + w1 // 2
        center2 = x2 + w2 // 2
        
        # Merge Condition:
        # 1. Hai box phải nằm gần nhau theo chiều ngang (thẳng hàng dọc)
        #    (Khoảng cách giữa 2 tâm nhỏ hơn 20 pixel)
        # 2. Hai box phải gần nhau (đừng gộp số 1 và số 3 nếu tụi nó xa quá)
        #Calculate the distance between the centers of the two rectangles (Take absolute value)
        dist_centers = abs(center1 - center2)
        
        
        dist_x = x2 - (x1 + w1) # Tính khoảng cách giữa cạnh phải của box 1 và cạnh trái của box 2 (Khoảng cách theo chiều ngang)
        
        y_overlap = not (y1 + h1 < y2 or y2 + h2 < y1) # Đảm bảo khoảng 2 box không bị trùng theo chiều dọc (Có sự chồng lấn về Y)
        x_overlap = (x2 < x1 + w1)
        # If the distance between the centers is less than 20 pixels, merge the rectangles
        if dist_centers < 40 or (dist_x < 15 and dist_x > -10 and y_overlap):
            #Create a new rectangle that encompasses both the current and next rectangles
            
            new_x = min(x1, x2)
            # Find the minimum X coordinate between the two rectangles
            # Giá trị x để vẽ khung mới bao quát hết 2 khung nhỏ phải là điểm sát mép trái nhất
            new_y = min(y1, y2)
            # Find the minimum Y coordinate between the two rectangles
            
            new_w = max(x1+w1, x2+w2) - new_x
            # Calculate the width of the new rectangle
            # Lấy điểm sát mép phải trên cùng sau đó tính độ dài bằng cách lấy điểm cuối trừ điểm đầu
            new_h = max(y1+h1, y2+h2) - new_y
            # Calculate the height of the new rectangle
            # Tương tự lấy cạnh sát mép dưới phải rồi trừ đi điểm đầu 
            
            merged_rects.append((new_x, new_y, new_w, new_h))
            # Append the new merged rectangle to the list
            skip_next = True # Skip the next box since it has been merged
            print(f"[LOGIC] Da gop 2 manh vo cua so tai vi tri X={new_x}")
            # Print the notification of merging two rectangles at position X
        else: # If the distance is above the threshold, do not merge (>20 pixels)
            merged_rects.append((x1, y1, w1, h1))
            # Append the current rectangle as is to the merged list
            
    return merged_rects # Return the final list of merged rectangles
#  HÀM MỚI 3: CÁ LỚN NUỐT CÁ BÉ (Xóa box nhỏ nằm trong box to)
def remove_inside_boxes(rects):
    valid_rects = []
    for i in range(len(rects)):
        # Lấy box hiện tại (Box A)
        x1, y1, w1, h1 = rects[i]
        is_inside = False
        
        # So sánh Box A với tất cả các box khác (Box B)
        for j in range(len(rects)):
            if i == j: continue # Không so sánh với chính nó
            x2, y2, w2, h2 = rects[j]
            
            # Kiểm tra: Nếu Box A nằm lọt thỏm trong Box B
            # (Mép trái A >= Mép trái B VÀ Mép phải A <= Mép phải B, tương tự với trên/dưới)
            if x1 >= x2 and y1 >= y2 and (x1+w1) <= (x2+w2) and (y1+h1) <= (y2+h2):
                is_inside = True # Xác nhận là "kẻ ăn bám"
                print(f"[LOGIC] Phat hien hop nho tai X={x1} nam trong hop to -> XOA!")
                break 
                
        # Nếu không nằm trong đứa nào hết thì mới giữ lại
        if not is_inside:
            valid_rects.append((x1, y1, w1, h1))
            
    return valid_rects
#  Main Function
def process_image(image_path):
    print(f"[INFO] Dang xu ly anh: {image_path}") 
    # Print the information message about processing the image at the specified path
    print("[INFO] Dang don dep folder cu...")
    if os.path.exists(OUTPUT_DIR):
        for file_name in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file_name)
            try:
                # Xóa thẳng tay nếu là file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"  -> Cleaned: {file_name}")
            except Exception as e:
                print(f"  [!] Cannot delete {file_name}. Error: {e} (Please check if the file is open in another program)")
    img = read_image_safe(image_path)
    # Call the safe image reading function to read the image from the specified path and save it into the "img" variable
    
    if img is None: # If cannot read the image (empty img variable)
        print("[ERROR] Cannot read the image. Please check the file path and try again.")
        return

    # Create a copy of the original image for display purposes
    img_display = img.copy()
    
    # Check if the image has an alpha channel (4 channels) PNG File with 4 color channels (Hình trong suốt)
    if len(img.shape) == 3 and img.shape[2] == 4:
        
        
         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
         # Convert the image from RGBA to RGB format by removing the alpha channel (Chuyển về RGB bình thường)
         img_display = img.copy()
         # Copy the converted image for display purposes again

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Covert the image from color into Grayscale (Gray color) for easier computation processing 
    # Chuyển từ ảnh dạng màu sang ảnh xám để dễ xử lý hơn
    
    # --- PRE-PROCESSING ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Make the image blurred slightly using Gaussian Blur to reduce noise and small noisy details
    # Làm ảnh mờ đi một chút để giảm nhiễu và các chi tiết nhỏ không cần thiết, làm mờ bằng những khung 5x5 pixel cho đỡ bị mờ quá
    
    
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)
    # Threshold là ngưỡng để xác định, phân biệt 
    # Dùng Adaptive Thresholding để chuyển ảnh thành nhị phân (đen trắng) với 255 là trắng tuyệt đối
    # Loại Thresholding: Inverse Binary (_INV) để chuyển mặc định (Chữ đen nền trắng) thành ngược lại (Chữ trắng nền đen)
    # 11,5: Kích thước vùng lân cận 11 pixel để tính Threshold  và hằng số trừ đi từ giá trị trung bình để dễ điều chỉnh hơn (Kiểu trừ bớt đi 5 để tính toán khắt khe hơn, đỡ bị nhận nhầm nhiễu thành số)
    
    
    # Create a square frame (Kernel) 3x3 pixel for Morphological Operations
    # Tạo một cái khuôn (kernel) hình vuông nhỏ 3x3 pixel
    kernel_clean = np.ones((3,3), np.uint8)
    
    # Use OPEN technique to remove small noise in the binary image with the 3x3 kernel
    # Sử dụng phép toán Mở (Opening) để loại bỏ nhiễu nhỏ trong ảnh nhị phân với cái khuôn 3x3
    # MorphologyEx là phép toán: Erosion (Xói mòn) followed by Dilation (Phình to) giúp loại bỏ các chấm nhỏ li ti bằng các vùng 3x3 (kernel_clean) còn sót lại sau khi tính Thresholdsau đó fill lại các vùng lớn hơn 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    # Bước Closing (Hàn gắn)
    # Create a rectangular kernel 5x3 pixel for Morphological Closing   
    # Tạo 1 cái khuôn hình chữ nhật 5x3 pixel để thực hiện phép toán Đóng (Closing)
    kernel_heal = np.ones((5,3), np.uint8) 
    # Ưu tiên chiều cao để không bị dính qua mấy số khác bên cạnh
    # Use the Closing technique to close small gaps within the objects in the binary image
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_heal)
    # Dùng phép CLOSE (Đóng): Để nối liền các vết đứt nhỏ (như vết nứt trên số) bằng khuôn chữ nhật đứng
    # Sau khi phép Open đã loại bỏ nhiễu nhỏ, thì phép Close sẽ giúp nối liền các vết nứt nhỏ trên các con số (như số 5 bị đứt quãng) bằng các khung chữ nhật 5x3 pixel (kernel_heal)


    # --- TÌM CONTOUR ---
    # Tìm tất cả các đường viền (contours) màu trắng trên nền đen bằng cách vẽ khung.
    # RETR_EXTERNAL: Chỉ lấy viền ngoài cùng (bỏ qua lỗ hổng bên trong số 0,8,6,9).
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ChAIN_APPROX_SIMPLE: Nén các đoạn thẳng ngang, dọc, chéo và chỉ giữ lại các điểm đầu cuối để tiết kiệm bộ nhớ.

    # --- LỌC BAN ĐẦU ---
    initial_rects = [] 
    # Tạo 1 danh sách để chứa các khung chữ nhật ban đầu tìm được từ contours   
    for cnt in contours:
        # Lặp qua từng contour tìm được
        x, y, w, h = cv2.boundingRect(cnt)
        # Take the bounding rectangle for each contour (tọa độ hình chữ nhật bao quanh cái đường viền) (x,y is the top-left corner, w is width, h is height)
        if w < 300 and h < 300: # Chỉ lấy những hình nhỏ hơn 300px
            # Logic: Lấy hết các mảnh vỡ, miễn là không phải bụi quá nhỏ
            if (h > 5 and w > 5): # Và phải lớn hơn 5px (để loại bỏ đốm li ti quá nhỏ trong hình).
                initial_rects.append((x, y, w, h)) 
                # Thêm vào danh sách ban đầu.

    print(f"[INFO] Tim thay {len(initial_rects)} manh vo ban dau.")
    # In ra số lượng mảnh tìm được.
    
    # --- GỘP BOX BẰNG LOGIC ---
    final_rects = merge_broken_parts(initial_rects)
    # Gọi hàm gộp lần 1: Nối các mảnh số 5 bị gãy
    
    
    # Chạy thêm 1 lần nữa cho chắc (đề phòng số bị gãy làm 3 khúc)
    final_rects = merge_broken_parts(final_rects)
    
    final_rects = remove_inside_boxes(final_rects)
    print(f"[INFO] Sau khi gop manh vo, con lai {len(final_rects)} so hoan chinh.")

   # --- CẮT & LƯU ẢNH ---
    final_digits = []
    # Danh sách chứa ảnh các số đã cắt xong
    
    valid_count = 0 
    # Biến đếm chuẩn để lưu tên file liên tục (0, 1, 2, 3...) không bị nhảy cóc nếu lỡ bỏ qua rác
    
    # Duyệt qua danh sách các hình chữ nhật đã gộp
    for (x, y, w, h) in final_rects:
        
        # 1. Diện tích quá nhỏ (< 150px) -> Bỏ
        # 2. Vừa lùn VÀ vừa hẹp (Bụi vuông < 25x25) -> Bỏ
        # 3. Quá ốm (Nét đứt mỏng dính w < 😎 -> Bỏ
        if (w * h < 150) or (h < 25 and w < 25) or (w < 8): 
            print(f"[-] Vut 1 manh vun (rac) tai X={x}") # Ghi không dấu cho khỏi lỗi Terminal
            continue 

        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Vẽ khung màu xanh lá cây lên ảnh hiển thị.
        
        cv2.putText(img_display, str(valid_count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Viết số thứ tự lên đầu mỗi khung (Dùng valid_count thay vì i)
        
        pad = 5
        # Đặt lề (padding) là 5 pixel
        roi = thresh[max(0, y-pad):min(thresh.shape[0], y+h+pad),
                     max(0, x-pad):min(thresh.shape[1], x+w+pad)]
        # Cắt vùng ảnh chứa số (ROI - Region Of Interest) từ ảnh đen trắng
        # Dùng max/min để đảm bảo không cắt lố ra ngoài mép ảnh
        
        # Nếu cắt ra vùng rỗng thì bỏ qua
        if roi.size == 0: continue

        # Square Padding
        h_roi, w_roi = roi.shape
        # Lấy chiều cao, chiều rộng của mảnh vừa cắt
        max_dim = max(h_roi, w_roi)
        # Tìm cạnh lớn nhất (để làm kích thước cho hình vuông)
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        # Tạo một bức ảnh vuông màu đen thui có kích thước bằng cạnh lớn nhất
        start_x = (max_dim - w_roi) // 2
        # Tính vị trí x để đặt số vào giữa
        start_y = (max_dim - h_roi) // 2
        # Tính vị trí y để đặt số vào giữa
        
        square_img[start_y:start_y+h_roi, start_x:start_x+w_roi] = roi
        # Dán mảnh số vừa cắt vào chính giữa cái nền đen đó

        final_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        # Resize ảnh vuông đó về kích thước chuẩn 28x28 pixel
        
        cv2.imwrite(f'{OUTPUT_DIR}/digit_{valid_count}.png', final_img)
        # Lưu ảnh ra file trong thư mục 'output_digit'
        
        final_digits.append(final_img)
        # Thêm vào danh sách để hiển thị
        
        valid_count += 1 
        # Tăng biến đếm lên 1 khi lưu thành công
    # --- SHOW KẾT QUẢ ---
    # Tạo khung hình lớn kích thước 12x6 inch.
    plt.figure(figsize=(12, 6))
    # Tạo ô vẽ số 1 (bên trái).
    plt.subplot(1, 2, 1)
    
    plt.title("Final Result (Merged Boxes)")
    
    # Hiển thị ảnh gốc đã vẽ khung xanh (phải chuyển màu từ BGR sang RGB thì matplotlib mới hiện đúng màu).
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    
    # Tạo ô vẽ số 2 (bên phải).
    plt.subplot(1, 2, 2)
    
    # Đặt tiêu đề.
    plt.title("Binary Input")
    
    # Hiển thị ảnh đen trắng.
    plt.imshow(thresh, cmap='gray')
    
    #Hiện cửa sổ kết quả lên màn hình.
    plt.show()

#
if __name__ == "__main__":
    process_image(INPUT_FILE)