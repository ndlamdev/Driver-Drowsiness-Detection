# Hệ thống Phát hiện Buồn ngủ của Tài xế (Driver Drowsiness Detection)

## 📋 Mô tả

Dự án này là một hệ thống phát hiện buồn ngủ của tài xế sử dụng Deep Learning và Computer Vision. Hệ thống sử dụng camera để theo dõi trạng thái mắt của tài xế trong thời gian thực và phát cảnh báo khi phát hiện dấu hiệu buồn ngủ (mắt nhắm).

## ✨ Tính năng

- **Phát hiện khuôn mặt và mắt tự động**: Sử dụng Haar Cascade để phát hiện khuôn mặt, mắt trái và mắt phải
- **Phân loại trạng thái mắt**: Sử dụng mô hình CNN để phân loại mắt đang mở hay nhắm
- **Hệ thống cảnh báo**: 
  - Phát âm thanh cảnh báo khi phát hiện buồn ngủ
  - Hiển thị khung màu đỏ nhấp nháy xung quanh video
- **Giao diện trực quan**: Hiển thị trạng thái mắt và điểm số cảnh báo trên video

## 🛠️ Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam hoặc camera tích hợp
- Windows/Linux/macOS

## 📦 Cài đặt

### 1. Clone hoặc tải dự án về máy

```bash
git clone https://github.com/ndlamdev/Driver-Drowsiness-Detection
cd Driver-Drowsiness-Detection
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

**Lưu ý**: Nếu gặp lỗi khi cài đặt, bạn có thể cài từng thư viện chính:

```bash
pip install tensorflow
pip install opencv-python
pip install pygame
pip install scikit-learn
pip install joblib
pip install numpy
pip install matplotlib
```

## 📁 Cấu trúc dự án

```
drowsidess_detection/
├── data/
│   ├── alarm.wav                    # File âm thanh cảnh báo
│   ├── cnncat2.keras                # Mô hình CNN đã được huấn luyện
│   ├── label_encoder.pkl            # Bộ mã hóa nhãn
│   └── data-haarcascades/           # Các file Haar Cascade cho phát hiện khuôn mặt/mắt
│       ├── haarcascade_frontalface_alt.xml
│       ├── haarcascade_lefteye_2splits.xml
│       └── haarcascade_righteye_2splits.xml
├── model/
│   ├── DriverDrowsinessDetection.py      # Class xử lý mô hình CNN
│   └── VideoDriverDrowsinessDetection.py  # Class xử lý video và phát hiện thời gian thực
├── main.py                          # File chính để chạy chương trình
└── requirements.txt                 # Danh sách các thư viện cần thiết
```

## 🚀 Hướng dẫn sử dụng

### Chạy chương trình

1. **Đảm bảo camera đã được kết nối và hoạt động**

2. **Chạy file main.py**:

```bash
python main.py
```

3. **Sử dụng chương trình**:
   - Chương trình sẽ mở cửa sổ video hiển thị camera
   - Đảm bảo khuôn mặt của bạn được nhìn thấy rõ trong khung hình
   - Hệ thống sẽ tự động phát hiện khuôn mặt và mắt
   - Trạng thái mắt (Open/Closed) sẽ được hiển thị ở góc dưới bên trái
   - Điểm số cảnh báo (Score) sẽ được hiển thị ở góc dưới bên phải

4. **Thoát chương trình**:
   - Nhấn phím `q` để thoát

### Cách hoạt động

- **Hệ thống đếm điểm (Score System)**:
  - Khi cả hai mắt nhắm: Điểm tăng lên (tối đa 30)
  - Khi có ít nhất một mắt mở: Điểm giảm xuống (tối thiểu 0)
  
- **Cảnh báo buồn ngủ**:
  - Khi điểm số vượt quá **15**: Hệ thống sẽ:
    - Phát âm thanh cảnh báo liên tục
    - Hiển thị khung màu đỏ nhấp nháy xung quanh video
    - Hiển thị trạng thái "Closed" trên màn hình

- **Ngừng cảnh báo**:
  - Khi điểm số giảm xuống dưới 15: Cảnh báo sẽ tự động tắt

## ⚙️ Tùy chỉnh

### Thay đổi camera

Nếu bạn muốn sử dụng camera khác (không phải camera mặc định), mở file `main.py` và thay đổi tham số `cam`:

```python
# Trong VideoDriverDrowsinessDetection.start()
# cam=0: Camera đầu tiên
# cam=1: Camera thứ hai
# cam=2: Camera thứ ba, v.v.
game.start(cam=0)  # Thay đổi số này
```

### Điều chỉnh ngưỡng cảnh báo

Trong file `model/VideoDriverDrowsinessDetection.py`, bạn có thể thay đổi:

- **Ngưỡng cảnh báo** (dòng 69): Thay đổi `score > 15` thành giá trị khác
- **Thời gian nhắm mắt tối đa** (dòng 60): Thay đổi `score > 30` để điều chỉnh điểm tối đa

### Thay đổi âm thanh cảnh báo

Thay thế file `data/alarm.wav` bằng file âm thanh cảnh báo khác của bạn (định dạng WAV).

## 🔧 Xử lý lỗi thường gặp

### Lỗi: "Cannot open camera"
- **Nguyên nhân**: Camera chưa được kết nối hoặc đang được sử dụng bởi ứng dụng khác
- **Giải pháp**: 
  - Kiểm tra kết nối camera
  - Đóng các ứng dụng đang sử dụng camera
  - Thử thay đổi số camera trong code (0, 1, 2...)

### Lỗi: "Model chưa được load"
- **Nguyên nhân**: File mô hình không tồn tại hoặc đường dẫn sai
- **Giải pháp**: 
  - Đảm bảo file `data/cnncat2.keras` tồn tại
  - Kiểm tra đường dẫn trong `main.py`

### Lỗi: "Không tìm thấy file label_encoder.pkl"
- **Nguyên nhân**: File encoder bị thiếu
- **Giải pháp**: Đảm bảo file `data/label_encoder.pkl` tồn tại cùng thư mục với file model

### Lỗi liên quan đến thư viện
- **Giải pháp**: Cài đặt lại các thư viện:
  ```bash
  pip install --upgrade tensorflow opencv-python pygame scikit-learn
  ```

### Chương trình không phát hiện được khuôn mặt
- **Nguyên nhân**: 
  - Ánh sáng không đủ
  - Khuôn mặt quá xa camera
  - Góc camera không phù hợp
- **Giải pháp**:
  - Cải thiện ánh sáng
  - Điều chỉnh vị trí ngồi
  - Đảm bảo khuôn mặt chiếm phần lớn khung hình

## 📊 Thông số kỹ thuật

- **Mô hình**: CNN (Convolutional Neural Network)
- **Kích thước ảnh đầu vào**: 64x64 pixels (grayscale)
- **Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV với Haar Cascade
- **Audio**: Pygame

## 🔬 Huấn luyện mô hình (Nâng cao)

Nếu bạn muốn huấn luyện lại mô hình với dữ liệu của riêng mình:

1. Chuẩn bị dữ liệu ảnh mắt (mở/nhắm) trong cấu trúc thư mục:
   ```
   dataset/
   ├── train/
   │   ├── Open/
   │   └── Close/
   └── test/
       ├── Open/
       └── Close/
   ```

2. Sử dụng class `DriverDrowsinessDetection`:
   ```python
   from model.DriverDrowsinessDetection import DriverDrowsinessDetection
   
   # Khởi tạo model
   detector = DriverDrowsinessDetection()
   detector.init_model_to_train(total_class=2, img_size=64)
   
   # Load và chuẩn bị dữ liệu
   # ... (xử lý dữ liệu)
   
   # Huấn luyện
   detector.train(x_train, y_train, epochs=15, batch_size=32)
   
   # Lưu model
   detector.save_model("data/cnncat2.keras")
   ```

## 📝 Ghi chú

- Đảm bảo có đủ ánh sáng khi sử dụng để hệ thống phát hiện chính xác
- Hệ thống hoạt động tốt nhất khi khuôn mặt chiếm 30-50% khung hình
- Tránh đeo kính râm hoặc che khuất mắt
- Nếu sử dụng kính, đảm bảo kính không phản chiếu ánh sáng quá mạnh

## 📄 License

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu.

## 👨‍💻 Tác giả

Dự án phát hiện buồn ngủ của tài xế sử dụng Deep Learning.

## 📧 Liên hệ

Nếu bạn có bất kỳ câu hỏi, đề xuất hoặc muốn liên hệ, vui lòng:

- **Email**: [ndlam.dev@gmail.com](mailto:ndlam.dev@gmail.com)
- **LinkedIn**: [https://www.linkedin.com/in/ndlamdev](https://www.linkedin.com/in/ndlamdev)
- **Số điện thoại**: +84 855354919

---

**Lưu ý an toàn**: Hệ thống này chỉ là công cụ hỗ trợ. Không nên hoàn toàn phụ thuộc vào hệ thống này khi lái xe. Luôn luôn tập trung và nghỉ ngơi đầy đủ trước khi lái xe.

