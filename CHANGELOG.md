# Changelog

## [1.0.0] - 2025-11-4

### ✨ Tính năng mới

- **Hệ thống phát hiện buồn ngủ của tài xế**: Ứng dụng phát hiện buồn ngủ sử dụng Deep Learning và Computer Vision
- **Phát hiện khuôn mặt và mắt tự động**: Sử dụng Haar Cascade để phát hiện khuôn mặt, mắt trái và mắt phải trong thời gian thực
- **Phân loại trạng thái mắt**: Mô hình CNN phân loại mắt đang mở hay nhắm với độ chính xác cao
- **Hệ thống cảnh báo thông minh**:
  - Phát âm thanh cảnh báo khi phát hiện buồn ngủ
  - Hiển thị khung màu đỏ nhấp nháy xung quanh video để thu hút sự chú ý
  - Hệ thống đếm điểm (Score System) để theo dõi trạng thái mắt liên tục
- **Giao diện trực quan**: Hiển thị trạng thái mắt (Open/Closed) và điểm số cảnh báo trên video
- **Hỗ trợ nhiều camera**: Có thể tùy chỉnh chọn camera để sử dụng

### 🎯 Tính năng kỹ thuật

- **Mô hình CNN**: Sử dụng Convolutional Neural Network với kiến trúc:
  - 3 lớp Conv2D với MaxPooling2D
  - Lớp Dense với Dropout để tránh overfitting
  - Softmax activation cho phân loại
- **Xử lý ảnh**: 
  - Resize ảnh về 64x64 pixels (grayscale)
  - Normalization và preprocessing tự động
- **Real-time detection**: Xử lý video stream từ webcam với tốc độ cao
- **Hệ thống điểm số**: 
  - Tự động tăng điểm khi cả hai mắt nhắm
  - Tự động giảm điểm khi có ít nhất một mắt mở
  - Ngưỡng cảnh báo: 15 điểm

### 📦 Dependencies

- **TensorFlow/Keras 2.20.0**: Framework Deep Learning cho mô hình CNN
- **OpenCV 4.12.0**: Xử lý video và phát hiện khuôn mặt/mắt
- **Pygame 2.6.1**: Phát âm thanh cảnh báo
- **scikit-learn 1.7.2**: Label encoding và preprocessing
- **NumPy 2.2.6**: Xử lý mảng và tính toán
- **joblib 1.5.2**: Lưu và tải mô hình

### 📁 Cấu trúc dự án

- `main.py`: File chính để chạy chương trình
- `model/DriverDrowsinessDetection.py`: Class xử lý mô hình CNN (train, evaluate, predict)
- `model/VideoDriverDrowsinessDetection.py`: Class xử lý video và phát hiện thời gian thực
- `data/cnncat2.keras`: Mô hình CNN đã được huấn luyện
- `data/label_encoder.pkl`: Bộ mã hóa nhãn
- `data/alarm.wav`: File âm thanh cảnh báo
- `data/data-haarcascades/`: Các file Haar Cascade cho phát hiện khuôn mặt/mắt

### 🔧 Cải tiến

- Tối ưu hóa hiệu suất xử lý video
- Cải thiện độ chính xác phát hiện mắt
- Giao diện người dùng trực quan và dễ sử dụng

### 📝 Documentation

- README.md đầy đủ với hướng dẫn chi tiết
- Hướng dẫn cài đặt và sử dụng
- Hướng dẫn xử lý lỗi thường gặp
- Hướng dẫn tùy chỉnh và nâng cao

### 🐛 Bug fixes

- Sửa lỗi phát âm thanh cảnh báo lặp lại
- Tối ưu hóa việc phát hiện khuôn mặt trong điều kiện ánh sáng yếu

### 🔒 Bảo mật & An toàn

- Không lưu trữ hoặc truyền dữ liệu video ra ngoài
- Xử lý hoàn toàn local, không cần kết nối internet
- Lưu ý an toàn: Hệ thống chỉ là công cụ hỗ trợ, không thay thế sự tập trung của tài xế

### 👥 Đóng góp

Dự án này được phát triển cho mục đích giáo dục và nghiên cứu.

### 📧 Liên hệ

- **Email**: ndlam.dev@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/ndlamdev
- **Số điện thoại**: +84 855354919

---

**Lưu ý**: Đây là phiên bản đầu tiên của dự án. Các phiên bản tiếp theo sẽ được cập nhật trong file này.

