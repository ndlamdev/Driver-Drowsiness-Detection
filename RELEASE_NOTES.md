# Release Notes - Version 1.0.0

## 🚀 Hệ thống Phát hiện Buồn ngủ của Tài xế

### Mô tả ngắn gọn

Hệ thống phát hiện buồn ngủ của tài xế sử dụng Deep Learning và Computer Vision để theo dõi trạng thái mắt trong thời gian thực và phát cảnh báo khi phát hiện dấu hiệu buồn ngủ.

### ✨ Tính năng chính

- ✅ **Phát hiện khuôn mặt và mắt tự động** - Sử dụng Haar Cascade
- ✅ **Phân loại trạng thái mắt** - Mô hình CNN với độ chính xác cao
- ✅ **Hệ thống cảnh báo thông minh** - Âm thanh + khung nhấp nháy
- ✅ **Giao diện trực quan** - Hiển thị trạng thái và điểm số real-time
- ✅ **Hỗ trợ nhiều camera** - Tùy chỉnh camera dễ dàng

### 🎯 Tính năng kỹ thuật

- **Mô hình**: CNN (Convolutional Neural Network)
- **Framework**: TensorFlow/Keras 2.20.0
- **Computer Vision**: OpenCV 4.12.0
- **Real-time processing**: Xử lý video stream từ webcam
- **Hệ thống điểm số**: Tự động theo dõi và cảnh báo khi điểm > 15

### 📦 Cài đặt nhanh

```bash
# Clone repository
git clone <repository-url>
cd drowsidess_detection

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy chương trình
python main.py
```

### 🎮 Sử dụng

1. Đảm bảo camera đã được kết nối
2. Chạy `python main.py`
3. Đảm bảo khuôn mặt được nhìn thấy rõ trong khung hình
4. Hệ thống sẽ tự động phát hiện và cảnh báo khi phát hiện buồn ngủ
5. Nhấn `q` để thoát

### 📋 Yêu cầu hệ thống

- Python 3.8+
- Webcam hoặc camera tích hợp
- Windows/Linux/macOS

### 🔧 Dependencies chính

- TensorFlow 2.20.0
- OpenCV 4.12.0
- Pygame 2.6.1
- scikit-learn 1.7.2
- NumPy 2.2.6

### 📝 Tài liệu

Xem file `README.md` để biết hướng dẫn chi tiết về:
- Cài đặt đầy đủ
- Tùy chỉnh và nâng cao
- Xử lý lỗi thường gặp
- Huấn luyện mô hình

### ⚠️ Lưu ý quan trọng

Hệ thống này chỉ là công cụ hỗ trợ. Không nên hoàn toàn phụ thuộc vào hệ thống này khi lái xe. Luôn luôn tập trung và nghỉ ngơi đầy đủ trước khi lái xe.

### 📧 Liên hệ

- **Email**: ndlam.dev@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/ndlamdev
- **Số điện thoại**: +84 855354919

---

**Phiên bản**: 1.0.0  
**Ngày phát hành**: 2024-12-XX  
**License**: Educational and Research purposes

