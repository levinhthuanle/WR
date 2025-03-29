import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv11 đã được huấn luyện trước cho ước lượng tư thế
model = YOLO('yolo11n-pose.pt')

# Mở webcam (source=0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Thực hiện dự đoán tư thế trên khung hình
    results = model(frame)

    # Vẽ kết quả lên khung hình
    annotated_frame = results[0].plot()

    # Hiển thị khung hình đã được chú thích
    cv2.imshow('YOLOv11 Pose Tracking', annotated_frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
