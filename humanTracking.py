import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Thư viện để theo dõi ID của từng người

# Khởi tạo mô hình YOLOv11 Pose
model = YOLO("yolo11n-pose.pt")  # Thay bằng phiên bản phù hợp nếu cần

# Khởi tạo bộ theo dõi SORT
tracker = Sort()

# Dictionary lưu lịch sử đường di chuyển của từng người
track_history = {}

def draw_tracking_line(frame, track_id, centroid):
    """
    Vẽ đường di chuyển của từng người theo ID.
    """
    if track_id not in track_history:
        track_history[track_id] = []  # Khởi tạo danh sách tọa độ

    # Lưu lại vị trí di chuyển
    track_history[track_id].append(centroid)

    # Giữ lại tối đa 30 điểm để tránh vẽ quá nhiều
    if len(track_history[track_id]) > 30:
        track_history[track_id].pop(0)

    # Vẽ đường nối giữa các điểm
    for i in range(1, len(track_history[track_id])):
        cv2.line(frame, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy mô hình YOLO để phát hiện người
        results = model(frame)

        detections = []
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()  # Lấy keypoints tọa độ (x, y)

            for person in keypoints:
                # Lấy trung tâm của hông (keypoint số 8 hoặc 11)
                if len(person) >= 12:
                    x_center = int((person[8][0] + person[11][0]) / 2)
                    y_center = int((person[8][1] + person[11][1]) / 2)
                    detections.append([x_center, y_center, x_center + 1, y_center + 1, 1])  # (x1, y1, x2, y2, score)

        # Cập nhật bộ theo dõi ID của từng người
        track_ids = tracker.update(np.array(detections))

        # Vẽ bounding box + ID
        for track in track_ids:
            x1, y1, x2, y2, track_id = map(int, track)

            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            draw_tracking_line(frame, track_id, centroid)

            # Hiển thị ID
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

        # Hiển thị kết quả
        cv2.imshow("Human Tracking with YOLOv11", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
