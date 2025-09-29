# Clothing Size Prediction Pipeline

Hệ thống dự đoán kích thước quần áo với pipeline hoàn chỉnh: **Image + Height → YOLO Pose → Body Measurements → Random Forest Size Prediction**

## 🎯 Pipeline Flow

```
1. 📸 Input: Image + Real Height (cm)
        ↓
2. 👁️ YOLO Pose Detection (Extract 17 keypoints)
        ↓  
3. 📏 Body Measurement Estimation (From keypoints + height ratio)
        ↓
4. 🧠 Random Forest Prediction (Size classification)
        ↓
5. 📊 Output: Predicted Size + Confidence
```

## 🚀 Tính năng

- **YOLO11n Pose Detection**: Phát hiện 17 keypoints trên cơ thể người
- **Intelligent Body Measurement**: Tính toán 5 thông số cơ thể từ keypoints
- **Random Forest Classification**: Dự đoán size với độ tin cậy cao

## 📊 Đầu ra dự đoán

Hệ thống sẽ dự đoán các thông số sau:

1. **Shoulder Width** (Chiều rộng vai): Khoảng cách giữa hai vai
2. **Belly** (Vòng bụng): Chu vi vùng bụng/eo  
3. **Neck Circumference** (Vòng cổ): Chu vi cổ
4. **Hip Circumference** (Vòng hông): Chu vi vùng hông
5. **Shirt Length** (Chiều dài áo): Từ vai đến hông
6. **Size** (Kích thước): S, M, L, XL, XXL

## 🛠️ Setup và Cài đặt

### Yêu cầu
```bash
pip install ultralytics opencv-python pandas scikit-learn numpy joblib
```

### Chuẩn bị Model Files
```bash
# 1. Train Random Forest model
python export_rf_weights.py

# 2. Download YOLO pose model (tự động tải khi chạy lần đầu)
# File sẽ được lưu tại: models/yolo11n-pose.pt
```

### Cấu trúc thư mục
```
Coding/
├── main.py                          # 🎯 Chương trình chính (NEW PIPELINE)
├── demo_pipeline.py                 # 🧪 Demo pipeline mới  
├── export_rf_weights.py             # 🏋️ Train Random Forest model
├── models/
│   ├── yolo11n-pose.pt             # YOLO pose detection model
│   ├── random_forest_model.pkl     # Trained Random Forest model
│   └── random_forest_weights.txt   # Model weights export
├── tests/                          # Thư mục chứa ảnh test
├── output/                         # Thư mục lưu kết quả
└── clothing_size_dataset_synthetic_2000.csv  # Dataset training
```

## 🎯 Cách sử dụng

### 1. Chạy Pipeline chính (main.py)

```bash
# Interactive mode
python main.py

# Command line mode  
python main.py "tests/thanh.jpg" 170
```

### 2. Chạy Demo Pipeline

```bash
python demo_pipeline.py
```

### 3. Sử dụng trong Python code

```python
from main import ClothingSizePredictionPipeline

# Khởi tạo pipeline
pipeline = ClothingSizePredictionPipeline()

# Chạy complete pipeline
results = pipeline.run_complete_pipeline(
    image_path="tests/thanh.jpg",
    real_height_cm=170,
    output_path="output/result.jpg"
)

# Kết quả
print(f"Predicted size: {results['predicted_size']}")
print("Measurements:", results['measurements'])
```

## 📝 Input và Output

### Input
- **Hình ảnh**: Ảnh chụp toàn thân người đứng thẳng
- **Chiều cao thực tế**: Chiều cao thực của người trong ảnh (đơn vị: cm)

### Output
- **Kích thước cơ thể**: 5 thông số đo (cm)
- **Size dự đoán**: S, M, L, XL, XXL  
- **Độ tin cậy**: Xác suất cho từng size
- **Ảnh đã annotate**: Ảnh với keypoints được đánh dấu

## 🔧 Cách hoạt động chi tiết

### Step 1: YOLO Pose Detection
- Input: Hình ảnh
- Process: YOLO11n-pose phát hiện 17 keypoints
- Output: Dictionary các tọa độ keypoints

### Step 2: Body Measurement Estimation  
- Input: Keypoints + chiều cao thực tế
- Process: 
  - Tính tỷ lệ pixel/cm từ chiều cao
  - Tính khoảng cách giữa các keypoints
  - Convert sang thông số cơ thể thực tế
- Output: 5 thông số cơ thể (cm)

### Step 3: Random Forest Size Prediction
- Input: 5 thông số cơ thể  
- Process: Random Forest classifier
- Output: Size prediction + confidence

## 📊 Độ chính xác

- **YOLO Pose**: Độ chính xác keypoint detection ~90%+
- **Random Forest**: Accuracy ~85-90% trên test set
- **Overall Pipeline**: Phụ thuộc vào chất lượng ảnh và pose detection

## 🎨 Ví dụ kết quả

```
🎯 CLOTHING SIZE PREDICTION PIPELINE
============================================================
Flow: Image + Height → YOLO Pose → Body Measurements → RF Size Prediction

📋 INPUT:
   • Image: tests/thanh.jpg
   • Real Height: 170 cm

👁️ Step 1: Extracting keypoints from image...
✅ Detected 16 keypoints

📏 Step 2: Estimating body measurements...
   • Shoulder Width: 41.2 cm
   • Belly Circumference: 87.5 cm
   • Neck Circumference: 36.8 cm  
   • Hip Circumference: 93.1 cm
   • Shirt Length: 69.7 cm

🧠 Step 3: Predicting clothing size with Random Forest...
✅ Predicted size: L (confidence: 0.742)

🎉 FINAL PREDICTION RESULTS
============================================================
📏 PREDICTED SIZE: L

📐 ESTIMATED BODY MEASUREMENTS:
   • Shoulder Width: 41.2 cm
   • Belly: 87.5 cm
   • Neck Circumference: 36.8 cm
   • Hip Circumference: 93.1 cm
   • Shirt Length: 69.7 cm

📊 SIZE PREDICTION CONFIDENCE:
   • L: 74.2%
   • M: 18.3%  
   • XL: 6.1%
   • S: 1.2%
   • XXL: 0.2%

💾 Results saved to: output/clothing_size_prediction.jpg
```

## ⚠️ Lưu ý

- **Ảnh chất lượng tốt**: Người đứng thẳng, rõ nét, toàn thân
- **Chiều cao chính xác**: Rất quan trọng cho việc tính toán tỷ lệ
- **Keypoint detection**: Cần ít nhất 12-15 keypoints được phát hiện
- **Lighting**: Ánh sáng đầy đủ để YOLO detect tốt

## 🚧 Files chính

- **`main.py`**: 🎯 Complete pipeline mới
- **`export_rf_weights.py`**: 🏋️ Train Random Forest model  
- **`demo_pipeline.py`**: 🧪 Demo và test pipeline
- **`estimate_size_model.py`**: 📊 So sánh các ML models
- **`body_measurement_predictor.py`**: 🔧 Old implementation (legacy)

## 📞 Troubleshooting

Nếu gặp lỗi:

1. **Model not found**: Chạy `python export_rf_weights.py` trước
2. **YOLO model download**: Lần đầu sẽ tự động download ~6MB  
3. **No person detected**: Kiểm tra chất lượng ảnh, lighting
4. **KeyError**: Một số keypoints không detect được, thử ảnh khác

## 🎯 Performance Tips

- **Ảnh input**: 640x640 pixel optimal cho YOLO
- **Pose**: Đứng thẳng, tay để dọc thân người
- **Background**: Background đơn giản giúp detection tốt hơn
- **Distance**: Chụp cách khoảng 2-3 mét để có toàn thân