# Football Video Dataset to YOLO Format Converter

Dự án này chuyển đổi dataset video bóng đá có annotations thành định dạng YOLO để training các mô hình object detection. Dataset được extract từ video files (.mp4) và annotation files (.json) để tạo ra images và labels tương ứng theo chuẩn YOLO.

## 📁 Cấu trúc dự án

```
Deeplearning_videoDataset_footbal_createFormat/
├── create_yolo_format_dataset.py    # Script chính để chuyển đổi dataset
├── dataset.py                       # VOC Dataset class cho PyTorch
├── data/
│   └── football/                    # Thư mục chứa dữ liệu gốc
│       ├── video1/
│       │   ├── *.mp4               # Video files
│       │   └── *.json              # Annotation files
│       └── video2/
│           ├── *.mp4
│           └── *.json
└── football_yolo/                   # Output directory (được tạo tự động)
    └── train/
        ├── images/                  # Extracted frames
        └── labels/                  # YOLO format labels
```

## 🎯 Mục đích

Dự án này giải quyết bài toán:
- **Input**: Video bóng đá (.mp4) + Annotation files (.json) 
- **Output**: Dataset YOLO format với images và labels để train object detection models
- **Target objects**: Phát hiện các đối tượng trong video bóng đá (players, ball, etc.)

## 🔧 Yêu cầu hệ thống

### Dependencies
```bash
pip install opencv-python
pip install torch torchvision
pip install numpy
```

### Cấu trúc dữ liệu đầu vào

**JSON Annotation Format:**
```json
{
  "images": [
    {
      "width": 1920,
      "height": 1080,
      "id": 1
    }
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 3,
      "bbox": [x, y, width, height]
    }
  ]
}
```

**Categories:**
- `category_id = 3`: Class 0 (trong YOLO format)
- `category_id > 3`: Class 1 (trong YOLO format)

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu

Đặt dữ liệu theo cấu trúc:
```
data/football/
├── video_folder_1/
│   ├── video.mp4
│   └── annotations.json
├── video_folder_2/
│   ├── video.mp4
│   └── annotations.json
```

### 2. Chạy script chuyển đổi

```bash
python create_yolo_format_dataset.py
```

### 3. Kết quả

Script sẽ tạo ra:
- **Images**: `football_yolo/train/images/` - Các frame được extract từ video
- **Labels**: `football_yolo/train/labels/` - YOLO format labels tương ứng

**YOLO Label Format:**
```
class_id x_center y_center width height
```
Tất cả giá trị được normalize về [0,1]

## 📊 Chi tiết kỹ thuật

### create_yolo_format_dataset.py

**Chức năng chính:**
1. **Video Processing**: Đọc từng frame từ video sử dụng OpenCV
2. **Annotation Parsing**: Parse JSON annotations và lọc objects (category_id > 2)
3. **Coordinate Conversion**: Chuyển đổi bbox từ absolute coordinates sang YOLO format
4. **Data Export**: Lưu images (.jpg) và labels (.txt)

**Công thức chuyển đổi coordinates:**
```python
# From absolute to normalized
xmin_norm = xmin / image_width
ymin_norm = ymin / image_height
width_norm = width / image_width  
height_norm = height / image_height

# Calculate center point
x_center = xmin_norm + width_norm/2
y_center = ymin_norm + height_norm/2
```

**Class mapping:**
```python
if obj["category_id"] == 3:
    cls = 0  # First class
else:
    cls = 1  # Second class
```

### dataset.py

**VOCDataset Class:**
- Kế thừa từ `torchvision.datasets.VOCDetection`
- Hỗ trợ 21 categories chuẩn Pascal VOC
- Preprocessing cho PyTorch training
- Coordinate scaling theo image resize

**Features:**
- Data augmentation với transforms
- Bounding box scaling
- Label encoding
- PyTorch tensor output

## 🎮 Cách chạy thử nghiệm

### Test VOC Dataset:
```python
from dataset import VOCDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Setup transforms
train_transform = Compose([
    Resize((416, 416)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = VOCDataset(
    root="data/voc", 
    year="2012", 
    image_set="trainval", 
    download=False, 
    transform=train_transform
)

# Get sample
image, target = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Target: {target}")
```

## 🔍 Debugging & Kiểm tra

Trong `create_yolo_format_dataset.py` có phần code test (đã comment) để visualize bounding boxes:

```python
# Uncomment để kiểm tra bbox
# xmax = xmin + width
# ymax = ymin + height
# cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
#               color=(255, 0, 255), thickness=1)
```

## 📈 Workflow

1. **Input Processing**: Scan thư mục `data/football` tìm video và annotation files
2. **Frame Extraction**: Extract từng frame từ video
3. **Annotation Matching**: Match annotations với frame tương ứng
4. **Object Filtering**: Chỉ lấy objects có `category_id > 2`
5. **Format Conversion**: Chuyển bbox sang YOLO format
6. **File Export**: Lưu image + label files

## 🎯 Kết quả mong đợi

- **Images**: Format JPG, được đánh số theo pattern `{video_id}_{frame_number}.jpg`
- **Labels**: Format TXT, mỗi dòng là một object với format YOLO
- **Structure**: Sẵn sàng để training với YOLO models (YOLOv5, YOLOv8, etc.)

## 🔧 Customization

### Thay đổi output path:
```python
output_path = "your_custom_path"  # Thay đổi trong create_yolo_format_dataset.py
```

### Thay đổi class mapping:
```python
# Modify class assignment logic
if obj["category_id"] == 3:
    cls = 0
elif obj["category_id"] == 4:
    cls = 1
# Add more classes as needed
```

### Thay đổi image format:
```python
cv2.imwrite(os.path.join(output_path, "images", "{}_{}.png".format(video_id+1, counter)), frame)
```

## 📝 Notes

- **Memory Usage**: Xử lý video lớn có thể tốn nhiều RAM
- **Processing Time**: Thời gian phụ thuộc vào độ dài video và số lượng annotations
- **Quality**: Output images giữ nguyên chất lượng của video gốc
- **Compatibility**: Output tương thích với tất cả YOLO implementations

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👤 Author

**QuyDatSadBoy**
- GitHub: [@QuyDatSadBoy](https://github.com/QuyDatSadBoy)

---

*Project này là một phần của pipeline xử lý dữ liệu cho training object detection models trên video bóng đá.*
