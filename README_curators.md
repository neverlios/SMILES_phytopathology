
# 🍅 Tomato Plant Pathology Detection using Computer Vision  

> This project develops a **computer vision pipeline** for automated detection of phytopathologies in tomato plants using real-world field images. The system identifies disease symptoms from camera captures, enabling early intervention for improved crop management.

# Problem Statement  
Tomato crops face significant threats from phytopathologies that reduce yield and quality. Current disease identification methods rely on manual scouting - a time-consuming process requiring specialized expertise that often leads to delayed treatment. Existing automated solutions struggle with real-world field conditions including lighting variations, occlusions, and symptom similarity across diseases. This project addresses these limitations by developing robust deep learning models trained on authentic agricultural imagery to provide accurate, real-time disease detection directly in field environments.

# 🎯 Objective

- Develop high-accuracy detection models for tomato plant diseases
- Create a modular computer vision pipeline adaptable to new pathologies
- Enable real-time inference suitable for field deployment

# 📦 Key Features

- Support for multiple state-of-the-art detection architectures
- Data augmentation pipeline for agricultural imagery
- Model optimization for edge deployment
- Comprehensive evaluation metrics
- Dockerized inference API

# 📁 Project Structure
```bash
tomato-disease-detection/
├── README.md                      # Project documentation
├── configs/                       # Model configurations
│   ├── yolo.yaml                  
│   └── faster_rcnn.yaml
├── data/
│   ├── raw_images/                # Original field images
│   ├── augmented/                 # Augmented training data
│   ├── annotations/               # Bounding box annotations
│   └── test_set/                  # Validation images
├── models/                        # Trained model weights
├── src/
│   ├── data_processing.py         # Dataset preparation
│   ├── augmentation.py            # Image transformations
│   ├── train.py                   # Model training
│   ├── evaluate.py                # Performance metrics
│   ├── inference.py               # Prediction pipeline
│   └── utils.py                   # Helper functions
├── notebooks/                     # Exploratory analysis
│   └── EDA.ipynb                  
├── api/                           # Deployment
│   └── app.py                     # Flask/Docker API
└── tests/                         # Unit tests
    └── test_metrics.py
```

# 📥 Input Data Format
```json
{
  "image_path": "/data/field_123.jpg",
  "model_type": "yolov8",
  "confidence_threshold": 0.65,
  "output_format": "json"
}
```

# 📤 Output Data Format
```json
{
  "image_id": "field_123.jpg",
  "detections": [
    {
      "disease": "early_blight",
      "confidence": 0.92,
      "bounding_box": [x_min, y_min, x_max, y_max],
      "severity": "moderate"
    },
    {
      "disease": "leaf_mold",
      "confidence": 0.78,
      "bounding_box": [x_min, y_min, x_max, y_max],
      "severity": "mild"
    }
  ],
  "timestamp": "2025-07-14T14:30:00Z"
}
```

# 🧠 Suggested Detection Models
| Model | Strengths | Implementation |
|-------|-----------|----------------|
| **YOLOv8** | Real-time performance | Ultralytics implementation |
| **Faster R-CNN** | High accuracy | Torchvision |
| **DETR** | Transformer architecture | HuggingFace Transformers |
| **EfficientDet** | Edge optimization | TensorFlow Hub |

# 📊 Evaluation Metrics
| Metric | Purpose |
|--------|---------|
| **mAP@0.5** | Primary detection accuracy |
| **F1-Score** | Precision-recall balance |
| **IoU** | Bounding box quality |
| **Inference Speed** | FPS on target hardware |
| **False Positive Rate** | Error analysis |

# 🛠️ Tools & Libraries
- `PyTorch`, `TensorFlow` - Deep learning frameworks
- `OpenCV` - Image processing
- `Albumentations` - Data augmentation
- `FiftyOne` - Dataset visualization
- `Docker` - Containerization
- `ONNXRuntime` - Model optimization

# 📚 Validation Dataset
Create annotated test set with:
- Minimum 500 field images
- Balanced pathology distribution
- Bounding box annotations (COCO format)
- Disease severity labels
- Multiple lighting/weather conditions

# 🧪 Sample Usage
```bash
python src/inference.py --image field_sample.jpg --model yolov8
```

**Output Preview**:
```json
{
  "detections": [
    {
      "disease": "late_blight",
      "confidence": 0.94,
      "bounding_box": [125, 308, 201, 352],
      "severity": "severe"
    }
  ]
}
```

# 📝 Setup Instructions
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/tomato-disease-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download dataset to `data/raw_images`
4. Start training:
   ```bash
   python src/train.py --config configs/yolo.yaml
   ```

# 🔗 Related Resources
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [COCO Annotation Format](https://cocodataset.org/#format-data)
- [Tomato  dataset](https://universe.roboflow.com/thesis-team-ana/tomato-disease-detection-izemp/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

# ✅ Validation Criteria
1. Installation without errors
2. Training pipeline executes successfully
3. mAP@0.5 > 0.75 on validation set
4. Inference speed > 15 FPS on test hardware
5. Docker API returns predictions within 500ms

# 📜 License
MIT License
