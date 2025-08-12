# YOLOv8 Squirrel Detection Project

A comprehensive squirrel detection system using YOLOv8 (You Only Look Once version 8), fine-tuned specifically for detecting squirrels in various outdoor environments including parks, backyards, forests, and urban settings.

## Project Overview

This project demonstrates modern computer vision capabilities by creating a robust squirrel detection application that can accurately identify and locate squirrels in both static images and video streams across various scenarios including different lighting conditions, seasons, poses (sitting, climbing, jumping), and backgrounds (trees, ground, feeders, buildings).

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- ultralytics>=8.0.0
- opencv-python>=4.5.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- PyYAML>=6.0
- Pillow>=8.0.0
- matplotlib>=3.3.0
- torch>=1.9.0
- torchvision>=0.10.0

## Project Structure

```
├── src/                          # Source code (stuff I used to make the models)
│   ├── main.py                   # Main training and evaluation script
│   ├── squirrel_trainer.py       # Core training functionality
│   ├── downloader.py            # Image downloading utilities
│   └── download_wrapper.py      # Dataset preparation wrapper
├── yolo_squirrel_project/        # Project workspace
│   ├── dataset/                 # Training data
│   │   ├── train/               # Training images and labels
│   │   ├── val/                 # Validation images and labels
│   │   ├── test/                # Test images and labels
│   │   └── dataset.yaml         # Dataset configuration
│   └── runs/                    # Training and evaluation results
├── sample_images/               # Sample test images
├── demo.py                      # Quick demo script
├── evaluate_model.py           # Model evaluation script
├── best_model.pt               # Trained model weights
└── requirements.txt            # Dependencies
```

## Steps to train from scratch

### 1. Download and Prepare Dataset

If you have a CSV file with squirrel image URLs and IDs (I recommend exporting some tables from iNaturalist):

```bash
# Download all images to training folder
python src/download_wrapper.py download_all your_dataset.csv

# Download specific number of images to different splits
python src/download_wrapper.py download your_dataset.csv 100 train
python src/download_wrapper.py download your_dataset.csv 50 val
python src/download_wrapper.py download your_dataset.csv 25 test
```

### 2. Annotate Images

I used LabelImg to create bounding box annotations:

- **LabelImg**: https://github.com/tzutalin/labelImg

**Annotation Format**: YOLO format with class_id=0 for squirrels
```
0 center_x center_y width height
```
Example: `0 0.5 0.5 0.3 0.4`

### 3. Train the Model

```bash
python src/main.py
```

Choose from the interactive menu:
1. **Continue training** from existing model
2. **Start fresh training**
3. **Evaluate existing model**
4. **Test sample predictions**

### Or if you just want to use the model I packaged

Test the trained model on a single image:

```bash
python demo.py --image path/to/your/image.jpg --model best_model.pt
```

### 5. Evaluate Model Performance

Run comprehensive evaluation on test set:

```bash
python evaluate_model.py --images sample_images/ --model best_model.pt
```

## Training Configuration

Default training parameters can be modified in `src/main.py`:

```python
epochs = 50              # Training epochs
imgsz = 640             # Input image size
batch = 8               # Batch size
device = 'cpu'          # Device ('cpu', 'cuda', 'auto')
patience = 20           # Early stopping patience
save_period = 10        # Save checkpoint interval
```

## Model Performance

The system provides comprehensive evaluation metrics:

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision across IoU 0.5-0.95
- **Precision**: True positive rate
- **Recall**: Detection completeness
- **Confusion Matrix**: Classification accuracy visualization
- **PR Curves**: Precision-Recall relationship analysis

### Performance Interpretation
- **mAP50 > 0.8**: Excellent performance
- **mAP50 > 0.6**: Good performance with room for improvement
- **mAP50 > 0.4**: Moderate performance, consider more training data
- **mAP50 < 0.4**: Poor performance, needs more data or different approach

## Usage Examples

### Training a New Model

```bash
# Prepare dataset
python src/download_wrapper.py download_all squirrel_dataset.csv

# Annotate images using LabelImg or similar tools
# Images: yolo_squirrel_project/dataset/train/images/
# Labels: yolo_squirrel_project/dataset/train/labels/

# Start training
python src/main.py
# Select option 2: "Start fresh training"
```

### Continuing Training

```bash
python src/main.py
# Select option 1: "Continue training from latest model"
```

### Batch Evaluation

```bash
# Evaluate model on custom image directory
python evaluate_model.py --images /path/to/test/images/ --model best_model.pt

# Results saved to evaluation_results.json
```

### Real-time Inference

```bash
# Single image prediction
python demo.py --image squirrel_photo.jpg

# Results saved to demo_results/predictions/
```

## Dataset Management

### Check Dataset Statistics

```bash
python src/download_wrapper.py stats
```

### Prepare Empty Annotation Files

```bash
python src/download_wrapper.py prepare train
python src/download_wrapper.py prepare val
python src/download_wrapper.py prepare test
```

## References

- **YOLOv8**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- **Computer Vision**: Object detection and deep learning techniques
- **Wildlife Detection**: Applications in ecological research and monitoring

## Sources

Yaseen, M. (2024). What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector. arXiv preprint arXiv:2408.15857.

Terven, J., Córdova-Esparza, D. M., & Romero-González, J. A. (2023). A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS. Machine Learning and Knowledge Extraction, 5(4), 1680-1716.

Hendriks, P., & Bosch, J. (2024). Automatic object detection for behavioural research using YOLOv8. Behavior Research Methods, 56(5), 4951-4971.

Su, X., Zhao, J., Yang, K., & Gao, S. (2023). YOLO-SE: Improved YOLOv8 for Remote Sensing Object Detection and Recognition. Applied Sciences, 13(24), 12977.

Reis, D., Kupec, J., Hong, J., & Daoudi, A. (2023). Real-Time Flying Object Detection with YOLOv8. arXiv preprint arXiv:2305.09972.
