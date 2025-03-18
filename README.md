# Basketball Player Detection & Jersey Number Recognition

An AI-powered basketball player detection system that processes video input and outputs video with bounding boxes around players, displaying their jersey numbers.

## Features

- **Player Detection & Tracking**: Uses YOLOv8 for player detection and DeepSORT for multi-object tracking
- **Jersey Number Recognition**: Implements OCR techniques to extract jersey numbers
- **Overlay & Video Processing**: Draws bounding boxes and overlays jersey numbers
- **Performance Optimization**: Utilizes GPU acceleration for faster processing
- **Input & Output Handling**: Supports various video formats and resolutions

## Project Structure

```
player-detector/
├── player_detector/            # Main package
│   ├── detection/              # Player detection models
│   ├── tracking/               # Multi-object tracking
│   ├── recognition/            # Jersey number recognition
│   ├── utils/                  # Utility functions
│   └── visualization/          # Visualization tools
├── models/                     # Pre-trained model weights
├── data/                       # Sample data and test videos
├── main.py                     # Main entry point
├── config.py                   # Configuration settings
└── requirements.txt            # Project dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ToyeshC/player-detector.git
   cd player-detector
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download required model weights:
   ```
   # This will automatically download the YOLOv8x model (about 130MB)
   python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
   
   # For faster but less accurate detection, use a smaller model:
   # python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"  # 6MB
   # python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"  # 21MB
   # python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"  # 48MB
   # python -c "from ultralytics import YOLO; YOLO('yolov8l.pt')"  # 86MB
   ```

## Usage

### Basic Usage

```python
python main.py --input path/to/video.mp4 --output processed_video.mp4
```

### Advanced Options

```python
python main.py --input path/to/video.mp4 --output processed_video.mp4 --conf 0.5 --device 0 --optimization onnx
```

### Macbook or CPU-only Options

For systems with limited resources (like MacBook Air):

```python
python main.py --input path/to/video.mp4 --output processed_video.mp4 --model yolov8n.pt --conf 0.5 --device cpu --frame-skip 3 --no-display
```

## Configuration

Edit `config.py` to customize:
- Detection confidence thresholds
- OCR settings
- Tracking parameters
- Visualization options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
