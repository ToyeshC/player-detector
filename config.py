"""
Configuration settings for the basketball player detection and jersey number recognition system.
"""

# Detection settings
DETECTION = {
    "model": "yolov8x.pt",  # Model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
    "confidence": 0.5,      # Confidence threshold for object detection
    "classes": [0],         # Class IDs to detect (0 for person in COCO dataset)
    "device": "auto",       # Device to use for inference (auto, cpu, 0, 1, etc.)
    "img_size": 640         # Input image size for the model
}

# Tracking settings
TRACKING = {
    "tracker": "bytetrack",  # Tracker to use (bytetrack, deepsort, botsort)
    "track_high_thresh": 0.6,  # High detection threshold
    "track_low_thresh": 0.1,   # Low detection threshold
    "new_track_thresh": 0.7,   # New track threshold
    "track_buffer": 30,        # How many frames to keep tracks alive without detection
    "match_thresh": 0.8        # Threshold for feature similarity matching
}

# Jersey number recognition settings
RECOGNITION = {
    "ocr_method": "easyocr",  # OCR method (easyocr, tesseract)
    "confidence": 0.6,         # Confidence threshold for OCR
    "preprocessing": True,     # Whether to use preprocessing techniques
    "min_text_size": 10,       # Minimum size of text to detect (in pixels)
    "text_detection_area": {   # Region of interest relative to bounding box (from top)
        "top_offset": 0.0,      # Offset from top of bounding box (as percentage)
        "height_percentage": 0.4 # Percentage of bounding box height to consider
    },
    "max_jersey_number": 99    # Maximum jersey number to detect (to filter out noise)
}

# Visualization settings
VISUALIZATION = {
    "bbox_thickness": 2,       # Thickness of bounding box lines
    "bbox_color": (0, 255, 0), # Color of bounding box (BGR format)
    "text_color": (255, 255, 255), # Color of text (BGR format)
    "text_bg_color": (0, 0, 0), # Background color of text (BGR format)
    "text_size": 0.8,          # Size of text relative to bounding box
    "show_track_id": False,    # Whether to show track ID alongside jersey number
    "show_confidence": False   # Whether to show detection/OCR confidence
}

# Input/Output settings
IO = {
    "supported_video_formats": [".mp4", ".avi", ".mov", ".mkv"],
    "default_output_format": ".mp4",
    "default_fps": 30,
    "default_resolution": (1920, 1080),
    "save_frames": False,      # Whether to save individual frames
    "frames_dir": "output/frames"
}

# Performance optimization settings
OPTIMIZATION = {
    "method": None,           # Optimization method (None, onnx, tensorrt)
    "batch_size": 1,          # Batch size for inference
    "frame_skip": 0,          # Process every Nth frame (0 = process all frames)
    "use_fp16": False,        # Whether to use FP16 precision
    "num_workers": 4          # Number of workers for data loading
} 