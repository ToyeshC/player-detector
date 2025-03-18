"""
Player detection module using YOLOv8 for basketball player detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import os
from pathlib import Path

class PlayerDetector:
    """
    Player detector class using YOLOv8 for basketball player detection.
    """
    
    def __init__(self, config, model_path=None):
        """
        Initialize the player detector.
        
        Args:
            config (dict): Configuration dictionary with detection settings.
            model_path (str, optional): Path to the model weights. If None, uses the model specified in config.
        """
        self.config = config
        
        # Initialize model
        model_path = model_path or config["model"]
        print(f"Loading YOLOv8 model: {model_path}")
        
        # Check if model exists locally, download if not
        if not os.path.exists(model_path) and not model_path.startswith(('http://', 'https://')):
            # Check in the models directory
            models_dir = Path("models")
            model_in_dir = models_dir / model_path
            if model_in_dir.exists():
                model_path = str(model_in_dir)
        
        self.model = YOLO(model_path)
        
        # Set model parameters
        self.device = config["device"]
        self.conf_threshold = config["confidence"]
        self.classes = config["classes"]  # Class IDs to detect
        self.img_size = config["img_size"]
        
        # Warm up the model
        self._warmup()
        
    def _warmup(self):
        """Warm up the model with a dummy image."""
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)
        print("Model warmed up!")
        
    def detect(self, frame):
        """
        Detect players in the given frame.
        
        Args:
            frame (numpy.ndarray): Input frame to detect players in.
            
        Returns:
            list: List of detections, each containing [x1, y1, x2, y2, confidence, class_id]
        """
        # Start timing for performance measurement
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            classes=self.classes,
            device=self.device,
            verbose=False
        )
        
        # Process results
        detections = []
        
        # Get detections from the first (and only) image
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Add to detections
                detections.append([int(x1), int(y1), int(x2), int(y2), confidence, class_id])
        
        inference_time = time.time() - start_time
        
        return {
            "detections": detections,
            "inference_time": inference_time,
            "model_name": getattr(self.model, "names", str(self.model))  # Use 'names' if available or string representation
        }
    
    def optimize_model(self, method="onnx", use_fp16=False):
        """
        Optimize the model for inference.
        
        Args:
            method (str): Optimization method: "onnx" or "tensorrt".
            use_fp16 (bool): Whether to use FP16 precision for optimization.
            
        Returns:
            bool: True if optimization was successful, False otherwise.
        """
        if method == "onnx":
            try:
                # Export to ONNX
                # Use model path or a default name if 'name' attribute is not available
                model_name = getattr(self.model, "__file__", "model")
                if isinstance(model_name, str):
                    model_basename = os.path.basename(model_name).split('.')[0]
                else:
                    model_basename = "yolov8_model"
                    
                onnx_path = Path("models") / f"{model_basename}_optimized.onnx"
                self.model.export(format="onnx", dynamic=True, simplify=True, fp16=use_fp16)
                print(f"Model exported to ONNX: {onnx_path}")
                return True
            except Exception as e:
                print(f"Error optimizing model: {e}")
                return False
        else:
            print(f"Optimization method '{method}' not supported.")
            return False 