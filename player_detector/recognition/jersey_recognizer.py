"""
Jersey number recognition module using OCR techniques.
"""

import cv2
import numpy as np
import time
from PIL import Image
import re
import easyocr
import pytesseract
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional, Union

class JerseyRecognizer:
    """
    Jersey number recognizer class using OCR techniques.
    """
    
    def __init__(self, config):
        """
        Initialize the jersey number recognizer.
        
        Args:
            config (dict): Configuration dictionary with recognition settings.
        """
        self.config = config
        self.ocr_method = config["ocr_method"].lower()
        self.confidence_threshold = config["confidence"]
        self.min_text_size = config["min_text_size"]
        self.max_jersey_number = config["max_jersey_number"]
        self.top_offset = config["text_detection_area"]["top_offset"]
        self.height_percentage = config["text_detection_area"]["height_percentage"]
        self.use_preprocessing = config["preprocessing"]
        
        # Initialize OCR engine
        if self.ocr_method == "easyocr":
            print("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(['en'], gpu=self._is_gpu_available())
        elif self.ocr_method == "tesseract":
            # Check if tesseract is installed
            if not self._check_tesseract_installed():
                raise RuntimeError("Tesseract OCR is not installed or not in PATH. Please install it.")
            print("Using Tesseract OCR...")
        else:
            raise ValueError(f"OCR method '{self.ocr_method}' not supported.")
            
        # Cache of recognized numbers to reduce redundant processing
        self.recognition_cache = {}
        self.cache_hit_count = 0
        self.total_queries = 0
        
    def _check_tesseract_installed(self):
        """Check if Tesseract OCR is installed."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except pytesseract.TesseractNotFoundError:
            return False
    
    def _is_gpu_available(self):
        """Check if GPU is available for EasyOCR."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _preprocess_roi(self, roi):
        """
        Preprocess the region of interest for better OCR results.
        
        Args:
            roi (numpy.ndarray): Region of interest from the frame.
            
        Returns:
            numpy.ndarray: Preprocessed region of interest.
        """
        if not self.use_preprocessing:
            return roi
        
        # Resize if too small
        h, w = roi.shape[:2]
        if h < 30 or w < 30:
            # Scale up while maintaining aspect ratio
            scale = max(30 / h, 30 / w)
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        
        # Dilate to enhance text
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        
        return dilated
    
    def _extract_numbers(self, text):
        """
        Extract numbers from OCR text.
        
        Args:
            text (str): Text extracted by OCR.
            
        Returns:
            str: Extracted jersey number or None if no valid number found.
        """
        # Remove spaces and non-alphanumeric characters
        text = re.sub(r'[^0-9]', '', text)
        
        # Check if we have a valid jersey number
        if text and len(text) <= 3:  # Most jersey numbers are 1-3 digits
            number = int(text)
            if 0 <= number <= self.max_jersey_number:
                return text
        
        return None
    
    def _get_roi_from_bbox(self, frame, bbox):
        """
        Get the region of interest (ROI) from the bounding box.
        
        Args:
            frame (numpy.ndarray): Current video frame.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].
            
        Returns:
            numpy.ndarray: Region of interest.
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate the jersey number ROI based on configuration
        # Focus on the upper part of the bounding box where jersey numbers typically are
        roi_height = int((y2 - y1) * self.height_percentage)
        roi_y1 = int(y1 + (y2 - y1) * self.top_offset)
        roi_y2 = roi_y1 + roi_height
        
        # Ensure ROI is within frame boundaries
        h, w = frame.shape[:2]
        roi_x1 = max(0, x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(w, x2)
        roi_y2 = min(h, roi_y2)
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        return roi, (roi_x1, roi_y1, roi_x2, roi_y2)
    
    def _recognize_with_easyocr(self, roi):
        """
        Recognize text using EasyOCR.
        
        Args:
            roi (numpy.ndarray): Region of interest.
            
        Returns:
            tuple: (jersey_number, confidence)
        """
        # Generate a cache key based on ROI content
        cache_key = hash(roi.tobytes())
        
        # Check cache first
        if cache_key in self.recognition_cache:
            self.cache_hit_count += 1
            return self.recognition_cache[cache_key]
        
        # Increment total queries counter
        self.total_queries += 1
        
        # Recognize text with EasyOCR
        result = self.reader.readtext(roi)
        
        # Process results
        best_number = None
        best_confidence = 0
        
        for detection in result:
            bbox, text, confidence = detection
            
            if confidence >= self.confidence_threshold:
                jersey_number = self._extract_numbers(text)
                if jersey_number and confidence > best_confidence:
                    best_number = jersey_number
                    best_confidence = confidence
        
        # Cache the result
        self.recognition_cache[cache_key] = (best_number, best_confidence)
        
        return best_number, best_confidence
    
    def _recognize_with_tesseract(self, roi):
        """
        Recognize text using Tesseract OCR.
        
        Args:
            roi (numpy.ndarray): Region of interest.
            
        Returns:
            tuple: (jersey_number, confidence)
        """
        # Generate a cache key based on ROI content
        cache_key = hash(roi.tobytes())
        
        # Check cache first
        if cache_key in self.recognition_cache:
            self.cache_hit_count += 1
            return self.recognition_cache[cache_key]
        
        # Increment total queries counter
        self.total_queries += 1
        
        # Configure Tesseract for digits only and get confidence
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 -c tessedit_create_hocr=1'
        
        # Convert OpenCV image to PIL Image for Tesseract
        pil_img = Image.fromarray(roi)
        
        # Recognize text with Tesseract
        text = pytesseract.image_to_string(pil_img, config=config)
        
        # Get confidence data
        hocr = pytesseract.image_to_pdf_or_hocr(pil_img, extension='hocr', config=config)
        confidence = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        
        # Extract confidence value (approximate from overall OCR confidence)
        conf_value = confidence.get('orientation_conf', 0) / 100.0
        
        # Process results
        jersey_number = self._extract_numbers(text)
        
        # Cache the result
        self.recognition_cache[cache_key] = (jersey_number, conf_value)
        
        return jersey_number, conf_value
    
    def recognize(self, frame, track):
        """
        Recognize the jersey number for a tracked player.
        
        Args:
            frame (numpy.ndarray): Current video frame.
            track (dict): Track information including bounding box.
            
        Returns:
            dict: Recognition results.
        """
        # Start timing for performance measurement
        start_time = time.time()
        
        # Get bounding box and track ID
        bbox = track["bbox"]
        track_id = track["track_id"]
        
        # Extract ROI for jersey number recognition
        roi, roi_coords = self._get_roi_from_bbox(frame, bbox)
        
        # Skip if ROI is too small
        if roi.size == 0 or roi.shape[0] < self.min_text_size or roi.shape[1] < self.min_text_size:
            return {
                "jersey_number": None,
                "confidence": 0,
                "roi_coords": roi_coords,
                "recognition_time": time.time() - start_time
            }
        
        # Preprocess ROI
        processed_roi = self._preprocess_roi(roi)
        
        # Recognize jersey number using the selected OCR method
        if self.ocr_method == "easyocr":
            jersey_number, confidence = self._recognize_with_easyocr(processed_roi)
        elif self.ocr_method == "tesseract":
            jersey_number, confidence = self._recognize_with_tesseract(processed_roi)
        else:
            jersey_number, confidence = None, 0
        
        recognition_time = time.time() - start_time
        
        return {
            "jersey_number": jersey_number,
            "confidence": confidence,
            "roi_coords": roi_coords,
            "recognition_time": recognition_time
        }
    
    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics.
        """
        hit_rate = 0
        if self.total_queries > 0:
            hit_rate = self.cache_hit_count / self.total_queries * 100
            
        return {
            "cache_size": len(self.recognition_cache),
            "cache_hits": self.cache_hit_count,
            "total_queries": self.total_queries,
            "hit_rate": hit_rate
        } 