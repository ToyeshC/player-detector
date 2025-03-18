"""
Utility module for video input/output operations.
"""

import cv2
import os
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Generator

class VideoSource:
    """
    Video source class for reading video frames.
    """
    
    def __init__(self, source, config=None):
        """
        Initialize the video source.
        
        Args:
            source (str): Path to video file or camera index.
            config (dict, optional): Configuration dictionary.
        """
        self.config = config or {}
        self.source = source
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_skip = self.config.get("frame_skip", 0) if config else 0
        
        # Initialize video capture
        self._init_capture()
        
    def _init_capture(self):
        """Initialize video capture from the source."""
        # Check if source is a camera index (integer or string digit)
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            self.source = int(self.source)
            print(f"Opening camera: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            # Assume file path
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"Video file not found: {self.source}")
            
            print(f"Opening video file: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
        
        # Check if video capture initialized successfully
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames")
    
    def __iter__(self):
        """Iterator protocol for video frames."""
        return self
    
    def __next__(self):
        """Get the next frame."""
        # Skip frames if needed
        skip_count = 0
        while skip_count < self.frame_skip:
            self.cap.grab()
            skip_count += 1
        
        # Read the next frame
        ret, frame = self.cap.read()
        
        if not ret:
            # Reset video to beginning if it's a camera
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    raise StopIteration
            else:
                raise StopIteration
        
        return frame
    
    def read(self):
        """
        Read the next frame.
        
        Returns:
            tuple: (ret, frame) where ret is True if frame was read successfully.
        """
        try:
            frame = next(self)
            return True, frame
        except StopIteration:
            return False, None
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_progress(self):
        """
        Get the current progress in the video.
        
        Returns:
            float: Progress percentage (0-100).
        """
        if self.frame_count > 0:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return (current_frame / self.frame_count) * 100
        return 0
    
    def get_frame_position(self):
        """
        Get the current frame position.
        
        Returns:
            int: Current frame position.
        """
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def set_frame_position(self, frame_number):
        """
        Set the current frame position.
        
        Args:
            frame_number (int): Frame number to set.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def __del__(self):
        """Clean up resources."""
        self.release()


class VideoWriter:
    """
    Video writer class for writing processed frames to a video file.
    """
    
    def __init__(self, output_path, fps=30, resolution=None, config=None):
        """
        Initialize the video writer.
        
        Args:
            output_path (str): Path to output video file.
            fps (int, optional): Frames per second.
            resolution (tuple, optional): Resolution (width, height).
            config (dict, optional): Configuration dictionary.
        """
        self.config = config or {}
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.frame_count = 0
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save frames if requested
        self.save_frames = self.config.get("save_frames", False) if config else False
        if self.save_frames:
            self.frames_dir = self.config.get("frames_dir", "output/frames") if config else "output/frames"
            os.makedirs(self.frames_dir, exist_ok=True)
    
    def init_writer(self, frame):
        """
        Initialize the video writer with the first frame's properties.
        
        Args:
            frame (numpy.ndarray): First frame to write.
        """
        if self.writer is not None:
            return
        
        # Get frame dimensions if resolution not specified
        if self.resolution is None:
            height, width = frame.shape[:2]
            self.resolution = (width, height)
        
        # Determine output codec based on file extension
        filename, ext = os.path.splitext(self.output_path)
        if ext.lower() == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext.lower() == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            # Default to MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        print(f"Initializing video writer: {self.output_path}, {self.resolution}, {self.fps} FPS")
        self.writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            self.resolution
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
    
    def write(self, frame):
        """
        Write a frame to the output video.
        
        Args:
            frame (numpy.ndarray): Frame to write.
        """
        # Initialize writer if needed
        if self.writer is None:
            self.init_writer(frame)
        
        # Resize frame if necessary
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)
        
        # Write frame to video file
        self.writer.write(frame)
        self.frame_count += 1
        
        # Save individual frame if requested
        if self.save_frames:
            frame_filename = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
    
    def release(self):
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Video saved to: {self.output_path} ({self.frame_count} frames)")
    
    def __del__(self):
        """Clean up resources."""
        self.release()


def resize_frame(frame, target_size):
    """
    Resize a frame to the target size while maintaining aspect ratio.
    
    Args:
        frame (numpy.ndarray): Input frame.
        target_size (int): Target size (longest dimension).
        
    Returns:
        numpy.ndarray: Resized frame.
    """
    height, width = frame.shape[:2]
    
    # Calculate new dimensions
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    return resized_frame 