"""
Visualization module for rendering bounding boxes and jersey numbers.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time

class Visualizer:
    """
    Visualizer class for rendering bounding boxes and jersey numbers.
    """
    
    def __init__(self, config):
        """
        Initialize the visualizer.
        
        Args:
            config (dict): Configuration dictionary with visualization settings.
        """
        self.config = config
        self.bbox_thickness = config["bbox_thickness"]
        self.bbox_color = config["bbox_color"]
        self.text_color = config["text_color"]
        self.text_bg_color = config["text_bg_color"]
        self.text_size = config["text_size"]
        self.show_track_id = config["show_track_id"]
        self.show_confidence = config["show_confidence"]
        
        # For visualizing track history
        self.show_track_history = False
        self.track_history_length = 20
        
        # For FPS counter
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        
    def draw_tracks(self, frame, tracks, track_history=None, jersey_info=None):
        """
        Draw bounding boxes and jersey numbers for tracked players.
        
        Args:
            frame (numpy.ndarray): Current video frame.
            tracks (list): List of track dictionaries.
            track_history (dict, optional): Dictionary of track histories.
            jersey_info (dict, optional): Dictionary of jersey number information.
            
        Returns:
            numpy.ndarray: Frame with visualization overlays.
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw each track
        for track in tracks:
            # Get bounding box and track ID
            bbox = track["bbox"]
            track_id = track["track_id"]
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)
            
            # Prepare text to display
            display_text = ""
            
            # Add jersey number if available
            if jersey_info and track_id in jersey_info:
                jersey_number = jersey_info[track_id].get("jersey_number")
                confidence = jersey_info[track_id].get("confidence", 0)
                
                if jersey_number:
                    display_text += f"#{jersey_number}"
                    
                    # Add confidence if requested
                    if self.show_confidence and confidence > 0:
                        display_text += f" ({confidence:.2f})"
            
            # Add track ID if requested
            if self.show_track_id:
                if display_text:
                    display_text += f" ID:{track_id}"
                else:
                    display_text = f"ID:{track_id}"
            
            # Draw text if we have something to display
            if display_text:
                # Calculate text size for proper background
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = (x2 - x1) * self.text_size / 200  # Scale text based on bbox width
                scale = max(0.5, min(scale, 1.0))  # Limit scale between 0.5 and 1.0
                thickness = max(1, int(scale * 2))
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(display_text, font, scale, thickness)
                
                # Draw text background
                cv2.rectangle(
                    vis_frame, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width + 10, y1), 
                    self.text_bg_color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    vis_frame, 
                    display_text, 
                    (x1 + 5, y1 - 5), 
                    font, 
                    scale, 
                    self.text_color, 
                    thickness
                )
            
            # Draw track history if available and enabled
            if self.show_track_history and track_history and track_id in track_history:
                history = track_history[track_id][-self.track_history_length:]
                
                # Draw trajectory line
                for i in range(1, len(history)):
                    # Get start and end points (center of bounding boxes)
                    prev_x1, prev_y1, prev_x2, prev_y2 = history[i-1]
                    curr_x1, curr_y1, curr_x2, curr_y2 = history[i]
                    
                    prev_center_x = (prev_x1 + prev_x2) // 2
                    prev_center_y = (prev_y1 + prev_y2) // 2
                    curr_center_x = (curr_x1 + curr_x2) // 2
                    curr_center_y = (curr_y1 + curr_y2) // 2
                    
                    # Calculate color based on position in history (older=faded, newer=bright)
                    color_intensity = int(255 * (i / len(history)))
                    color = (0, color_intensity, 0)
                    
                    # Draw line segment
                    cv2.line(
                        vis_frame,
                        (prev_center_x, prev_center_y),
                        (curr_center_x, curr_center_y),
                        color,
                        thickness=2
                    )
        
        return vis_frame
    
    def draw_roi_boxes(self, frame, roi_boxes):
        """
        Draw region of interest boxes for jersey number recognition.
        
        Args:
            frame (numpy.ndarray): Current video frame.
            roi_boxes (list): List of ROI box coordinates.
            
        Returns:
            numpy.ndarray: Frame with ROI boxes.
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw each ROI box
        for roi_box in roi_boxes:
            if roi_box:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for ROI
        
        return vis_frame
    
    def draw_fps(self, frame, processing_time):
        """
        Draw FPS counter on the frame.
        
        Args:
            frame (numpy.ndarray): Current video frame.
            processing_time (float): Time taken to process the frame.
            
        Returns:
            numpy.ndarray: Frame with FPS counter.
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Calculate FPS
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            vis_frame, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return vis_frame
    
    def draw_processing_stats(self, frame, stats):
        """
        Draw processing statistics on the frame.
        
        Args:
            frame (numpy.ndarray): Current video frame.
            stats (dict): Dictionary containing processing statistics.
            
        Returns:
            numpy.ndarray: Frame with processing statistics.
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Extract statistics
        detection_time = stats.get("detection_time", 0)
        tracking_time = stats.get("tracking_time", 0)
        recognition_time = stats.get("recognition_time", 0)
        total_time = stats.get("total_time", 0)
        
        # Draw statistics
        y_offset = 60
        line_height = 30
        
        stats_text = [
            f"Detection: {detection_time*1000:.1f} ms",
            f"Tracking: {tracking_time*1000:.1f} ms",
            f"Recognition: {recognition_time*1000:.1f} ms",
            f"Total: {total_time*1000:.1f} ms"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(
                vis_frame, 
                text, 
                (10, y_offset + i * line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
        
        return vis_frame
    
    def set_track_history_visibility(self, show=True, length=20):
        """
        Set the visibility of track history.
        
        Args:
            show (bool): Whether to show track history.
            length (int): Number of history points to display.
        """
        self.show_track_history = show
        self.track_history_length = length 