"""
Player tracking module using ByteTrack for multi-object tracking.
"""

import cv2
import numpy as np
import time
from collections import defaultdict
import supervision as sv
# Updated imports for supervision package
from supervision import Detections
from supervision import ByteTrack
from typing import Dict, List, Tuple, Optional

class PlayerTracker:
    """
    Player tracker class using ByteTrack for multi-object tracking.
    """
    
    def __init__(self, config):
        """
        Initialize the player tracker.
        
        Args:
            config (dict): Configuration dictionary with tracking settings.
        """
        self.config = config
        
        # Initialize tracker based on the config
        tracker_type = config["tracker"].lower()
        
        if tracker_type == "bytetrack":
            # Use default initialization with no parameters
            self.tracker = ByteTrack()
            # Store config values for potential use elsewhere
            self.track_high_thresh = config["track_high_thresh"]
            self.track_buffer = config["track_buffer"]
            self.match_thresh = config["match_thresh"]
        else:
            raise ValueError(f"Tracker type '{tracker_type}' not supported.")
        
        # Dictionary to store track history
        self.track_history = defaultdict(list)
        self.player_info = {}  # Dictionary to store player info (jersey number, etc.)
        
    def update(self, frame, detections):
        """
        Update tracks with new detections.
        
        Args:
            frame (numpy.ndarray): Current video frame.
            detections (dict): Dictionary containing detection results.
            
        Returns:
            dict: Dictionary containing tracking results.
        """
        # Start timing for performance measurement
        start_time = time.time()
        
        # Convert detections to supervision format
        detection_list = detections["detections"]
        
        if not detection_list:
            tracking_time = time.time() - start_time
            return {
                "tracks": [],
                "tracking_time": tracking_time
            }
        
        # Extract detections data
        xyxy = np.array([d[:4] for d in detection_list])
        confidence = np.array([d[4] for d in detection_list])
        class_id = np.array([d[5] for d in detection_list])
        
        # Create a Detections object
        detections_obj = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Update tracker with new detections - using the current API
        # Note: The API might now be using track() instead of update_with_detections()
        try:
            # Try the newer API first
            tracks = self.tracker.track(detections_obj)
        except (AttributeError, TypeError):
            # Fall back to the older API if needed
            tracks = self.tracker.update_with_detections(detections_obj)
        
        # Update track history
        track_results = []
        if hasattr(tracks, 'tracker_id') and tracks.tracker_id is not None:
            for i, (xyxy_box, confidence_val, class_id_val, track_id) in enumerate(zip(
                tracks.xyxy,
                tracks.confidence,
                tracks.class_id,
                tracks.tracker_id,
            )):
                # Convert array elements to native Python types
                x1, y1, x2, y2 = map(int, xyxy_box)
                confidence_val = float(confidence_val)
                class_id_val = int(class_id_val)
                track_id = int(track_id)
                
                # Add current position to track history
                self.track_history[track_id].append((x1, y1, x2, y2))
                
                # Keep track history length bounded
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Add track to results
                track_results.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence_val,
                    "class_id": class_id_val,
                    "track_id": track_id
                })
        
        tracking_time = time.time() - start_time
        
        return {
            "tracks": track_results,
            "tracking_time": tracking_time
        }
    
    def get_track_history(self, track_id, max_length=None):
        """
        Get the history of a specific track.
        
        Args:
            track_id (int): The ID of the track.
            max_length (int, optional): Maximum number of history points to return.
            
        Returns:
            list: List of track history points.
        """
        history = self.track_history.get(track_id, [])
        if max_length is not None:
            history = history[-max_length:]
        return history
    
    def set_player_info(self, track_id, info):
        """
        Set information for a specific player track.
        
        Args:
            track_id (int): The ID of the track.
            info (dict): Dictionary containing player information (jersey number, etc.).
        """
        self.player_info[track_id] = info
    
    def get_player_info(self, track_id):
        """
        Get information for a specific player track.
        
        Args:
            track_id (int): The ID of the track.
            
        Returns:
            dict: Dictionary containing player information (jersey number, etc.).
        """
        return self.player_info.get(track_id, {}) 