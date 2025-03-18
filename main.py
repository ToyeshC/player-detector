#!/usr/bin/env python3
"""
Basketball Player Detection and Jersey Number Recognition System.

This script processes a video to detect basketball players, track them,
and recognize their jersey numbers.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import project modules
from player_detector.detection import PlayerDetector
from player_detector.tracking import PlayerTracker
from player_detector.recognition import JerseyRecognizer
from player_detector.visualization import Visualizer
from player_detector.utils.video_utils import VideoSource, VideoWriter
import config as cfg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Basketball Player Detection and Jersey Number Recognition')
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input video file or camera index')
    parser.add_argument('--output', '-o', type=str, default='output/processed_video.mp4',
                        help='Path to output video file')
    
    # Detection arguments
    parser.add_argument('--model', type=str, default=None,
                        help='Path to detection model (overrides config)')
    parser.add_argument('--conf', type=float, default=None,
                        help='Detection confidence threshold (overrides config)')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image size for detection (overrides config)')
    
    # Tracking arguments
    parser.add_argument('--tracker', type=str, default=None,
                        help='Tracker type: bytetrack (overrides config)')
    
    # Recognition arguments
    parser.add_argument('--ocr', type=str, default=None,
                        help='OCR method: easyocr, tesseract (overrides config)')
    parser.add_argument('--ocr-conf', type=float, default=None,
                        help='OCR confidence threshold (overrides config)')
    
    # Performance arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cpu, 0, 1, etc. (overrides config)')
    parser.add_argument('--optimization', type=str, default=None,
                        help='Optimization method: None, onnx (overrides config)')
    parser.add_argument('--frame-skip', type=int, default=None,
                        help='Process every Nth frame (overrides config)')
    
    # Visualization arguments
    parser.add_argument('--show-stats', action='store_true',
                        help='Show processing statistics in output video')
    parser.add_argument('--show-tracking', action='store_true',
                        help='Show tracking history in output video')
    parser.add_argument('--show-roi', action='store_true',
                        help='Show ROI boxes for jersey number recognition')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable real-time display')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """
    Update configuration from command-line arguments.
    
    Args:
        config: Configuration dictionary.
        args: Command-line arguments.
        
    Returns:
        Updated configuration dictionary.
    """
    # Detection settings
    if args.model is not None:
        config["DETECTION"]["model"] = args.model
    if args.conf is not None:
        config["DETECTION"]["confidence"] = args.conf
    if args.img_size is not None:
        config["DETECTION"]["img_size"] = args.img_size
    if args.device is not None:
        config["DETECTION"]["device"] = args.device
    
    # Tracking settings
    if args.tracker is not None:
        config["TRACKING"]["tracker"] = args.tracker
    
    # Recognition settings
    if args.ocr is not None:
        config["RECOGNITION"]["ocr_method"] = args.ocr
    if args.ocr_conf is not None:
        config["RECOGNITION"]["confidence"] = args.ocr_conf
    
    # Performance settings
    if args.optimization is not None:
        config["OPTIMIZATION"]["method"] = args.optimization
    if args.frame_skip is not None:
        config["OPTIMIZATION"]["frame_skip"] = args.frame_skip
    
    return config


def process_video(args):
    """
    Process the input video and produce the output video.
    
    Args:
        args: Command-line arguments.
    """
    # Load configuration
    config = {
        "DETECTION": cfg.DETECTION,
        "TRACKING": cfg.TRACKING,
        "RECOGNITION": cfg.RECOGNITION,
        "VISUALIZATION": cfg.VISUALIZATION,
        "IO": cfg.IO,
        "OPTIMIZATION": cfg.OPTIMIZATION
    }
    
    # Update configuration from command-line arguments
    config = update_config_from_args(config, args)
    
    # Initialize components
    print("Initializing components...")
    detector = PlayerDetector(config["DETECTION"])
    tracker = PlayerTracker(config["TRACKING"])
    recognizer = JerseyRecognizer(config["RECOGNITION"])
    visualizer = Visualizer(config["VISUALIZATION"])
    
    # Set visualization options
    if args.show_tracking:
        visualizer.set_track_history_visibility(True)
    
    # Initialize video source
    video_source = VideoSource(args.input, config["OPTIMIZATION"])
    
    # Initialize video writer
    fps = video_source.fps
    resolution = (video_source.width, video_source.height)
    output_config = config["IO"]
    video_writer = VideoWriter(args.output, fps, resolution, output_config)
    
    # Optimize model if requested
    if config["OPTIMIZATION"]["method"]:
        print(f"Optimizing model using {config['OPTIMIZATION']['method']}...")
        detector.optimize_model(
            method=config["OPTIMIZATION"]["method"],
            use_fp16=config["OPTIMIZATION"]["use_fp16"]
        )
    
    # Process frames
    print(f"Processing video from {args.input}...")
    
    # Dictionary to store jersey info for each track
    jersey_info = {}
    
    # Initialize progress bar if processing a file
    progress_bar = None
    if not isinstance(args.input, int) and video_source.frame_count > 0:
        progress_bar = tqdm(total=video_source.frame_count, unit='frame')
    
    try:
        while True:
            # Start timing for FPS calculation
            frame_start_time = time.time()
            
            # Read frame
            ret, frame = video_source.read()
            if not ret:
                break
            
            # Process frame
            # 1. Detect players
            detection_results = detector.detect(frame)
            
            # 2. Track players
            tracking_results = tracker.update(frame, detection_results)
            
            # 3. Recognize jersey numbers for each track
            roi_boxes = []
            for track in tracking_results["tracks"]:
                track_id = track["track_id"]
                
                # Skip if we already have a high-confidence recognition for this track
                if track_id in jersey_info and jersey_info[track_id].get("confidence", 0) > 0.8:
                    continue
                
                # Recognize jersey number
                recognition_results = recognizer.recognize(frame, track)
                roi_boxes.append(recognition_results["roi_coords"])
                
                # Update jersey info if we got a number
                if recognition_results["jersey_number"]:
                    jersey_info[track_id] = {
                        "jersey_number": recognition_results["jersey_number"],
                        "confidence": recognition_results["confidence"]
                    }
            
            # 4. Visualize results
            vis_frame = frame.copy()
            
            # Draw tracks and jersey numbers
            vis_frame = visualizer.draw_tracks(
                vis_frame, 
                tracking_results["tracks"], 
                tracker.track_history if args.show_tracking else None,
                jersey_info
            )
            
            # Draw ROI boxes if requested
            if args.show_roi:
                vis_frame = visualizer.draw_roi_boxes(vis_frame, roi_boxes)
            
            # Calculate performance stats
            frame_time = time.time() - frame_start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            
            # Draw stats if requested
            if args.show_stats:
                stats = {
                    "detection_time": detection_results.get("inference_time", 0),
                    "tracking_time": tracking_results.get("tracking_time", 0),
                    "recognition_time": recognition_results.get("recognition_time", 0) if "recognition_results" in locals() else 0,
                    "total_time": frame_time
                }
                vis_frame = visualizer.draw_processing_stats(vis_frame, stats)
                vis_frame = visualizer.draw_fps(vis_frame, frame_time)
            
            # Write frame to output video
            video_writer.write(vis_frame)
            
            # Display frame if not disabled
            if not args.no_display:
                cv2.imshow('Player Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)
            else:
                # Print progress for live video
                print(f"\rProcessing: {fps:.1f} FPS", end="")
        
        # Print newline after progress updates
        print()
        
        # Print cache stats
        cache_stats = recognizer.get_cache_stats()
        print(f"OCR Cache: {cache_stats['cache_size']} entries, {cache_stats['hit_rate']:.1f}% hit rate")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted.")
    
    finally:
        # Clean up resources
        if progress_bar is not None:
            progress_bar.close()
        
        video_source.release()
        video_writer.release()
        
        if not args.no_display:
            cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {args.output}")


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Process video
    process_video(args)


if __name__ == "__main__":
    main() 