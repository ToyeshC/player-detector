#!/usr/bin/env python3
"""
Example script to demonstrate the basketball player detection and tracking system.

This script provides a simple way to run the system with default settings.
"""

import os
import sys
import argparse
import time
import config as cfg
from player_detector.detection import PlayerDetector
from player_detector.tracking import PlayerTracker
from player_detector.recognition import JerseyRecognizer
from player_detector.visualization import Visualizer
from player_detector.utils.video_utils import VideoSource, VideoWriter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Basketball Player Detection Example')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input video file or camera index')
    parser.add_argument('--output', '-o', type=str, default='output/example_output.mp4',
                        help='Path to output video file')
    return parser.parse_args()


def run_example(input_path, output_path):
    """
    Run a simple example of the basketball player detection system.
    
    Args:
        input_path (str): Path to input video file or camera index
        output_path (str): Path to output video file
    """
    print(f"Running example with input: {input_path}, output: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Initialize components with default settings
    detector = PlayerDetector(cfg.DETECTION)
    tracker = PlayerTracker(cfg.TRACKING)
    recognizer = JerseyRecognizer(cfg.RECOGNITION)
    visualizer = Visualizer(cfg.VISUALIZATION)
    
    # Enable track history visualization
    visualizer.set_track_history_visibility(True)
    
    # Initialize video source and writer
    video_source = VideoSource(input_path)
    video_writer = VideoWriter(
        output_path, 
        fps=video_source.fps, 
        resolution=(video_source.width, video_source.height)
    )
    
    # Store jersey info for each track
    jersey_info = {}
    
    # Process frames
    print("Processing video...")
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = video_source.read()
            if not ret:
                break
            
            # Process frame
            detection_results = detector.detect(frame)
            tracking_results = tracker.update(frame, detection_results)
            
            # Recognize jersey numbers
            for track in tracking_results["tracks"]:
                track_id = track["track_id"]
                
                # Skip if we already have a high-confidence recognition
                if track_id in jersey_info and jersey_info[track_id].get("confidence", 0) > 0.7:
                    continue
                
                # Recognize jersey number
                recognition_results = recognizer.recognize(frame, track)
                
                # Update jersey info if we got a number
                if recognition_results["jersey_number"]:
                    jersey_info[track_id] = {
                        "jersey_number": recognition_results["jersey_number"],
                        "confidence": recognition_results["confidence"]
                    }
            
            # Visualize results
            vis_frame = visualizer.draw_tracks(
                frame, 
                tracking_results["tracks"], 
                tracker.track_history,
                jersey_info
            )
            
            # Add FPS information
            frame_time = time.time() - start_time
            fps = frame_count / frame_time if frame_time > 0 else 0
            
            cv2.putText(
                vis_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Write frame to output video
            video_writer.write(vis_frame)
            
            # Display frame
            cv2.imshow('Example', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"\rProcessed {frame_count} frames... ({fps:.1f} FPS)", end="")
        
        # Print final statistics
        total_time = time.time() - start_time
        average_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nProcessed {frame_count} frames in {total_time:.1f} seconds ({average_fps:.1f} FPS)")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted.")
    
    finally:
        # Clean up
        video_source.release()
        video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"Output saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    run_example(args.input, args.output)


if __name__ == "__main__":
    main() 