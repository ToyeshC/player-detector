"""
Player Detection and Jersey Number Recognition System.
"""

from player_detector.detection.detector import PlayerDetector
from player_detector.tracking.tracker import PlayerTracker
from player_detector.recognition.jersey_recognizer import JerseyRecognizer
from player_detector.visualization.visualizer import Visualizer
from player_detector.utils.video_utils import VideoSource, VideoWriter

__version__ = "0.1.0"
__all__ = [
    "PlayerDetector",
    "PlayerTracker",
    "JerseyRecognizer",
    "Visualizer",
    "VideoSource",
    "VideoWriter"
] 