"""
Centralized constants for the 3D Gait Tracking Pipeline.
Includes physical constraints, algorithmic thresholds, and default values.
"""

# Signal Processing Constants
MIN_SAMPLES = 200
HUMAN_RANGE_FREQ = (0.5, 5.0)  # Hz
HUMAN_MAX_DEGREE = 180         # Max peak-to-peak ROM for gait
MAX_ACC = 200                  # Max acceleration peak (m/s^2 or similar unit)
MAX_ANGULAR_JUMP = 30.0        # Max degrees per frame at ~83Hz

# Vision & Keypoints Constants
DEFAULT_TRIANGLE_SIZE = 15     # Pixels for fallback triangle
HUBER_MAX_ANGLE_DEG = 20.0     # Max deviation for thigh edge fitting
SOBEL_KSIZE = 3                # Sobel operator kernel size
GAUSSIAN_BLUR_KSIZE = (7, 7)   # Kernel size for pre-processing frames
GRADIENT_THRESHOLD = 15        # Minimum gradient magnitude for edge detection
NUM_THIGH_SLICES = 15          # Number of slices for edge detection
THIGH_WIDTH_FACTOR = 0.4       # Max search width relative to femur length
THIGH_LENGTH_FACTOR = 0.35     # Search length relative to femur length
SAFETY_MARGIN_PX = 2.0         # Pixels to shrink the final thigh mask
KNEE_OFFSET_FACTOR = 0.1       # Offset from knee joint towards hip (0.05 = 5% of femur)

# Dataset & External Model Asset Constants
DEFAULT_ARTICULATION = "leg"
SENSORS_COLUMNS = ['Timestamp', 'accX', 'accY', 'accZ', 'Roll', 'Pitch', 'Yaw']
MEDIAPIPE_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
UNIDEPTH_V2_MODEL_ID = "lpiccinelli/unidepth-v2-vitl14"
