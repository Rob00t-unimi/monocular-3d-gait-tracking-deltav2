"""
Custom exceptions for the 3D Gait Tracking Pipeline.
Ensures specific error handling across the signal and vision processing stages.
"""

class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""
    pass

class SignalError(PipelineError):
    """Base exception for signal processing errors."""
    pass

class SignalLoadingError(SignalError):
    """Raised when an IMU CSV file cannot be loaded or is malformed."""
    pass

class SignalValidationError(SignalError):
    """Raised when a signal fails physiological or physical constraints."""
    pass

class VisionError(PipelineError):
    """Base exception for computer vision and keypoint errors."""
    pass

class FrameNotFoundError(VisionError):
    """Raised when a requested video frame cannot be retrieved."""
    pass

class KeypointNotFoundError(VisionError):
    """Raised when required keypoints (e.g. RHip, RKnee) are missing."""
    pass
