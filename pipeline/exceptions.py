"""Custom exception types for the texture generation pipeline."""


class PipelineError(Exception):
    """Base exception for all texture generation pipeline errors."""


class MediaPipeLandmarkError(PipelineError):
    """MediaPipe failed to detect face landmarks in the portrait."""
