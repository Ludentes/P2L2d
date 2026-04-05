class ComfyUIError(Exception):
    """Base exception for all ComfyUI client errors."""


class ComfyUIConnectionError(ComfyUIError):
    """Cannot reach the ComfyUI server."""


class ComfyUIJobError(ComfyUIError):
    """ComfyUI reported an error for this job or workflow."""


class ComfyUITimeoutError(ComfyUIError):
    """Polling timed out before the job completed."""
