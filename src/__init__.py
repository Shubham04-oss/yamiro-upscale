"""
Yamiro Upscaler - AI-Powered Anime Image Upscaling

Optimized for Apple Silicon (MPS) with Real-ESRGAN and SwinIR support.
"""

__version__ = "1.0.0"
__author__ = "Yamiro Team"
__email__ = "contact@yamiro.dev"
__description__ = "AI-powered anime image upscaling optimized for Apple Silicon"

from .inference.upscaler import YamiroUpscaler, UpscalerConfig, create_upscaler
from .inference.model_loader import get_device_info, get_optimal_device
from .inference.video import VideoUpscaler, upscale_video

__all__ = [
    'YamiroUpscaler',
    'UpscalerConfig', 
    'create_upscaler',
    'get_device_info',
    'get_optimal_device',
    'VideoUpscaler',
    'upscale_video'
]