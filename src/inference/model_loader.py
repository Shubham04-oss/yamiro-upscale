"""
Yamiro Upscaler - Model Loading and Device Management

Handles PyTorch device detection with MPS optimization for Apple Silicon,
model loading, and fallback strategies.
"""

import torch
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Suppress specific warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device and PyTorch information."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": False,
        "mps_built": False,
    }
    
    # Check MPS availability (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        info["mps_built"] = torch.backends.mps.is_built()
        info["mps_available"] = torch.backends.mps.is_available()
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    return info


def get_optimal_device(prefer: str = 'auto') -> torch.device:
    """
    Get the optimal PyTorch device for inference.
    
    Args:
        prefer: 'auto', 'mps', 'cuda', or 'cpu'
        
    Returns:
        torch.device: The best available device
    """
    device_info = get_device_info()
    
    if prefer == 'auto':
        # Auto-detect best device
        if device_info["mps_available"]:
            device = torch.device('mps')
            logger.info(f"✅ Using Apple Silicon MPS acceleration")
        elif device_info["cuda_available"]:
            device = torch.device('cuda')
            logger.info(f"✅ Using CUDA GPU: {device_info.get('cuda_device_name', 'Unknown')}")
        else:
            device = torch.device('cpu')
            logger.warning("⚠️  Using CPU - consider using a GPU for better performance")
    
    elif prefer == 'mps':
        if device_info["mps_available"]:
            device = torch.device('mps')
            logger.info("✅ Using MPS (Apple Silicon) as requested")
        else:
            device = torch.device('cpu')
            logger.warning("⚠️  MPS not available, falling back to CPU")
    
    elif prefer == 'cuda':
        if device_info["cuda_available"]:
            device = torch.device('cuda')
            logger.info(f"✅ Using CUDA as requested: {device_info.get('cuda_device_name', 'Unknown')}")
        else:
            device = torch.device('cpu')
            logger.warning("⚠️  CUDA not available, falling back to CPU")
    
    else:  # cpu or unknown
        device = torch.device('cpu')
        logger.info("ℹ️  Using CPU as requested")
    
    return device


def get_memory_info(device: torch.device) -> Dict[str, float]:
    """Get memory information for the device."""
    info = {}
    
    if device.type == 'cuda':
        info['allocated'] = torch.cuda.memory_allocated(device) / 1024**3  # GB
        info['cached'] = torch.cuda.memory_reserved(device) / 1024**3
        info['max_memory'] = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    elif device.type == 'mps':
        # MPS memory tracking (limited API)
        try:
            info['allocated'] = torch.mps.current_allocated_memory() / 1024**3
        except:
            info['allocated'] = 0.0
        info['cached'] = 0.0  # Not available for MPS
        info['max_memory'] = 0.0  # Not directly available
    
    else:  # CPU
        import psutil
        vm = psutil.virtual_memory()
        info['allocated'] = (vm.total - vm.available) / 1024**3
        info['cached'] = vm.cached / 1024**3
        info['max_memory'] = vm.total / 1024**3
    
    return info


class ModelLoader:
    """Handles loading and managing AI upscaling models."""
    
    SUPPORTED_MODELS = {
        'realesrgan_x4': {
            'scale': 4,
            'model_name': 'RealESRGAN_x4plus',
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        },
        'realesrgan_x2': {
            'scale': 2,
            'model_name': 'RealESRGAN_x2plus',
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        },
        'realesrgan_anime': {
            'scale': 4,
            'model_name': 'RealESRGAN_x4plus_anime_6B',
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        }
    }
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
    
    def download_model(self, model_key: str) -> Path:
        """Download model weights if not already present."""
        if model_key not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_key}")
        
        model_info = self.SUPPORTED_MODELS[model_key]
        model_file = self.models_dir / f"{model_info['model_name']}.pth"
        
        if model_file.exists():
            logger.info(f"✅ Model already exists: {model_file}")
            return model_file
        
        logger.info(f"📥 Downloading {model_key} model...")
        
        # Download using torch.hub or requests
        try:
            import requests
            response = requests.get(model_info['model_path'], stream=True)
            response.raise_for_status()
            
            with open(model_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded: {model_file}")
            return model_file
            
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            raise
    
    def load_realesrgan(self, model_key: str, device: torch.device, half_precision: bool = False):
        """Load Real-ESRGAN model."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError:
            logger.warning("⚠️  Real-ESRGAN not available, using demo upscaler")
            return self._create_demo_upscaler(device)
        
        if model_key in self.loaded_models:
            logger.info(f"♻️  Using cached model: {model_key}")
            return self.loaded_models[model_key]
        
        model_path = self.download_model(model_key)
        model_info = self.SUPPORTED_MODELS[model_key]
        
        # Configure model architecture
        if 'anime' in model_key:
            netscale = 4
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=netscale)
        else:
            netscale = model_info['scale']
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
        
        # Initialize upsampler
        dni_weight = None  # No need for face enhancement here
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            dni_weight=dni_weight,
            model=model,
            tile=0,  # Disable tiling for now
            tile_pad=10,
            pre_pad=0,
            half=half_precision and device.type != 'cpu',
            device=device
        )
        
        self.loaded_models[model_key] = upsampler
        logger.info(f"✅ Loaded {model_key} on {device}")
        
        return upsampler
    
    def _create_demo_upscaler(self, device: torch.device):
        """Create a demo upscaler using simple interpolation."""
        class DemoUpscaler:
            def __init__(self, device, scale_factor=4):
                self.device = device
                self.scale_factor = scale_factor
                self.tile = 512
                self.tile_pad = 10
                self.pre_pad = 0
                
            def enhance(self, img, outscale=None):
                """Simple bicubic upscaling for demo purposes."""
                import cv2
                
                height, width = img.shape[:2]
                new_height = height * self.scale_factor
                new_width = width * self.scale_factor
                
                # Use bicubic interpolation
                upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                logger.info(f"📐 Demo upscaled: {width}x{height} → {new_width}x{new_height}")
                return upscaled, None
        
        logger.info("🎭 Created demo upscaler (bicubic interpolation)")
        return DemoUpscaler(device)
    
    def clear_cache(self):
        """Clear loaded models from memory."""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("🧹 Cleared model cache")


def test_device_setup():
    """Test device setup and model loading."""
    logger.info("🧪 Testing Yamiro Upscaler device setup...")
    
    # Test device detection
    device_info = get_device_info()
    logger.info(f"Device Info: {device_info}")
    
    device = get_optimal_device('auto')
    logger.info(f"Selected device: {device}")
    
    # Test memory info
    memory_info = get_memory_info(device)
    logger.info(f"Memory Info: {memory_info}")
    
    # Test model loader
    loader = ModelLoader()
    logger.info(f"Models directory: {loader.models_dir}")
    
    logger.info("✅ Device setup test completed!")
    return device, loader


if __name__ == "__main__":
    test_device_setup()