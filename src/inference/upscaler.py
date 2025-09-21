"""
Yamiro Upscaler - Core Upscaling Engine

High-level interface for AI image upscaling with batch processing,
memory management, and performance optimization.
"""

import torch
import numpy as np
from PIL import Image
import cv2
import logging
import time
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import gc

from .model_loader import ModelLoader, get_optimal_device, get_memory_info

logger = logging.getLogger(__name__)


class UpscalerConfig:
    """Configuration for the upscaler."""
    
    def __init__(
        self,
        model_name: str = 'realesrgan_x4',
        device: str = 'auto',
        half_precision: bool = False,
        tile_size: int = 512,
        tile_pad: int = 10,
        pre_pad: int = 0,
        batch_size: int = 1,
        output_format: str = 'PNG',
        output_quality: int = 95
    ):
        self.model_name = model_name
        self.device = device
        self.half_precision = half_precision
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.batch_size = batch_size
        self.output_format = output_format.upper()
        self.output_quality = output_quality


class YamiroUpscaler:
    """
    Main upscaler class with MPS optimization and batch processing.
    """
    
    def __init__(self, config: Optional[UpscalerConfig] = None):
        self.config = config or UpscalerConfig()
        self.device = get_optimal_device(self.config.device)
        self.model_loader = ModelLoader()
        self.upsampler = None
        self.stats = {
            'images_processed': 0,
            'total_time': 0.0,
            'total_pixels_in': 0,
            'total_pixels_out': 0
        }
        
        logger.info(f"🚀 Yamiro Upscaler initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🎯 Model: {self.config.model_name}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the AI model."""
        try:
            self.upsampler = self.model_loader.load_realesrgan(
                self.config.model_name,
                self.device,
                self.config.half_precision
            )
            
            # Configure tiling for memory efficiency
            if self.config.tile_size > 0:
                self.upsampler.tile = self.config.tile_size
                self.upsampler.tile_pad = self.config.tile_pad
                self.upsampler.pre_pad = self.config.pre_pad
                logger.info(f"🔧 Tiling enabled: {self.config.tile_size}px")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """Preprocess input image to the format expected by Real-ESRGAN."""
        if isinstance(image, str):
            # Load from file path
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        elif isinstance(image, Image.Image):
            # Convert PIL to OpenCV format
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        elif isinstance(image, np.ndarray):
            # Assume it's already in the right format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB to BGR if needed
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return image
    
    def postprocess_image(self, image: np.ndarray) -> Image.Image:
        """Convert output back to PIL Image."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
    
    def upscale_single(
        self, 
        image: Union[Image.Image, np.ndarray, str],
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Upscale a single image.
        
        Args:
            image: Input image (PIL, numpy array, or file path)
            save_path: Optional path to save the result
            
        Returns:
            PIL Image: Upscaled image
        """
        start_time = time.time()
        
        # Preprocess
        input_array = self.preprocess_image(image)
        input_height, input_width = input_array.shape[:2]
        
        logger.info(f"🔄 Upscaling {input_width}x{input_height} image...")
        
        # Memory check
        memory_before = get_memory_info(self.device)
        
        try:
            # Run inference
            with torch.no_grad():
                output_array, _ = self.upsampler.enhance(input_array, outscale=None)
            
            # Postprocess
            result_image = self.postprocess_image(output_array)
            
            # Update stats
            processing_time = time.time() - start_time
            output_height, output_width = output_array.shape[:2]
            
            self.stats['images_processed'] += 1
            self.stats['total_time'] += processing_time
            self.stats['total_pixels_in'] += input_width * input_height
            self.stats['total_pixels_out'] += output_width * output_height
            
            # Memory check
            memory_after = get_memory_info(self.device)
            
            logger.info(f"✅ Upscaled to {output_width}x{output_height} in {processing_time:.2f}s")
            logger.info(f"📊 Memory: {memory_before.get('allocated', 0):.1f}GB → {memory_after.get('allocated', 0):.1f}GB")
            
            # Save if requested
            if save_path:
                self.save_image(result_image, save_path)
            
            return result_image
            
        except Exception as e:
            logger.error(f"❌ Upscaling failed: {e}")
            raise
        
        finally:
            # Clean up GPU memory
            self._cleanup_memory()
    
    def upscale_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        output_dir: Optional[str] = None,
        filename_prefix: str = "upscaled_"
    ) -> List[Image.Image]:
        """
        Upscale multiple images with batch processing.
        
        Args:
            images: List of images to upscale
            output_dir: Directory to save results
            filename_prefix: Prefix for output filenames
            
        Returns:
            List[Image.Image]: Upscaled images
        """
        logger.info(f"🔄 Batch upscaling {len(images)} images...")
        
        results = []
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image in enumerate(images):
            try:
                # Determine save path
                save_path = None
                if output_path:
                    if isinstance(image, str):
                        # Use original filename
                        original_name = Path(image).stem
                        save_path = output_path / f"{filename_prefix}{original_name}.{self.config.output_format.lower()}"
                    else:
                        save_path = output_path / f"{filename_prefix}{i:04d}.{self.config.output_format.lower()}"
                
                # Upscale
                result = self.upscale_single(image, str(save_path) if save_path else None)
                results.append(result)
                
                logger.info(f"📊 Progress: {i+1}/{len(images)} complete")
                
            except Exception as e:
                logger.error(f"❌ Failed to process image {i}: {e}")
                continue
        
        logger.info(f"✅ Batch processing complete: {len(results)}/{len(images)} successful")
        return results
    
    def upscale_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Upscale all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_extensions: Allowed file extensions
            recursive: Search subdirectories
            
        Returns:
            Dict with processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"{pattern}{ext}"))
            image_files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
        
        logger.info(f"📂 Found {len(image_files)} images in {input_dir}")
        
        if not image_files:
            logger.warning("⚠️  No images found!")
            return {"processed": 0, "failed": 0, "time": 0}
        
        # Process images
        start_time = time.time()
        processed = 0
        failed = 0
        
        for image_file in image_files:
            try:
                # Maintain directory structure if recursive
                if recursive:
                    rel_path = image_file.relative_to(input_path)
                    output_file = output_path / rel_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_file = output_path / image_file.name
                
                # Change extension to output format
                output_file = output_file.with_suffix(f".{self.config.output_format.lower()}")
                
                # Skip if already exists
                if output_file.exists():
                    logger.info(f"⏭️  Skipping existing: {output_file}")
                    continue
                
                self.upscale_single(str(image_file), str(output_file))
                processed += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process {image_file}: {e}")
                failed += 1
        
        total_time = time.time() - start_time
        
        stats = {
            "processed": processed,
            "failed": failed,
            "total_files": len(image_files),
            "time": total_time,
            "avg_time_per_image": total_time / max(processed, 1)
        }
        
        logger.info(f"📊 Directory processing complete: {stats}")
        return stats
    
    def save_image(self, image: Image.Image, path: str):
        """Save image with proper quality settings."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_kwargs = {}
        if self.config.output_format in ['JPEG', 'WEBP']:
            save_kwargs['quality'] = self.config.output_quality
            save_kwargs['optimize'] = True
        
        image.save(str(save_path), format=self.config.output_format, **save_kwargs)
        logger.info(f"💾 Saved: {save_path}")
    
    def _cleanup_memory(self):
        """Clean up GPU/MPS memory."""
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if self.stats['images_processed'] > 0:
            avg_time = self.stats['total_time'] / self.stats['images_processed']
            avg_pixels_in = self.stats['total_pixels_in'] / self.stats['images_processed']
            avg_pixels_out = self.stats['total_pixels_out'] / self.stats['images_processed']
            throughput = self.stats['images_processed'] / max(self.stats['total_time'], 0.001)
        else:
            avg_time = 0
            avg_pixels_in = 0
            avg_pixels_out = 0
            throughput = 0
        
        return {
            **self.stats,
            'avg_time_per_image': avg_time,
            'avg_input_pixels': avg_pixels_in,
            'avg_output_pixels': avg_pixels_out,
            'throughput_ips': throughput,
            'device': str(self.device),
            'model': self.config.model_name
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'images_processed': 0,
            'total_time': 0.0,
            'total_pixels_in': 0,
            'total_pixels_out': 0
        }
        logger.info("📊 Statistics reset")


def create_upscaler(
    model_name: str = 'realesrgan_x4',
    device: str = 'auto',
    **kwargs
) -> YamiroUpscaler:
    """Convenience function to create an upscaler with default config."""
    config = UpscalerConfig(model_name=model_name, device=device, **kwargs)
    return YamiroUpscaler(config)


# Convenience functions for quick usage
def upscale_image(
    image: Union[Image.Image, np.ndarray, str],
    model_name: str = 'realesrgan_x4',
    device: str = 'auto',
    save_path: Optional[str] = None
) -> Image.Image:
    """Quick function to upscale a single image."""
    upscaler = create_upscaler(model_name, device)
    return upscaler.upscale_single(image, save_path)


def upscale_directory(
    input_dir: str,
    output_dir: str,
    model_name: str = 'realesrgan_x4',
    device: str = 'auto',
    **kwargs
) -> Dict[str, Any]:
    """Quick function to upscale all images in a directory."""
    upscaler = create_upscaler(model_name, device, **kwargs)
    return upscaler.upscale_directory(input_dir, output_dir)


if __name__ == "__main__":
    # Test the upscaler
    logger.info("🧪 Testing Yamiro Upscaler...")
    
    try:
        upscaler = create_upscaler()
        stats = upscaler.get_stats()
        logger.info(f"📊 Initial stats: {stats}")
        logger.info("✅ Upscaler test completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise