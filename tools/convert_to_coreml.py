#!/usr/bin/env python3
"""
Yamiro Upscaler - CoreML Conversion Tool

Converts PyTorch models to CoreML for native macOS performance.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import coremltools as ct
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install torch coremltools pillow")
    sys.exit(1)

from inference.upscaler import YamiroUpscaler, UpscalerConfig
from inference.model_loader import get_optimal_device

logger = logging.getLogger(__name__)


class CoreMLConverter:
    """Convert Yamiro Upscaler models to CoreML."""
    
    def __init__(self, model_name: str = 'realesrgan_x4'):
        self.model_name = model_name
        self.upscaler = None
        
    def load_pytorch_model(self):
        """Load the PyTorch model."""
        logger.info(f"🔄 Loading PyTorch model: {self.model_name}")
        
        config = UpscalerConfig(
            model_name=self.model_name,
            device='cpu',  # Use CPU for conversion
            half_precision=False
        )
        
        self.upscaler = YamiroUpscaler(config)
        logger.info("✅ PyTorch model loaded")
    
    def create_example_input(self, size: tuple = (512, 512)) -> torch.Tensor:
        """Create example input for tracing."""
        # Create a sample RGB image tensor
        # Shape: (1, 3, H, W) - batch, channels, height, width
        example_input = torch.randn(1, 3, size[1], size[0])
        logger.info(f"📐 Created example input: {example_input.shape}")
        return example_input
    
    def convert_to_coreml(
        self,
        output_path: str,
        input_size: tuple = (512, 512),
        compute_units: str = 'all'
    ):
        """
        Convert the model to CoreML format.
        
        Args:
            output_path: Path to save the CoreML model
            input_size: Input image size (width, height)
            compute_units: 'all', 'cpu_only', or 'cpu_and_neural_engine'
        """
        if self.upscaler is None:
            self.load_pytorch_model()
        
        logger.info(f"🔄 Converting to CoreML...")
        
        try:
            # Get the underlying PyTorch model
            pytorch_model = self.upscaler.upsampler.model
            pytorch_model.eval()
            
            # Create example input
            example_input = self.create_example_input(input_size)
            
            # Trace the model
            logger.info("📊 Tracing PyTorch model...")
            traced_model = torch.jit.trace(pytorch_model, example_input)
            
            # Convert to CoreML
            logger.info("⚙️ Converting to CoreML...")
            
            # Set compute units
            compute_unit_map = {
                'all': ct.ComputeUnit.ALL,
                'cpu_only': ct.ComputeUnit.CPU_ONLY,
                'cpu_and_neural_engine': ct.ComputeUnit.CPU_AND_NE
            }
            
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(
                    name="input",
                    shape=example_input.shape,
                    dtype=np.float32
                )],
                outputs=[ct.TensorType(name="output")],
                compute_units=compute_unit_map.get(compute_units, ct.ComputeUnit.ALL),
                minimum_deployment_target=ct.target.macOS13  # macOS 13+ for better performance
            )
            
            # Add metadata
            coreml_model.short_description = f"Yamiro Upscaler - {self.model_name}"
            coreml_model.author = "Yamiro Team"
            coreml_model.license = "MIT"
            coreml_model.version = "1.0.0"
            
            # Set input/output descriptions
            coreml_model.input_description["input"] = f"Input image tensor ({input_size[0]}x{input_size[1]})"
            coreml_model.output_description["output"] = "Upscaled image tensor"
            
            # Save the model
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            coreml_model.save(str(output_path))
            
            logger.info(f"✅ CoreML model saved: {output_path}")
            
            # Print model info
            model_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"📊 Model size: {model_size:.1f} MB")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            raise
    
    def test_coreml_model(self, coreml_path: str, test_image_path: str = None):
        """Test the converted CoreML model."""
        logger.info(f"🧪 Testing CoreML model: {coreml_path}")
        
        try:
            # Load CoreML model
            coreml_model = ct.models.MLModel(coreml_path)
            
            # Create or load test image
            if test_image_path and Path(test_image_path).exists():
                test_image = Image.open(test_image_path)
                test_image = test_image.convert('RGB')
                test_image = test_image.resize((512, 512))
            else:
                # Create synthetic test image
                test_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
                test_image = Image.fromarray(test_array)
            
            # Preprocess for CoreML
            test_array = np.array(test_image).astype(np.float32) / 255.0
            test_tensor = np.transpose(test_array, (2, 0, 1))  # HWC to CHW
            test_input = np.expand_dims(test_tensor, axis=0)  # Add batch dimension
            
            logger.info(f"📐 Test input shape: {test_input.shape}")
            
            # Run inference
            import time
            start_time = time.time()
            
            prediction = coreml_model.predict({"input": test_input})
            
            inference_time = time.time() - start_time
            
            # Get output
            output_tensor = prediction["output"]
            logger.info(f"📐 Output shape: {output_tensor.shape}")
            logger.info(f"⏱️ Inference time: {inference_time:.3f}s")
            
            # Convert back to image
            output_array = np.squeeze(output_tensor, axis=0)  # Remove batch
            output_array = np.transpose(output_array, (1, 2, 0))  # CHW to HWC
            output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
            
            output_image = Image.fromarray(output_array)
            
            # Save test result
            test_output_path = Path(coreml_path).parent / "coreml_test_output.png"
            output_image.save(test_output_path)
            
            logger.info(f"✅ CoreML test successful!")
            logger.info(f"💾 Test output saved: {test_output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ CoreML test failed: {e}")
            return False


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description="Convert Yamiro Upscaler to CoreML")
    parser.add_argument('--model', default='realesrgan_x4',
                       choices=['realesrgan_x4', 'realesrgan_x2', 'realesrgan_anime'],
                       help='Model to convert')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CoreML model path (.mlmodel)')
    parser.add_argument('--input-size', nargs=2, type=int, default=[512, 512],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Input image size for conversion')
    parser.add_argument('--compute-units', default='all',
                       choices=['all', 'cpu_only', 'cpu_and_neural_engine'],
                       help='CoreML compute units')
    parser.add_argument('--test', action='store_true',
                       help='Test the converted model')
    parser.add_argument('--test-image', 
                       help='Test image path (optional)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create converter
        converter = CoreMLConverter(args.model)
        
        # Convert model
        coreml_path = converter.convert_to_coreml(
            args.output,
            tuple(args.input_size),
            args.compute_units
        )
        
        # Test if requested
        if args.test:
            success = converter.test_coreml_model(coreml_path, args.test_image)
            if not success:
                sys.exit(1)
        
        logger.info("🎉 Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()