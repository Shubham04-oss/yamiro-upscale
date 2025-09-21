"""
Yamiro Upscaler - Gradio Web Interface

Interactive web demo for AI image upscaling with real-time preview,
batch processing, and performance monitoring.
"""

import gradio as gr
import numpy as np
from PIL import Image
import logging
import time
import threading
import json
from pathlib import Path
from typing import Optional, List, Tuple, Any
import tempfile

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.upscaler import YamiroUpscaler, UpscalerConfig
from inference.model_loader import get_device_info, get_optimal_device
from utils.profiler import SystemProfiler

logger = logging.getLogger(__name__)

# Global upscaler instance (lazy loaded)
_upscaler = None
_profiler = None


def get_upscaler(model_name: str = 'realesrgan_x4', device: str = 'auto') -> YamiroUpscaler:
    """Get or create upscaler instance."""
    global _upscaler
    
    if _upscaler is None or _upscaler.config.model_name != model_name:
        logger.info(f"🔄 Loading model: {model_name} on {device}")
        config = UpscalerConfig(model_name=model_name, device=device, half_precision=True)
        _upscaler = YamiroUpscaler(config)
    
    return _upscaler


def upscale_single_image(
    image: Image.Image,
    model_name: str,
    device: str,
    scale_factor: int,
    tile_size: int
) -> Tuple[Optional[Image.Image], str, str]:
    """
    Upscale a single image with progress tracking.
    
    Returns:
        Tuple of (upscaled_image, info_text, stats_json)
    """
    if image is None:
        return None, "❌ Please upload an image first", "{}"
    
    try:
        # Get upscaler
        upscaler = get_upscaler(model_name, device)
        
        if hasattr(upscaler, 'upsampler') and upscaler.upsampler and hasattr(upscaler.upsampler, 'tile'):
            if upscaler.upsampler.tile != tile_size:
                upscaler.upsampler.tile = tile_size
                logger.info(f"🔧 Updated tile size: {tile_size}")
        
        # Upscale image
        start_time = time.time()
        result_image = upscaler.upscale_single(image)
        processing_time = time.time() - start_time
        
        # Generate info
        original_size = image.size
        upscaled_size = result_image.size
        actual_scale = upscaled_size[0] / original_size[0]
        
        info_text = f"""
        ✅ **Upscaling Complete!**
        
        📊 **Details:**
        - Original: {original_size[0]} × {original_size[1]} ({original_size[0] * original_size[1]:,} pixels)
        - Upscaled: {upscaled_size[0]} × {upscaled_size[1]} ({upscaled_size[0] * upscaled_size[1]:,} pixels)
        - Scale Factor: {actual_scale:.1f}x
        - Processing Time: {processing_time:.2f}s
        - Device: {upscaler.device}
        - Model: {model_name}
        """
        
        # Generate stats JSON
        stats = {
            'processing_time': processing_time,
            'original_size': original_size,
            'upscaled_size': upscaled_size,
            'scale_factor': actual_scale,
            'model': model_name,
            'device': str(upscaler.device),
            'pixels_per_second': (original_size[0] * original_size[1]) / processing_time
        }
        
        return result_image, info_text, json.dumps(stats, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Upscaling failed: {e}")
        error_msg = f"❌ **Error:** {str(e)}"
        return None, error_msg, json.dumps({'error': str(e)})


def upscale_batch_images(
    files: List[Any],
    model_name: str,
    device: str,
    scale_factor: int,
    tile_size: int
) -> Tuple[List[Image.Image], str, str]:
    """
    Upscale multiple images in batch.
    
    Returns:
        Tuple of (upscaled_images, info_text, stats_json)
    """
    if not files:
        return [], "❌ Please upload some images first", "{}"
    
    try:
        upscaler = get_upscaler(model_name, device)
        
        if upscaler.upsampler.tile != tile_size:
            upscaler.upsampler.tile = tile_size
        
        results = []
        all_stats = []
        
        total_files = len(files)
        start_time = time.time()
        
        for i, file in enumerate(files):
            try:
                # Load image
                image = Image.open(file.name)
                
                # Upscale
                img_start = time.time()
                result = upscaler.upscale_single(image)
                img_time = time.time() - img_start
                
                results.append(result)
                
                # Track stats
                all_stats.append({
                    'filename': Path(file.name).name,
                    'processing_time': img_time,
                    'original_size': image.size,
                    'upscaled_size': result.size
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Generate summary
        if results:
            total_original_pixels = sum(s['original_size'][0] * s['original_size'][1] for s in all_stats)
            avg_time = sum(s['processing_time'] for s in all_stats) / len(all_stats)
            
            info_text = f"""
            ✅ **Batch Processing Complete!**
            
            📊 **Summary:**
            - Images Processed: {len(results)}/{total_files}
            - Total Time: {total_time:.1f}s
            - Average Time/Image: {avg_time:.2f}s
            - Total Pixels Processed: {total_original_pixels:,}
            - Throughput: {total_original_pixels / total_time:.0f} pixels/sec
            """
            
            batch_stats = {
                'total_images': len(results),
                'successful': len(results),
                'failed': total_files - len(results),
                'total_time': total_time,
                'average_time_per_image': avg_time,
                'individual_stats': all_stats
            }
        else:
            info_text = "❌ No images were successfully processed"
            batch_stats = {'error': 'No successful processing'}
        
        return results, info_text, json.dumps(batch_stats, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        return [], f"❌ **Error:** {str(e)}", json.dumps({'error': str(e)})


def get_system_info() -> str:
    """Get formatted system information."""
    device_info = get_device_info()
    
    info_md = f"""
    # 🖥️ System Information
    
    **Platform:** {device_info.get('platform', 'Unknown')}  
    **Machine:** {device_info.get('machine', 'Unknown')}  
    **Python:** {device_info.get('python_version', 'Unknown')}  
    **PyTorch:** {device_info.get('torch_version', 'Unknown')}  
    
    ## GPU Support
    
    **MPS (Apple Silicon):** {'✅ Available' if device_info.get('mps_available') else '❌ Not Available'}  
    **CUDA:** {'✅ Available' if device_info.get('cuda_available') else '❌ Not Available'}  
    
    """
    
    if device_info.get('cuda_available'):
        info_md += f"**CUDA Device:** {device_info.get('cuda_device_name', 'Unknown')}  \n"
    
    return info_md


def benchmark_model(
    model_name: str,
    device: str,
    duration: int
) -> Tuple[str, str]:
    """
    Run a quick benchmark of the selected model.
    
    Returns:
        Tuple of (results_text, stats_json)
    """
    try:
        upscaler = get_upscaler(model_name, device)
        
        # Create test image
        test_image = Image.fromarray(
            np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        )
        
        # Warm up
        upscaler.upscale_single(test_image)
        
        # Benchmark
        times = []
        iterations = max(3, duration // 5)  # At least 3 iterations
        
        for i in range(iterations):
            start_time = time.time()
            result = upscaler.upscale_single(test_image)
            processing_time = time.time() - start_time
            times.append(processing_time)
        
        # Calculate stats
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = (512 * 512) / avg_time  # pixels per second
        
        results_text = f"""
        # 🏃‍♂️ Benchmark Results
        
        **Model:** {model_name}  
        **Device:** {device}  
        **Test Image:** 512×512 pixels  
        **Iterations:** {iterations}  
        
        ## Performance Metrics
        
        **Average Time:** {avg_time:.3f}s  
        **Min Time:** {min_time:.3f}s  
        **Max Time:** {max_time:.3f}s  
        **Throughput:** {throughput:,.0f} pixels/sec  
        **Images/sec:** {1/avg_time:.2f}  
        """
        
        benchmark_stats = {
            'model': model_name,
            'device': device,
            'iterations': iterations,
            'times': times,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'throughput_pixels_per_sec': throughput,
            'images_per_sec': 1/avg_time
        }
        
        return results_text, json.dumps(benchmark_stats, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        error_text = f"❌ **Benchmark Error:** {str(e)}"
        return error_text, json.dumps({'error': str(e)})


def create_yamiro_interface():
    """Create the main Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .yamiro-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .yamiro-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    
    .yamiro-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    .performance-info {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        title="🎌 Yamiro Upscaler",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="yamiro-header">
            <h1>🎌 Yamiro Upscaler</h1>
            <p>AI-Powered Anime Image Upscaling • Optimized for Apple Silicon</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # Single Image Tab
            with gr.TabItem("🖼️ Single Image"):
                with gr.Row():
                    with gr.Column(scale=2):
                        input_image = gr.Image(
                            type="pil",
                            label="📤 Upload Image",
                            height=400
                        )
                        
                        with gr.Row():
                            model_select = gr.Dropdown(
                                choices=['realesrgan_x4', 'realesrgan_x2', 'realesrgan_anime'],
                                value='realesrgan_x4',
                                label="🤖 AI Model"
                            )
                            device_select = gr.Dropdown(
                                choices=['auto', 'mps', 'cuda', 'cpu'],
                                value='auto',
                                label="📱 Device"
                            )
                        
                        with gr.Row():
                            scale_factor = gr.Slider(
                                minimum=2,
                                maximum=4,
                                step=1,
                                value=4,
                                label="🔍 Scale Factor"
                            )
                            tile_size = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                step=128,
                                value=512,
                                label="🧩 Tile Size (Memory)"
                            )
                        
                        upscale_btn = gr.Button(
                            "🚀 Upscale Image",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        output_image = gr.Image(
                            label="✨ Upscaled Result",
                            height=400
                        )
                        
                        info_text = gr.Markdown(
                            value="Upload an image and click 'Upscale Image' to begin.",
                            elem_classes=["performance-info"]
                        )
                
                with gr.Row():
                    stats_json = gr.Code(
                        label="📊 Processing Statistics",
                        language="json",
                        interactive=False
                    )
            
            # Batch Processing Tab
            with gr.TabItem("📁 Batch Processing"):
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="📤 Upload Multiple Images"
                        )
                        
                        with gr.Row():
                            batch_model = gr.Dropdown(
                                choices=['realesrgan_x4', 'realesrgan_x2', 'realesrgan_anime'],
                                value='realesrgan_x4',
                                label="🤖 AI Model"
                            )
                            batch_device = gr.Dropdown(
                                choices=['auto', 'mps', 'cuda', 'cpu'],
                                value='auto',
                                label="📱 Device"
                            )
                        
                        with gr.Row():
                            batch_scale = gr.Slider(
                                minimum=2,
                                maximum=4,
                                step=1,
                                value=4,
                                label="🔍 Scale Factor"
                            )
                            batch_tile_size = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                step=128,
                                value=512,
                                label="🧩 Tile Size"
                            )
                        
                        batch_btn = gr.Button(
                            "🚀 Process Batch",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        batch_output = gr.Gallery(
                            label="✨ Batch Results",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                
                batch_info = gr.Markdown(
                    value="Upload multiple images and click 'Process Batch' to begin.",
                    elem_classes=["performance-info"]
                )
                
                batch_stats = gr.Code(
                    label="📊 Batch Statistics",
                    language="json",
                    interactive=False
                )
            
            # Benchmark Tab
            with gr.TabItem("🏃‍♂️ Benchmark"):
                with gr.Row():
                    with gr.Column():
                        bench_model = gr.Dropdown(
                            choices=['realesrgan_x4', 'realesrgan_x2', 'realesrgan_anime'],
                            value='realesrgan_x4',
                            label="🤖 Model to Benchmark"
                        )
                        bench_device = gr.Dropdown(
                            choices=['auto', 'mps', 'cuda', 'cpu'],
                            value='auto',
                            label="📱 Device"
                        )
                        bench_duration = gr.Slider(
                            minimum=10,
                            maximum=60,
                            step=5,
                            value=15,
                            label="⏱️ Duration (seconds)"
                        )
                        
                        benchmark_btn = gr.Button(
                            "🏃‍♂️ Run Benchmark",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        benchmark_results = gr.Markdown(
                            value="Click 'Run Benchmark' to test performance.",
                            elem_classes=["performance-info"]
                        )
                        
                        benchmark_stats_json = gr.Code(
                            label="📊 Detailed Results",
                            language="json",
                            interactive=False
                        )
            
            # System Info Tab
            with gr.TabItem("🖥️ System Info"):
                system_info_md = gr.Markdown(
                    value=get_system_info(),
                    elem_classes=["performance-info"]
                )
                
                refresh_info_btn = gr.Button("🔄 Refresh System Info")
        
        # Event handlers
        upscale_btn.click(
            fn=upscale_single_image,
            inputs=[input_image, model_select, device_select, scale_factor, tile_size],
            outputs=[output_image, info_text, stats_json],
            show_progress=True
        )
        
        batch_btn.click(
            fn=upscale_batch_images,
            inputs=[batch_files, batch_model, batch_device, batch_scale, batch_tile_size],
            outputs=[batch_output, batch_info, batch_stats],
            show_progress=True
        )
        
        benchmark_btn.click(
            fn=benchmark_model,
            inputs=[bench_model, bench_device, bench_duration],
            outputs=[benchmark_results, benchmark_stats_json],
            show_progress=True
        )
        
        refresh_info_btn.click(
            fn=get_system_info,
            outputs=[system_info_md]
        )
    
    return interface


def launch_yamiro_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch the Yamiro Upscaler web interface."""
    
    # Setup logging
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger.info("🚀 Starting Yamiro Upscaler Web Interface...")
    
    # Create interface
    interface = create_yamiro_interface()
    
    # Launch
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        quiet=not debug,
        favicon_path=None  # Could add custom favicon
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Yamiro Upscaler Web Interface")
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    launch_yamiro_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )