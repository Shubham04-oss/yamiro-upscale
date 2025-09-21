#!/usr/bin/env python3
"""
Yamiro Upscaler - Command Line Interface

Easy-to-use CLI for AI image upscaling with Real-ESRGAN.
Supports single images, batches, directories, and videos.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.upscaler import YamiroUpscaler, UpscalerConfig, create_upscaler
from inference.model_loader import get_device_info, test_device_setup

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('yamiro_upscaler.log')
        ]
    )


def print_banner():
    """Print the Yamiro Upscaler banner."""
    banner = Text.assemble(
        ("🎌 YAMIRO UPSCALER 🎌", "bold magenta"),
        "\n",
        ("AI-Powered Anime Image Upscaling", "dim"),
        "\n",
        ("Optimized for Apple Silicon (MPS)", "dim cyan")
    )
    console.print(Panel(banner, style="bold blue"))


def print_device_info():
    """Print device and system information."""
    device_info = get_device_info()
    
    table = Table(title="🖥️  System Information", style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="green")
    
    table.add_row("Platform", device_info.get('platform', 'Unknown'))
    table.add_row("Machine", device_info.get('machine', 'Unknown'))
    table.add_row("Python Version", device_info.get('python_version', 'Unknown'))
    table.add_row("PyTorch Version", device_info.get('torch_version', 'Unknown'))
    
    if device_info.get('mps_available'):
        table.add_row("MPS Support", "✅ Available", style="green")
    elif device_info.get('mps_built'):
        table.add_row("MPS Support", "⚠️  Built but not available", style="yellow")
    else:
        table.add_row("MPS Support", "❌ Not available", style="red")
    
    if device_info.get('cuda_available'):
        table.add_row("CUDA Support", "✅ Available", style="green")
        table.add_row("CUDA Device", device_info.get('cuda_device_name', 'Unknown'))
    else:
        table.add_row("CUDA Support", "❌ Not available", style="red")
    
    console.print(table)


def cmd_info(args):
    """Show system and device information."""
    print_device_info()
    
    try:
        device, loader = test_device_setup()
        console.print(f"\n✅ Device test passed: {device}", style="green")
        console.print(f"📁 Models directory: {loader.models_dir}", style="blue")
    except Exception as e:
        console.print(f"\n❌ Device test failed: {e}", style="red")
        return 1
    
    return 0


def cmd_upscale(args):
    """Main upscaling command."""
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"❌ Input path does not exist: {args.input}", style="red")
        return 1
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create upscaler config
    config = UpscalerConfig(
        model_name=args.model,
        device=args.device,
        half_precision=args.half_precision,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        output_format=args.format,
        output_quality=args.quality
    )
    
    try:
        # Initialize upscaler
        with console.status("🔄 Initializing AI model..."):
            upscaler = YamiroUpscaler(config)
        
        console.print("✅ Model loaded successfully!", style="green")
        
        # Process based on input type
        if input_path.is_file():
            # Single image
            output_file = output_path / f"upscaled_{input_path.name}"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("🔄 Upscaling image...", total=1)
                
                result = upscaler.upscale_single(str(input_path), str(output_file))
                progress.update(task, completed=1)
            
            console.print(f"✅ Saved upscaled image: {output_file}", style="green")
        
        elif input_path.is_dir():
            # Directory of images
            with console.status("🔍 Scanning directory..."):
                stats = upscaler.upscale_directory(
                    str(input_path),
                    str(output_path),
                    recursive=args.recursive
                )
            
            # Print results
            result_table = Table(title="📊 Processing Results")
            result_table.add_column("Metric", style="bold")
            result_table.add_column("Value", style="green")
            
            result_table.add_row("Images Processed", str(stats['processed']))
            result_table.add_row("Failed", str(stats['failed']))
            result_table.add_row("Total Time", f"{stats['time']:.1f}s")
            result_table.add_row("Avg Time/Image", f"{stats['avg_time_per_image']:.1f}s")
            
            console.print(result_table)
        
        else:
            console.print(f"❌ Unsupported input type: {input_path}", style="red")
            return 1
        
        # Show final stats
        if args.stats:
            stats = upscaler.get_stats()
            stats_table = Table(title="📈 Upscaler Statistics")
            stats_table.add_column("Metric", style="bold")
            stats_table.add_column("Value", style="cyan")
            
            for key, value in stats.items():
                if isinstance(value, float):
                    stats_table.add_row(key.replace('_', ' ').title(), f"{value:.3f}")
                else:
                    stats_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(stats_table)
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n⚠️  Process interrupted by user", style="yellow")
        return 1
    except Exception as e:
        console.print(f"\n❌ Error during upscaling: {e}", style="red")
        logger.exception("Upscaling failed")
        return 1


def cmd_benchmark(args):
    """Run performance benchmarks."""
    from bench.benchmark import run_benchmark_suite
    
    console.print("🏃‍♂️ Running Yamiro Upscaler benchmarks...", style="blue")
    
    try:
        results = run_benchmark_suite(
            model_name=args.model,
            device=args.device,
            duration=args.duration,
            resolutions=args.resolutions,
            batch_sizes=args.batch_sizes
        )
        
        # Save results
        if args.output:
            output_file = Path(args.output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            console.print(f"💾 Benchmark results saved: {output_file}", style="green")
        
        # Print summary
        console.print("📊 Benchmark completed!", style="green")
        
        return 0
        
    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="red")
        logger.exception("Benchmark failed")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🎌 Yamiro Upscaler - AI-powered anime image upscaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale a single image
  yamiro-upscale upscale -i image.jpg -o results/
  
  # Upscale all images in a directory
  yamiro-upscale upscale -i photos/ -o upscaled/ --recursive
  
  # Use different model and device
  yamiro-upscale upscale -i image.jpg -o results/ --model realesrgan_anime --device mps
  
  # Run benchmarks
  yamiro-upscale benchmark --duration 60 --output bench_results.json
  
  # Show system info
  yamiro-upscale info
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-banner', action='store_true', help='Skip the banner')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and device information')
    
    # Upscale command
    upscale_parser = subparsers.add_parser('upscale', help='Upscale images')
    upscale_parser.add_argument('--input', '-i', required=True, help='Input image file or directory')
    upscale_parser.add_argument('--output', '-o', required=True, help='Output directory')
    upscale_parser.add_argument('--model', '-m', default='realesrgan_x4',
                               choices=['realesrgan_x4', 'realesrgan_x2', 'realesrgan_anime'],
                               help='AI model to use')
    upscale_parser.add_argument('--device', '-d', default='auto',
                               choices=['auto', 'mps', 'cuda', 'cpu'],
                               help='Processing device')
    upscale_parser.add_argument('--tile-size', type=int, default=512,
                               help='Tile size for memory-efficient processing')
    upscale_parser.add_argument('--batch-size', type=int, default=1,
                               help='Batch size for processing')
    upscale_parser.add_argument('--half-precision', action='store_true',
                               help='Use half precision (FP16) for faster processing')
    upscale_parser.add_argument('--format', default='PNG',
                               choices=['PNG', 'JPEG', 'WEBP'],
                               help='Output image format')
    upscale_parser.add_argument('--quality', type=int, default=95,
                               help='Output quality for JPEG/WEBP (1-100)')
    upscale_parser.add_argument('--recursive', '-r', action='store_true',
                               help='Process subdirectories recursively')
    upscale_parser.add_argument('--stats', action='store_true',
                               help='Show processing statistics')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--model', default='realesrgan_x4',
                                 choices=['realesrgan_x4', 'realesrgan_x2', 'realesrgan_anime'],
                                 help='Model to benchmark')
    benchmark_parser.add_argument('--device', default='auto',
                                 choices=['auto', 'mps', 'cuda', 'cpu'],
                                 help='Device to benchmark')
    benchmark_parser.add_argument('--duration', type=int, default=30,
                                 help='Benchmark duration in seconds')
    benchmark_parser.add_argument('--resolutions', nargs='+', type=int, default=[512, 1024],
                                 help='Image resolutions to test')
    benchmark_parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4],
                                 help='Batch sizes to test')
    benchmark_parser.add_argument('--output', '-o', help='Save benchmark results to file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Show banner
    if not args.no_banner:
        print_banner()
    
    # Handle commands
    if args.command == 'info':
        return cmd_info(args)
    elif args.command == 'upscale':
        return cmd_upscale(args)
    elif args.command == 'benchmark':
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n⚠️  Interrupted by user", style="yellow")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n💥 Unexpected error: {e}", style="red")
        logger.exception("CLI crashed")
        sys.exit(1)