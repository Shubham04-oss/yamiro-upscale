"""
Yamiro Upscaler - Performance Benchmarking Suite

Comprehensive benchmarking for latency, throughput, memory usage,
and thermal performance on Apple Silicon and other devices.
"""

import time
import json
import logging
import threading
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import gc
import subprocess
import platform

import numpy as np
from PIL import Image
import torch

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.upscaler import YamiroUpscaler, UpscalerConfig
from inference.model_loader import get_device_info, get_memory_info
from utils.profiler import SystemProfiler

logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.system_info = get_device_info()
        self.results = {}
    
    def add_test(self, test_name: str, results: Dict[str, Any]):
        """Add results from a specific test."""
        self.results[test_name] = {
            'timestamp': datetime.now().isoformat(),
            **results
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_timestamp': self.timestamp,
            'system_info': self.system_info,
            'results': self.results
        }
    
    def save(self, path: str):
        """Save results to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"💾 Benchmark results saved: {output_path}")


class LatencyBenchmark:
    """Benchmark inference latency with different image sizes."""
    
    def __init__(self, upscaler: YamiroUpscaler):
        self.upscaler = upscaler
    
    def generate_test_image(self, width: int, height: int) -> np.ndarray:
        """Generate a synthetic test image."""
        # Create a colorful synthetic image with patterns
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradients and patterns
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    (x * 255) // width,  # Red gradient
                    (y * 255) // height,  # Green gradient
                    ((x + y) * 255) // (width + height)  # Blue diagonal
                ]
        
        return image
    
    def run_single_test(
        self,
        width: int,
        height: int,
        iterations: int = 5,
        warmup_iterations: int = 2
    ) -> Dict[str, Any]:
        """Run latency test for a specific image size."""
        logger.info(f"🧪 Testing latency: {width}x{height} ({iterations} iterations)")
        
        # Generate test image
        test_image = self.generate_test_image(width, height)
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                self.upscaler.upscale_single(test_image)
                gc.collect()
                if self.upscaler.device.type == 'mps':
                    torch.mps.empty_cache()
                elif self.upscaler.device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Actual measurements
        times = []
        memory_before = []
        memory_after = []
        
        for i in range(iterations):
            # Memory before
            mem_before = get_memory_info(self.upscaler.device)
            memory_before.append(mem_before.get('allocated', 0))
            
            # Time the upscaling
            start_time = time.perf_counter()
            
            try:
                result = self.upscaler.upscale_single(test_image)
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                times.append(latency)
                
                # Memory after
                mem_after = get_memory_info(self.upscaler.device)
                memory_after.append(mem_after.get('allocated', 0))
                
                # Get output dimensions
                if i == 0:  # Only need this once
                    output_width, output_height = result.size
                
            except Exception as e:
                logger.error(f"Test iteration {i} failed: {e}")
                continue
            
            # Cleanup
            del result
            gc.collect()
            if self.upscaler.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.upscaler.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        if not times:
            raise RuntimeError("All test iterations failed")
        
        # Calculate statistics
        results = {
            'input_resolution': (width, height),
            'output_resolution': (output_width, output_height),
            'input_pixels': width * height,
            'output_pixels': output_width * output_height,
            'scale_factor': output_width / width,
            'iterations': len(times),
            'latency_stats': {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times),
                'raw_times': times
            },
            'memory_stats': {
                'mean_before': statistics.mean(memory_before),
                'mean_after': statistics.mean(memory_after),
                'peak_usage': max(memory_after) - min(memory_before),
                'raw_before': memory_before,
                'raw_after': memory_after
            },
            'throughput': {
                'pixels_per_second': (width * height) / statistics.mean(times),
                'images_per_second': 1.0 / statistics.mean(times)
            }
        }
        
        logger.info(f"✅ {width}x{height}: {results['latency_stats']['mean']:.3f}s ± {results['latency_stats']['stdev']:.3f}s")
        
        return results
    
    def run_suite(
        self,
        resolutions: List[Tuple[int, int]] = None,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Run latency tests across multiple resolutions."""
        if resolutions is None:
            resolutions = [
                (256, 256),
                (512, 512),
                (768, 768),
                (1024, 1024),
                (1536, 1536),
                (2048, 2048)
            ]
        
        logger.info(f"🚀 Running latency benchmark suite with {len(resolutions)} resolutions")
        
        results = {
            'test_type': 'latency',
            'device': str(self.upscaler.device),
            'model': self.upscaler.config.model_name,
            'iterations_per_test': iterations,
            'resolutions_tested': len(resolutions),
            'individual_results': []
        }
        
        for width, height in resolutions:
            try:
                test_result = self.run_single_test(width, height, iterations)
                results['individual_results'].append(test_result)
                
            except Exception as e:
                logger.error(f"❌ Failed to test {width}x{height}: {e}")
                continue
        
        # Calculate aggregate statistics
        if results['individual_results']:
            all_times = []
            all_throughputs = []
            
            for result in results['individual_results']:
                all_times.extend(result['latency_stats']['raw_times'])
                all_throughputs.append(result['throughput']['images_per_second'])
            
            results['aggregate_stats'] = {
                'overall_mean_latency': statistics.mean(all_times),
                'overall_median_latency': statistics.median(all_times),
                'mean_throughput': statistics.mean(all_throughputs),
                'total_tests_run': len(all_times)
            }
        
        logger.info(f"✅ Latency benchmark completed: {len(results['individual_results'])} successful tests")
        return results


class ThroughputBenchmark:
    """Benchmark sustained throughput over time."""
    
    def __init__(self, upscaler: YamiroUpscaler):
        self.upscaler = upscaler
    
    def run_test(
        self,
        resolution: Tuple[int, int] = (512, 512),
        duration: int = 30,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """Run sustained throughput test."""
        width, height = resolution
        logger.info(f"🔄 Running throughput test: {width}x{height}, {duration}s, batch_size={batch_size}")
        
        # Generate test images
        test_images = []
        for i in range(batch_size):
            # Slight variation in each image
            image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            test_images.append(image)
        
        # Warmup
        try:
            for image in test_images:
                self.upscaler.upscale_single(image)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        # Run sustained test
        start_time = time.perf_counter()
        end_time = start_time + duration
        
        processed_count = 0
        processing_times = []
        memory_samples = []
        
        while time.perf_counter() < end_time:
            batch_start = time.perf_counter()
            
            try:
                # Process batch
                for image in test_images:
                    self.upscaler.upscale_single(image)
                    processed_count += 1
                
                batch_end = time.perf_counter()
                batch_time = batch_end - batch_start
                processing_times.append(batch_time / batch_size)  # Per-image time
                
                # Sample memory
                mem_info = get_memory_info(self.upscaler.device)
                memory_samples.append(mem_info.get('allocated', 0))
                
                # Cleanup
                gc.collect()
                if self.upscaler.device.type == 'mps':
                    torch.mps.empty_cache()
                elif self.upscaler.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}")
                continue
        
        actual_duration = time.perf_counter() - start_time
        
        if not processing_times:
            raise RuntimeError("No successful processing iterations")
        
        results = {
            'test_type': 'throughput',
            'resolution': resolution,
            'target_duration': duration,
            'actual_duration': actual_duration,
            'batch_size': batch_size,
            'images_processed': processed_count,
            'batches_processed': len(processing_times),
            'throughput_stats': {
                'images_per_second': processed_count / actual_duration,
                'mean_time_per_image': statistics.mean(processing_times),
                'median_time_per_image': statistics.median(processing_times),
                'peak_time_per_image': max(processing_times),
                'min_time_per_image': min(processing_times)
            },
            'memory_stats': {
                'mean_usage': statistics.mean(memory_samples),
                'peak_usage': max(memory_samples),
                'min_usage': min(memory_samples),
                'memory_stability': statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
            }
        }
        
        logger.info(f"✅ Processed {processed_count} images in {actual_duration:.1f}s "
                   f"({results['throughput_stats']['images_per_second']:.2f} fps)")
        
        return results


class MemoryBenchmark:
    """Benchmark memory usage and scaling."""
    
    def __init__(self, upscaler: YamiroUpscaler):
        self.upscaler = upscaler
    
    def run_memory_scaling_test(
        self,
        resolutions: List[Tuple[int, int]] = None,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Test how memory usage scales with image size."""
        if resolutions is None:
            resolutions = [
                (256, 256), (512, 512), (1024, 1024), (2048, 2048)
            ]
        
        logger.info(f"🧠 Running memory scaling test with {len(resolutions)} resolutions")
        
        results = {
            'test_type': 'memory_scaling',
            'baseline_memory': None,
            'resolution_tests': []
        }
        
        # Get baseline memory
        gc.collect()
        if self.upscaler.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.upscaler.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        baseline_memory = get_memory_info(self.upscaler.device)
        results['baseline_memory'] = baseline_memory
        
        for width, height in resolutions:
            logger.info(f"🧪 Testing memory usage: {width}x{height}")
            
            memory_samples = []
            peak_memory = 0
            
            for i in range(iterations):
                # Clear memory
                gc.collect()
                if self.upscaler.device.type == 'mps':
                    torch.mps.empty_cache()
                elif self.upscaler.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Generate test image
                test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                
                # Measure memory before
                mem_before = get_memory_info(self.upscaler.device)
                
                try:
                    # Process image
                    result = self.upscaler.upscale_single(test_image)
                    
                    # Measure memory during/after
                    mem_after = get_memory_info(self.upscaler.device)
                    
                    memory_used = mem_after.get('allocated', 0) - baseline_memory.get('allocated', 0)
                    memory_samples.append(memory_used)
                    peak_memory = max(peak_memory, mem_after.get('allocated', 0))
                    
                    del result
                    
                except Exception as e:
                    logger.warning(f"Memory test iteration {i} failed: {e}")
                    continue
            
            if memory_samples:
                test_result = {
                    'resolution': (width, height),
                    'input_pixels': width * height,
                    'iterations': len(memory_samples),
                    'memory_usage': {
                        'mean': statistics.mean(memory_samples),
                        'peak': max(memory_samples),
                        'min': min(memory_samples),
                        'stdev': statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
                    },
                    'memory_efficiency': {
                        'bytes_per_input_pixel': statistics.mean(memory_samples) * 1024**3 / (width * height),
                        'peak_usage_gb': peak_memory
                    }
                }
                
                results['resolution_tests'].append(test_result)
                logger.info(f"✅ {width}x{height}: {test_result['memory_usage']['mean']:.2f}GB mean usage")
        
        return results


def run_benchmark_suite(
    model_name: str = 'realesrgan_x4',
    device: str = 'auto',
    duration: int = 30,
    resolutions: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
    output_dir: str = "bench/results"
) -> Dict[str, Any]:
    """
    Run complete benchmark suite.
    
    Args:
        model_name: AI model to benchmark
        device: Device to use for benchmarking
        duration: Duration for throughput tests
        resolutions: Image resolutions to test
        batch_sizes: Batch sizes for throughput tests
        output_dir: Directory to save results
        
    Returns:
        Complete benchmark results
    """
    logger.info(f"🚀 Starting Yamiro Upscaler benchmark suite")
    logger.info(f"📱 Model: {model_name}, Device: {device}")
    
    # Setup
    config = UpscalerConfig(model_name=model_name, device=device)
    upscaler = YamiroUpscaler(config)
    
    benchmark_result = BenchmarkResult()
    
    # Default test parameters
    if resolutions is None:
        resolutions = [256, 512, 1024]
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 4]
    
    test_resolutions = [(r, r) for r in resolutions]
    
    # System profiler
    profiler = SystemProfiler()
    profiler.start_monitoring()
    
    try:
        # 1. Latency Benchmark
        logger.info("📊 Running latency benchmarks...")
        latency_bench = LatencyBenchmark(upscaler)
        latency_results = latency_bench.run_suite(test_resolutions)
        benchmark_result.add_test('latency', latency_results)
        
        # 2. Throughput Benchmark
        logger.info("🔄 Running throughput benchmarks...")
        throughput_bench = ThroughputBenchmark(upscaler)
        
        for batch_size in batch_sizes:
            for resolution in test_resolutions:
                test_name = f"throughput_{resolution[0]}x{resolution[1]}_batch{batch_size}"
                throughput_results = throughput_bench.run_test(resolution, duration, batch_size)
                benchmark_result.add_test(test_name, throughput_results)
        
        # 3. Memory Benchmark
        logger.info("🧠 Running memory benchmarks...")
        memory_bench = MemoryBenchmark(upscaler)
        memory_results = memory_bench.run_memory_scaling_test(test_resolutions)
        benchmark_result.add_test('memory_scaling', memory_results)
        
        # 4. System metrics
        logger.info("🖥️  Collecting system metrics...")
        system_metrics = profiler.get_metrics()
        benchmark_result.add_test('system_metrics', system_metrics)
        
    finally:
        profiler.stop_monitoring()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"benchmark_{model_name}_{device}_{timestamp}.json"
    benchmark_result.save(str(output_file))
    
    logger.info("✅ Benchmark suite completed!")
    return benchmark_result.to_dict()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Yamiro Upscaler Benchmark Suite")
    parser.add_argument('--model', default='realesrgan_x4', help='Model to benchmark')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--duration', type=int, default=30, help='Throughput test duration')
    parser.add_argument('--output', default='bench/results', help='Output directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        results = run_benchmark_suite(
            model_name=args.model,
            device=args.device,
            duration=args.duration,
            output_dir=args.output
        )
        
        print("🎉 Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise