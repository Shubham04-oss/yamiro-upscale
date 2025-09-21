#!/usr/bin/env python3
"""
Quick test script to verify Yamiro Upscaler functionality
without requiring model downloads.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set OpenMP fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from inference.model_loader import get_device_info, get_optimal_device, ModelLoader
        print("✅ model_loader imported successfully")
        
        from inference.upscaler import UpscalerConfig
        print("✅ upscaler config imported successfully")
        
        from utils.profiler import SystemProfiler
        print("✅ profiler imported successfully")
        
        from bench.benchmark import BenchmarkResult
        print("✅ benchmark imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_device_detection():
    """Test device detection and selection."""
    print("\n🔍 Testing device detection...")
    
    try:
        from inference.model_loader import get_device_info, get_optimal_device
        
        info = get_device_info()
        print(f"📱 Platform: {info['platform']}")
        print(f"🔧 Machine: {info['machine']}")
        print(f"🐍 Python: {info['python_version']}")
        print(f"🔥 PyTorch: {info['torch_version']}")
        print(f"🚀 MPS Available: {info['mps_available']}")
        print(f"💻 CUDA Available: {info['cuda_available']}")
        
        device = get_optimal_device('auto')
        print(f"✅ Selected device: {device}")
        
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_system_profiler():
    """Test system profiling functionality."""
    print("\n📊 Testing system profiler...")
    
    try:
        from utils.profiler import SystemProfiler
        
        profiler = SystemProfiler()
        snapshot = profiler.get_current_snapshot()
        
        print(f"💻 CPU Usage: {snapshot['cpu_percent_overall']:.1f}%")
        print(f"🧠 Memory Usage: {snapshot['memory_percent']:.1f}%")
        print(f"💾 Memory Used: {snapshot['memory_used_gb']:.1f} GB")
        
        # Quick monitoring test
        profiler.start_monitoring()
        import time
        time.sleep(1)
        profiler.stop_monitoring()
        
        metrics = profiler.get_metrics()
        print(f"📈 Monitoring samples: {metrics['sample_count']}")
        
        return True
    except Exception as e:
        print(f"❌ System profiler test failed: {e}")
        return False

def test_config_creation():
    """Test configuration objects."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from inference.upscaler import UpscalerConfig
        
        # Test default config
        config = UpscalerConfig()
        print(f"✅ Default config: {config.model_name}, {config.device}")
        
        # Test custom config
        custom_config = UpscalerConfig(
            model_name='realesrgan_anime',
            device='mps',
            half_precision=True,
            tile_size=256
        )
        print(f"✅ Custom config: {custom_config.model_name}, tile_size={custom_config.tile_size}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model_loader():
    """Test model loader without actually downloading models."""
    print("\n📦 Testing model loader...")
    
    try:
        from inference.model_loader import ModelLoader
        
        loader = ModelLoader()
        print(f"✅ Model loader created, models dir: {loader.models_dir}")
        
        # Check supported models
        print(f"📚 Supported models: {list(loader.SUPPORTED_MODELS.keys())}")
        
        for model_name, model_info in loader.SUPPORTED_MODELS.items():
            print(f"  - {model_name}: {model_info['scale']}x scale")
        
        return True
    except Exception as e:
        print(f"❌ Model loader test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎌 Yamiro Upscaler - Quick Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_device_detection,
        test_system_profiler,
        test_config_creation,
        test_model_loader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Yamiro Upscaler is ready to use.")
        print("\nNext steps:")
        print("1. Try the CLI: python src/cli.py info")
        print("2. Launch web interface: python src/webui/app.py")
        print("3. Run benchmarks: python src/cli.py benchmark")
        
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())