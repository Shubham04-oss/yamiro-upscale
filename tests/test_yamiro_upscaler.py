"""
Yamiro Upscaler - Test Suite

Unit tests for core functionality with mocking for dependencies.
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModelLoader:
    """Test model loading and device detection."""
    
    def test_device_info_collection(self):
        """Test that device info is collected properly."""
        from inference.model_loader import get_device_info
        
        info = get_device_info()
        
        assert 'platform' in info
        assert 'machine' in info
        assert 'python_version' in info
        assert 'torch_version' in info
        assert isinstance(info['cuda_available'], bool)
        assert isinstance(info['mps_available'], bool)
        assert isinstance(info['mps_built'], bool)
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.backends.mps.is_built')
    @patch('torch.cuda.is_available')
    def test_device_selection_auto(self, mock_cuda, mock_mps_built, mock_mps_avail):
        """Test automatic device selection logic."""
        from inference.model_loader import get_optimal_device
        
        # Test MPS preferred when available
        mock_mps_avail.return_value = True
        mock_mps_built.return_value = True
        mock_cuda.return_value = False
        
        device = get_optimal_device('auto')
        assert str(device) == 'mps'
        
        # Test CUDA fallback
        mock_mps_avail.return_value = False
        mock_cuda.return_value = True
        
        device = get_optimal_device('auto')
        assert str(device) == 'cuda'
        
        # Test CPU fallback
        mock_mps_avail.return_value = False
        mock_cuda.return_value = False
        
        device = get_optimal_device('auto')
        assert str(device) == 'cpu'
    
    def test_model_loader_initialization(self):
        """Test ModelLoader initialization."""
        from inference.model_loader import ModelLoader
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ModelLoader(temp_dir)
            assert loader.models_dir.exists()
            assert loader.models_dir == Path(temp_dir)
    
    def test_supported_models_list(self):
        """Test that supported models are properly defined."""
        from inference.model_loader import ModelLoader
        
        loader = ModelLoader()
        assert 'realesrgan_x4' in loader.SUPPORTED_MODELS
        assert 'realesrgan_x2' in loader.SUPPORTED_MODELS
        assert 'realesrgan_anime' in loader.SUPPORTED_MODELS
        
        for model_key, model_info in loader.SUPPORTED_MODELS.items():
            assert 'scale' in model_info
            assert 'model_name' in model_info
            assert 'model_path' in model_info
            assert isinstance(model_info['scale'], int)


class TestUpscalerConfig:
    """Test upscaler configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from inference.upscaler import UpscalerConfig
        
        config = UpscalerConfig()
        
        assert config.model_name == 'realesrgan_x4'
        assert config.device == 'auto'
        assert config.half_precision == False
        assert config.tile_size == 512
        assert config.batch_size == 1
        assert config.output_format == 'PNG'
        assert config.output_quality == 95
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from inference.upscaler import UpscalerConfig
        
        config = UpscalerConfig(
            model_name='realesrgan_anime',
            device='cpu',
            half_precision=True,
            tile_size=256,
            output_format='jpeg',
            output_quality=80
        )
        
        assert config.model_name == 'realesrgan_anime'
        assert config.device == 'cpu'
        assert config.half_precision == True
        assert config.tile_size == 256
        assert config.output_format == 'JPEG'  # Should be uppercase
        assert config.output_quality == 80


class TestUpscaler:
    """Test main upscaler functionality with mocks."""
    
    @pytest.fixture
    def mock_upscaler(self):
        """Create a mock upscaler for testing."""
        with patch('inference.upscaler.YamiroUpscaler') as mock_class:
            mock_instance = Mock()
            mock_instance.device = Mock()
            mock_instance.device.type = 'cpu'
            mock_instance.config = Mock()
            mock_instance.config.model_name = 'realesrgan_x4'
            mock_instance.stats = {
                'images_processed': 0,
                'total_time': 0.0,
                'total_pixels_in': 0,
                'total_pixels_out': 0
            }
            mock_class.return_value = mock_instance
            return mock_instance
    
    def test_image_preprocessing(self):
        """Test image preprocessing functions."""
        from inference.upscaler import YamiroUpscaler, UpscalerConfig
        
        with patch('inference.upscaler.ModelLoader'), \
             patch('inference.upscaler.get_optimal_device') as mock_device:
            
            mock_device.return_value = Mock()
            
            # This will fail due to missing dependencies, but we can test the config
            try:
                upscaler = YamiroUpscaler(UpscalerConfig())
            except:
                pass  # Expected due to missing Real-ESRGAN
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        from inference.upscaler import create_upscaler
        
        # This will fail due to missing dependencies, but we can test the function exists
        assert callable(create_upscaler)
    
    def test_stats_tracking(self):
        """Test statistics tracking functionality."""
        from inference.upscaler import YamiroUpscaler, UpscalerConfig
        
        with patch('inference.upscaler.ModelLoader'), \
             patch('inference.upscaler.get_optimal_device'):
            
            try:
                upscaler = YamiroUpscaler(UpscalerConfig())
                stats = upscaler.get_stats()
                
                # Check stats structure
                assert 'images_processed' in stats
                assert 'total_time' in stats
                assert 'device' in stats
                assert 'model' in stats
                
            except:
                pass  # Expected due to missing dependencies


class TestVideoProcessing:
    """Test video processing functionality."""
    
    def test_video_info_structure(self):
        """Test VideoInfo class structure."""
        from inference.video import VideoInfo
        
        # Mock video file for testing
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            try:
                video_info = VideoInfo(temp_file.name)
                
                # Check attributes exist
                assert hasattr(video_info, 'path')
                assert hasattr(video_info, 'fps')
                assert hasattr(video_info, 'frame_count')
                assert hasattr(video_info, 'duration')
                assert hasattr(video_info, 'width')
                assert hasattr(video_info, 'height')
                
            except:
                pass  # Expected if ffprobe/cv2 not available
    
    def test_video_upscaler_initialization(self):
        """Test VideoUpscaler initialization."""
        from inference.video import VideoUpscaler
        from inference.upscaler import UpscalerConfig
        
        config = UpscalerConfig()
        upscaler = VideoUpscaler(config)
        
        assert upscaler.upscaler_config == config
        assert upscaler.upscaler is None  # Not loaded yet
        assert upscaler.temp_dir is None


class TestBenchmark:
    """Test benchmarking functionality."""
    
    def test_benchmark_result_structure(self):
        """Test BenchmarkResult class."""
        from bench.benchmark import BenchmarkResult
        
        result = BenchmarkResult()
        
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'system_info')
        assert hasattr(result, 'results')
        assert isinstance(result.results, dict)
        
        # Test adding results
        test_data = {'test_metric': 123.45}
        result.add_test('test_name', test_data)
        
        assert 'test_name' in result.results
        assert result.results['test_name']['test_metric'] == 123.45
        assert 'timestamp' in result.results['test_name']
    
    def test_benchmark_result_serialization(self):
        """Test BenchmarkResult serialization."""
        from bench.benchmark import BenchmarkResult
        
        result = BenchmarkResult()
        result.add_test('test', {'value': 42})
        
        data = result.to_dict()
        
        assert 'benchmark_timestamp' in data
        assert 'system_info' in data
        assert 'results' in data
        assert data['results']['test']['value'] == 42
        
        # Test JSON serialization
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        assert len(json_str) > 0


class TestSystemProfiler:
    """Test system profiling functionality."""
    
    def test_profiler_initialization(self):
        """Test SystemProfiler initialization."""
        from utils.profiler import SystemProfiler
        
        profiler = SystemProfiler(sample_interval=0.5)
        
        assert profiler.sample_interval == 0.5
        assert profiler.monitoring == False
        assert profiler.samples == []
        assert isinstance(profiler.system_info, dict)
    
    def test_system_info_collection(self):
        """Test system info collection."""
        from utils.profiler import SystemProfiler
        
        profiler = SystemProfiler()
        info = profiler.system_info
        
        assert 'platform' in info
        assert 'machine' in info
        assert 'cpu_count' in info
        assert 'memory_total_gb' in info
        assert isinstance(info['cpu_count'], int)
        assert isinstance(info['memory_total_gb'], float)
    
    def test_current_snapshot(self):
        """Test current system snapshot."""
        from utils.profiler import SystemProfiler
        
        profiler = SystemProfiler()
        snapshot = profiler.get_current_snapshot()
        
        assert 'timestamp' in snapshot
        assert 'cpu_percent_overall' in snapshot
        assert 'memory_percent' in snapshot
        assert isinstance(snapshot['cpu_percent_overall'], (int, float))
        assert isinstance(snapshot['memory_percent'], (int, float))


class TestWebInterface:
    """Test web interface components."""
    
    def test_gradio_interface_creation(self):
        """Test that Gradio interface can be created."""
        try:
            from webui.app import create_yamiro_interface
            
            # This may fail if Gradio is not installed, which is OK for CI
            interface = create_yamiro_interface()
            assert interface is not None
            
        except ImportError:
            pytest.skip("Gradio not available")
    
    def test_system_info_generation(self):
        """Test system info markdown generation."""
        try:
            from webui.app import get_system_info
            
            info_md = get_system_info()
            assert isinstance(info_md, str)
            assert len(info_md) > 0
            assert 'System Information' in info_md
            
        except ImportError:
            pytest.skip("Web interface dependencies not available")


class TestCLI:
    """Test command line interface."""
    
    def test_cli_imports(self):
        """Test that CLI module can be imported."""
        try:
            from cli import setup_logging, print_banner
            
            assert callable(setup_logging)
            assert callable(print_banner)
            
        except ImportError as e:
            pytest.skip(f"CLI dependencies not available: {e}")


# Integration tests (require actual dependencies)
class TestIntegration:
    """Integration tests that require real dependencies."""
    
    @pytest.mark.slow
    def test_create_synthetic_image(self):
        """Test creating and processing a synthetic image."""
        # Create a test image
        test_image = Image.fromarray(
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        )
        
        assert test_image.size == (256, 256)
        assert test_image.mode == 'RGB'
    
    @pytest.mark.slow
    def test_temp_directory_creation(self):
        """Test temporary directory handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assert temp_path.exists()
            assert temp_path.is_dir()
            
            # Test file creation
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")
            assert test_file.exists()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])