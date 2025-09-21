# 🎌 Yamiro Upscaler

> AI-Powered Anime Image Upscaling • Optimized for Apple Silicon (MPS)

[![CI](https://github.com/username/yamiro-upscale/workflows/CI/badge.svg)](https://github.com/username/yamiro-upscale/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-green.svg)](https://www.apple.com/mac/)

A high-performance anime image upscaler using Real-ESRGAN and SwinIR, specifically optimized for Apple Silicon Macs (M1/M2/M3/M4). Features comprehensive benchmarking, video processing, and a beautiful web interface.

## ✨ Features

- 🚀 **Apple Silicon Optimized**: Native MPS (Metal Performance Shaders) support
- 🎨 **Multiple AI Models**: Real-ESRGAN x2/x4, Real-ESRGAN Anime
- 🖼️ **Batch Processing**: Process thousands of images efficiently  
- 🎬 **Video Upscaling**: Frame extraction, upscaling, and reassembly
- 🌐 **Web Interface**: Beautiful Gradio-based demo with real-time preview
- 📊 **Comprehensive Benchmarking**: Latency, throughput, memory, and thermal analysis
- 🔧 **CoreML Support**: Convert models for maximum macOS performance
- 💻 **CLI & API**: Flexible interfaces for all use cases

## 🚀 Quick Start

### Prerequisites

- macOS 12+ (Apple Silicon recommended)
- Python 3.9+
- 8GB+ RAM (16GB+ recommended for large images)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/yamiro-upscale.git
   cd yamiro-upscale
   ```

2. **Run the setup script** (macOS)
   ```bash
   chmod +x tools/install_macos.sh
   ./tools/install_macos.sh
   ```

3. **Activate environment and install PyTorch**
   ```bash
   conda activate yamiro-upscale
   pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

4. **Test installation**
   ```bash
   python src/cli.py info
   ```

### Manual Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate yamiro-upscale

# Install PyTorch with MPS support
pip3 install torch torchvision

# Install dependencies
pip install -r requirements.txt
```

## 💡 Usage Examples

### Command Line Interface

```bash
# Upscale a single image
python src/cli.py upscale -i image.jpg -o results/ --model realesrgan_x4 --device mps

# Batch process a directory
python src/cli.py upscale -i photos/ -o upscaled/ --recursive --batch-size 4

# Process video
python src/inference/video.py input.mp4 output_upscaled.mp4

# Run benchmarks
python src/cli.py benchmark --duration 30 --output benchmark_results.json

# Show system information
python src/cli.py info
```

### Web Interface

```bash
# Launch web demo
python src/webui/app.py

# Custom configuration
python src/webui/app.py --host 0.0.0.0 --port 8080 --share
```

Open http://localhost:7860 in your browser.

### Python API

```python
from src.inference.upscaler import create_upscaler
from PIL import Image

# Create upscaler
upscaler = create_upscaler(model_name='realesrgan_x4', device='mps')

# Upscale single image
image = Image.open('input.jpg')
result = upscaler.upscale_single(image, save_path='output.png')

# Batch processing
upscaler.upscale_directory('input_dir/', 'output_dir/')

# Get performance stats
stats = upscaler.get_stats()
print(f"Processed {stats['images_processed']} images")
```

## 🏗️ Architecture

### Core Components

```
yamiro-upscale/
├── src/
│   ├── inference/          # AI model inference
│   │   ├── model_loader.py  # Device detection & model loading
│   │   ├── upscaler.py      # Main upscaling engine
│   │   └── video.py         # Video processing pipeline
│   ├── bench/              # Performance benchmarking
│   ├── webui/              # Gradio web interface
│   └── utils/              # System profiling & utilities
├── tests/                  # Unit tests
├── tools/                  # Setup & conversion scripts
└── examples/               # Sample inputs & results
```

### Device Support

| Device | Status | Performance |
|--------|--------|-------------|
| Apple Silicon (M1/M2/M3/M4) | ✅ Optimized | Excellent |
| Intel Mac with AMD GPU | ⚠️ Limited | Good |
| NVIDIA GPU (CUDA) | ✅ Supported | Excellent |
| CPU Only | ✅ Supported | Basic |

## 📊 Benchmarks

### Apple Mac Mini M4 (Example Results)

| Resolution | Model | Device | Latency | Throughput |
|------------|-------|--------|---------|------------|
| 512×512 | Real-ESRGAN x4 | MPS | 2.1s | 0.48 fps |
| 1024×1024 | Real-ESRGAN x4 | MPS | 8.3s | 0.12 fps |
| 512×512 | Real-ESRGAN x2 | MPS | 1.2s | 0.83 fps |

*Benchmarks vary by hardware configuration and system load.*

### Running Benchmarks

```bash
# Quick benchmark
python src/cli.py benchmark --duration 30

# Comprehensive benchmark
python src/bench/benchmark.py --model realesrgan_x4 --duration 60 --output results.json

# Analyze results
python notebooks/benchmark_analysis.ipynb
```

## 🎯 Performance Tips

### Memory Optimization
- Use tiling for large images: `--tile-size 512`
- Enable half precision: `--half-precision`
- Process in batches: `--batch-size 4`

### Speed Optimization
- Use MPS device on Apple Silicon: `--device mps`
- Convert to CoreML for production: `tools/convert_to_coreml.py`
- Warm up models before batch processing

### Quality vs Speed
- **Fastest**: Real-ESRGAN x2, small tile size
- **Balanced**: Real-ESRGAN x4, medium tile size
- **Best Quality**: Real-ESRGAN Anime, large tile size

## 🔧 Advanced Usage

### CoreML Conversion

```bash
# Convert to CoreML for maximum performance
python tools/convert_to_coreml.py \
    --model realesrgan_x4 \
    --output models/realesrgan_x4.mlmodel \
    --input-size 512 512 \
    --test
```

### Custom Models

```python
# Add custom Real-ESRGAN models
from src.inference.model_loader import ModelLoader

loader = ModelLoader()
loader.SUPPORTED_MODELS['custom_model'] = {
    'scale': 4,
    'model_name': 'CustomModel',
    'model_path': 'path/to/model.pth'
}
```

### Video Processing

```python
from src.inference.video import VideoUpscaler

upscaler = VideoUpscaler()
stats = upscaler.upscale_video(
    'input.mp4',
    'output.mp4',
    start_time=10.0,    # Start at 10 seconds
    end_time=30.0,      # End at 30 seconds
    preserve_audio=True
)
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_yamiro_upscaler.py::TestModelLoader -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `black src/ && pytest tests/`
5. Submit a pull request

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api.md)
- [Benchmark Analysis](notebooks/benchmark_analysis.ipynb)
- [Performance Tuning](docs/performance.md)

## 🆘 Troubleshooting

### Common Issues

**MPS not available**
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"

# Install latest PyTorch nightly
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Out of memory errors**
- Reduce tile size: `--tile-size 256`
- Lower batch size: `--batch-size 1`
- Use CPU: `--device cpu`

**Slow performance**
- Use MPS device: `--device mps`
- Enable half precision: `--half-precision`
- Convert to CoreML: `tools/convert_to_coreml.py`

### Getting Help

- 📧 **Email**: contact@yamiro.dev
- 🐛 **Issues**: [GitHub Issues](https://github.com/username/yamiro-upscale/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/username/yamiro-upscale/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for the amazing upscaling models
- [BasicSR](https://github.com/XPixelGroup/BasicSR) for the framework
- [Gradio](https://gradio.app/) for the web interface
- Apple for Metal Performance Shaders

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=username/yamiro-upscale&type=Date)](https://star-history.com/#username/yamiro-upscale&Date)

---

<div align="center">
  <p>Made with ❤️ for the anime community</p>
  <p>🎌 <strong>Yamiro Upscaler</strong> - Bringing your favorite art to life</p>
</div>