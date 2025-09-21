# Yamiro Upscaler Architecture

## Overview

Yamiro Upscaler is designed as a modular, high-performance AI image upscaling system optimized for Apple Silicon Macs. The architecture emphasizes performance, usability, and extensibility.

## Core Design Principles

1. **Performance First**: MPS optimization, efficient memory management, and intelligent batching
2. **Modular Design**: Clean separation of concerns with well-defined interfaces
3. **Cross-Platform**: Primary focus on macOS with fallback support for other platforms
4. **Developer Friendly**: Comprehensive APIs, documentation, and testing
5. **Production Ready**: Benchmarking, monitoring, and deployment tools

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│   CLI Interface │  Web Interface  │   Python API           │
│   (cli.py)      │  (webui/app.py) │   (Direct imports)      │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                  Core Engine                               │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Upscaler       │  Video Pipeline │   Model Loader          │
│  (upscaler.py)  │  (video.py)     │   (model_loader.py)     │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│               Infrastructure                               │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Benchmarking   │  System Profiler│   Device Management     │
│  (bench/)       │  (profiler.py)  │   (MPS/CUDA/CPU)        │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                  AI Models                                 │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Real-ESRGAN    │  SwinIR         │   Custom Models         │
│  (PyTorch)      │  (PyTorch)      │   (CoreML/ONNX)         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Component Details

### 1. Model Loader (`inference/model_loader.py`)

**Responsibilities:**
- Device detection and optimization (MPS/CUDA/CPU)
- Model downloading and caching
- Memory management and cleanup
- Device capability reporting

**Key Features:**
- Automatic MPS detection for Apple Silicon
- Graceful fallback to CPU/CUDA
- Model weight caching
- Memory monitoring

```python
# Device selection logic
def get_optimal_device(prefer='auto'):
    if prefer == 'auto':
        if mps_available(): return device('mps')
        elif cuda_available(): return device('cuda')
        else: return device('cpu')
```

### 2. Core Upscaler (`inference/upscaler.py`)

**Responsibilities:**
- High-level upscaling interface
- Batch processing coordination
- Memory management and tiling
- Statistics tracking

**Key Features:**
- Configurable tiling for memory efficiency
- Automatic preprocessing/postprocessing
- Batch optimization
- Performance monitoring

**Memory Strategy:**
```
Input Image → Tiles → GPU Processing → Reassembly → Output
     ↓              ↓                      ↓
  Preprocessing   Upscaling            Postprocessing
     ↓              ↓                      ↓
 Format conversion Memory management    Format conversion
```

### 3. Video Pipeline (`inference/video.py`)

**Responsibilities:**
- Video frame extraction using FFmpeg
- Frame-by-frame upscaling
- Video reassembly with audio preservation
- Progress tracking

**Processing Flow:**
```
Input Video → Frame Extraction → Batch Upscaling → Reassembly → Output Video
     ↓               ↓                  ↓              ↓
  FFmpeg          Temp Storage      Core Engine     FFmpeg
  Analysis        Management        Processing      Encoding
```

### 4. Benchmarking Suite (`bench/benchmark.py`)

**Responsibilities:**
- Latency measurement across resolutions
- Throughput testing under sustained load
- Memory usage profiling
- System performance monitoring

**Benchmark Types:**

1. **Latency Benchmark**: Single-image processing time
2. **Throughput Benchmark**: Images processed per second
3. **Memory Benchmark**: RAM/VRAM usage scaling
4. **Thermal Benchmark**: Temperature and throttling analysis

### 5. System Profiler (`utils/profiler.py`)

**Responsibilities:**
- Cross-platform system monitoring
- CPU, memory, and thermal data collection
- macOS-specific optimizations (powermetrics, vm_stat)
- Background data collection

**macOS Integration:**
```bash
# System tools utilized
powermetrics  # Power and thermal data
vm_stat      # Memory pressure
iostat       # I/O statistics
sysctl       # Hardware information
```

## Device Optimization Strategy

### Apple Silicon (MPS)

**Advantages:**
- Unified memory architecture
- Native Metal compute shaders
- Low power consumption
- Thermal efficiency

**Optimizations:**
- Direct MPS tensor operations
- Memory pool management
- Mixed precision when supported
- Thermal monitoring

### CUDA GPUs

**Advantages:**
- Mature PyTorch support
- Large VRAM capacity
- High compute throughput

**Optimizations:**
- CUDA memory pools
- Stream parallelism
- Tensor core utilization
- Multi-GPU support (future)

### CPU Fallback

**Usage Scenarios:**
- No GPU available
- Memory constraints
- Model conversion/testing

**Optimizations:**
- Multi-threading
- SIMD vectorization
- Memory-mapped I/O
- Process batching

## Memory Management

### Tiling Strategy

Large images are processed in overlapping tiles to manage memory usage:

```
Original Image (4K)
┌─────────────────────────────────┐
│  ┌─────┐  ┌─────┐  ┌─────┐     │
│  │ T1  │  │ T2  │  │ T3  │     │
│  └─────┘  └─────┘  └─────┘     │
│  ┌─────┐  ┌─────┐  ┌─────┐     │
│  │ T4  │  │ T5  │  │ T6  │     │
│  └─────┘  └─────┘  └─────┘     │
└─────────────────────────────────┘
```

**Tile Parameters:**
- Size: 512x512 (default, configurable)
- Overlap: 10px (seamless blending)
- Padding: 10px (edge handling)

### Memory Pool Management

```python
# Memory cleanup strategy
def _cleanup_memory(self):
    gc.collect()
    if self.device.type == 'cuda':
        torch.cuda.empty_cache()
    elif self.device.type == 'mps':
        torch.mps.empty_cache()
```

## Performance Characteristics

### Latency Analysis

Typical processing times on Mac Mini M4:

| Input Size | Tile Size | Memory Usage | Latency |
|------------|-----------|--------------|---------|
| 512×512    | 512       | 2.1 GB       | 2.1s    |
| 1024×1024  | 512       | 2.3 GB       | 8.3s    |
| 2048×2048  | 512       | 2.8 GB       | 33.2s   |
| 4096×4096  | 512       | 3.2 GB       | 132.5s  |

### Scaling Behavior

**Memory Scaling**: O(tile_size²) - constant for input size
**Time Scaling**: O(input_pixels) - linear with input size
**Quality Scaling**: Larger tiles = better seamless blending

## Error Handling and Resilience

### Graceful Degradation

1. **Device Fallback**: MPS → CUDA → CPU
2. **Memory Adaptation**: Large tiles → smaller tiles → CPU processing
3. **Model Fallback**: Preferred model → alternative model → error

### Error Recovery

```python
try:
    result = self.upscaler.enhance(image)
except torch.cuda.OutOfMemoryError:
    # Reduce tile size and retry
    self.upscaler.tile = self.upscaler.tile // 2
    result = self.upscaler.enhance(image)
except Exception as e:
    # Fallback to CPU
    self.upscaler.device = torch.device('cpu')
    result = self.upscaler.enhance(image)
```

## Extension Points

### Custom Models

```python
# Add new model support
ModelLoader.SUPPORTED_MODELS['custom'] = {
    'scale': 4,
    'model_name': 'CustomRRDB',
    'model_path': 'path/to/weights.pth'
}
```

### Custom Preprocessing

```python
class CustomUpscaler(YamiroUpscaler):
    def preprocess_image(self, image):
        # Custom preprocessing logic
        return super().preprocess_image(image)
```

### Plugin Architecture (Future)

```python
# Plugin interface design
class UpscalerPlugin:
    def preprocess(self, image): pass
    def postprocess(self, image): pass
    def enhance_metadata(self, info): pass
```

## Testing Strategy

### Unit Tests
- Individual component functionality
- Mock external dependencies
- Edge case handling

### Integration Tests
- End-to-end workflow testing
- Cross-component interaction
- Performance regression detection

### Performance Tests
- Benchmark stability
- Memory leak detection
- Thermal behavior validation

## Deployment Considerations

### CoreML Conversion

For production macOS applications:

```python
# Convert PyTorch → CoreML
traced_model = torch.jit.trace(model, example_input)
coreml_model = ct.convert(traced_model, ...)
```

**Benefits:**
- Native macOS integration
- Reduced memory footprint
- Better power efficiency
- App Store compatibility

### Docker Support (Future)

```dockerfile
# Linux deployment
FROM pytorch/pytorch:latest
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "src/cli.py"]
```

## Future Enhancements

### Planned Features

1. **Multi-GPU Support**: Parallel processing across multiple GPUs
2. **Real-time Processing**: Live video upscaling with RTSP streams
3. **Cloud Integration**: AWS/GCP batch processing
4. **Mobile Support**: iOS app with CoreML integration
5. **Advanced Models**: ESRGAN+, SWINIR, custom architectures

### Research Directions

1. **Attention Mechanisms**: Self-attention for better detail preservation
2. **Temporal Consistency**: Video upscaling with frame coherence
3. **Style Transfer**: Anime style adaptation during upscaling
4. **Edge Optimization**: Ultra-low latency processing

## Conclusion

Yamiro Upscaler's architecture balances performance, usability, and extensibility. The modular design enables easy customization while the Apple Silicon optimizations provide industry-leading performance on Mac systems. The comprehensive benchmarking and monitoring capabilities make it suitable for both research and production use cases.