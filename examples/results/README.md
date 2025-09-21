# Results Directory

This directory contains upscaled images and processing results.

## Directory Structure

```
results/
├── images/          # Upscaled images
├── videos/          # Upscaled videos  
├── benchmarks/      # Benchmark results
└── comparisons/     # Before/after comparisons
```

## File Naming Convention

- **Single images**: `upscaled_[original_name].[format]`
- **Batch processing**: `upscaled_[timestamp]/`
- **Video output**: `[original_name]_upscaled.[format]`

## Benchmark Results

Benchmark files are saved as JSON with timestamps:
```
benchmark_realesrgan_x4_mps_20231201_143022.json
```

## Usage Tips

1. **Keep originals**: Always preserve your original images
2. **Compare results**: Use different models to find the best quality
3. **Check file sizes**: Upscaled images will be significantly larger
4. **Backup important results**: Consider cloud storage for valuable outputs

## Example Results Analysis

```python
import json
from pathlib import Path

# Load benchmark results
results_file = "benchmark_results.json"
with open(results_file) as f:
    data = json.load(f)

# Analyze performance
print(f"Device: {data['system_info']['platform']}")
print(f"Average latency: {data['results']['latency']['aggregate_stats']['overall_mean_latency']:.3f}s")
```