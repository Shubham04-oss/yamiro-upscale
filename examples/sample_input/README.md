# Sample Images

This directory contains sample images for testing Yamiro Upscaler.

## Test Images

Due to repository size constraints, we don't include large test images in the repo. 

### Downloading Sample Images

You can download anime-style test images from:

1. **Danbooru** (with proper licensing)
2. **Pixiv** (with artist permission)
3. **Safebooru** (safe content)

### Creating Test Images

Generate synthetic test images:

```python
import numpy as np
from PIL import Image

# Create a colorful test pattern
def create_test_image(size=(512, 512)):
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            image[y, x] = [
                (x * 255) // width,      # Red gradient
                (y * 255) // height,     # Green gradient
                ((x + y) * 255) // (width + height)  # Blue diagonal
            ]
    
    return Image.fromarray(image)

# Create and save test image
test_img = create_test_image()
test_img.save('test_pattern.png')
```

### Recommended Test Cases

1. **Low Resolution Anime Art** (256x256 to 512x512)
2. **Screenshots from Anime** (720p or 1080p)
3. **Pixel Art** (16x16 to 64x64)
4. **Character Portraits** (Various sizes)

### Usage Examples

```bash
# Single image
python src/cli.py upscale -i examples/sample_input/image.jpg -o examples/results/

# Batch processing
python src/cli.py upscale -i examples/sample_input/ -o examples/results/ --recursive

# Video processing
python src/inference/video.py input_video.mp4 output_video.mp4
```