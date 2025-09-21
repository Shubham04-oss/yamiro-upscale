"""
Yamiro Upscaler - Video Processing Pipeline

Handles video frame extraction, upscaling, and reassembly using FFmpeg.
Optimized for batch processing with progress tracking.
"""

import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json
import time
import threading
import queue

import cv2
import numpy as np
from PIL import Image
from rich.progress import Progress, TaskID
from rich.console import Console

from .upscaler import YamiroUpscaler, UpscalerConfig

logger = logging.getLogger(__name__)
console = Console()


class VideoInfo:
    """Container for video metadata."""
    
    def __init__(self, video_path: str):
        self.path = Path(video_path)
        self.fps = 0.0
        self.frame_count = 0
        self.duration = 0.0
        self.width = 0
        self.height = 0
        self.codec = ""
        self.bitrate = 0
        
        self._extract_info()
    
    def _extract_info(self):
        """Extract video information using FFprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(self.path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                self.fps = eval(video_stream.get('r_frame_rate', '0/1'))
                self.frame_count = int(video_stream.get('nb_frames', 0))
                self.width = int(video_stream.get('width', 0))
                self.height = int(video_stream.get('height', 0))
                self.codec = video_stream.get('codec_name', '')
            
            # Format info
            format_info = data.get('format', {})
            self.duration = float(format_info.get('duration', 0))
            self.bitrate = int(format_info.get('bit_rate', 0))
            
            # Calculate frame count if not available
            if self.frame_count == 0 and self.fps > 0 and self.duration > 0:
                self.frame_count = int(self.fps * self.duration)
                
        except Exception as e:
            logger.warning(f"Failed to extract video info: {e}")
            # Fallback using cv2
            self._extract_info_cv2()
    
    def _extract_info_cv2(self):
        """Fallback video info extraction using OpenCV."""
        try:
            cap = cv2.VideoCapture(str(self.path))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / max(self.fps, 1)
            cap.release()
            
        except Exception as e:
            logger.error(f"Failed to extract video info with cv2: {e}")
    
    def __str__(self):
        return (f"Video: {self.path.name} "
                f"({self.width}x{self.height}, {self.fps:.1f}fps, "
                f"{self.frame_count} frames, {self.duration:.1f}s)")


class VideoUpscaler:
    """Video upscaling with frame extraction and reassembly."""
    
    def __init__(self, upscaler_config: Optional[UpscalerConfig] = None):
        self.upscaler_config = upscaler_config or UpscalerConfig()
        self.upscaler = None
        self.temp_dir = None
        
    def _setup_upscaler(self):
        """Initialize the image upscaler."""
        if self.upscaler is None:
            self.upscaler = YamiroUpscaler(self.upscaler_config)
    
    def _setup_temp_dir(self) -> Path:
        """Create temporary directory for frame processing."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="yamiro_video_"))
        return self.temp_dir
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Tuple[int, Path]:
        """
        Extract frames from video using FFmpeg.
        
        Args:
            video_path: Input video path
            output_dir: Directory to save frames
            start_time: Start time in seconds
            end_time: End time in seconds
            max_frames: Maximum number of frames to extract
            
        Returns:
            Tuple of (frame_count, output_directory)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-v', 'warning']
        
        # Input options
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        
        cmd.extend(['-i', str(video_path)])
        
        # Output options
        if end_time is not None:
            cmd.extend(['-t', str(end_time - (start_time or 0))])
        
        if max_frames is not None:
            cmd.extend(['-frames:v', str(max_frames)])
        
        # High quality frame extraction
        cmd.extend([
            '-q:v', '1',  # Best quality
            '-pix_fmt', 'rgb24',  # RGB format
            str(output_path / 'frame_%06d.png')
        ])
        
        logger.info(f"Extracting frames: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Count extracted frames
            frame_files = list(output_path.glob('frame_*.png'))
            frame_count = len(frame_files)
            
            logger.info(f"✅ Extracted {frame_count} frames to {output_path}")
            return frame_count, output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Frame extraction failed: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def upscale_frames(
        self,
        frames_dir: str,
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> int:
        """
        Upscale all frames in a directory.
        
        Args:
            frames_dir: Directory containing input frames
            output_dir: Directory to save upscaled frames
            progress_callback: Optional callback for progress updates
            
        Returns:
            Number of successfully processed frames
        """
        self._setup_upscaler()
        
        frames_path = Path(frames_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all frame files
        frame_files = sorted(frames_path.glob('frame_*.png'))
        if not frame_files:
            raise ValueError(f"No frame files found in {frames_dir}")
        
        logger.info(f"🔄 Upscaling {len(frame_files)} frames...")
        
        processed = 0
        failed = 0
        
        for i, frame_file in enumerate(frame_files):
            try:
                # Generate output filename
                output_file = output_path / f"upscaled_{frame_file.name}"
                
                # Skip if already exists
                if output_file.exists():
                    processed += 1
                    continue
                
                # Upscale frame
                self.upscaler.upscale_single(str(frame_file), str(output_file))
                processed += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(frame_files))
                
            except Exception as e:
                logger.error(f"❌ Failed to upscale frame {frame_file}: {e}")
                failed += 1
        
        logger.info(f"✅ Frame upscaling complete: {processed} processed, {failed} failed")
        return processed
    
    def reassemble_video(
        self,
        frames_dir: str,
        output_path: str,
        fps: float,
        original_video: Optional[str] = None,
        codec: str = 'libx264',
        bitrate: Optional[str] = None,
        copy_audio: bool = True
    ):
        """
        Reassemble frames into video using FFmpeg.
        
        Args:
            frames_dir: Directory containing upscaled frames
            output_path: Output video path
            fps: Frame rate for output video
            original_video: Original video for audio/metadata copying
            codec: Video codec to use
            bitrate: Video bitrate (auto if None)
            copy_audio: Copy audio from original video
        """
        frames_path = Path(frames_dir)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for upscaled frames
        frame_pattern = str(frames_path / 'upscaled_frame_%06d.png')
        frame_files = list(frames_path.glob('upscaled_frame_*.png'))
        
        if not frame_files:
            raise ValueError(f"No upscaled frames found in {frames_dir}")
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-v', 'warning']
        
        # Input frames
        cmd.extend([
            '-framerate', str(fps),
            '-i', frame_pattern
        ])
        
        # Add original video for audio
        if copy_audio and original_video and Path(original_video).exists():
            cmd.extend(['-i', str(original_video)])
        
        # Video encoding options
        cmd.extend([
            '-c:v', codec,
            '-pix_fmt', 'yuv420p',  # Compatible format
            '-preset', 'medium',  # Balance speed/quality
        ])
        
        # Bitrate
        if bitrate:
            cmd.extend(['-b:v', bitrate])
        else:
            cmd.extend(['-crf', '18'])  # High quality
        
        # Audio options
        if copy_audio and original_video:
            cmd.extend(['-c:a', 'copy'])  # Copy audio stream
            cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])  # Map video from frames, audio from original
        else:
            cmd.extend(['-an'])  # No audio
        
        cmd.append(str(output_file))
        
        logger.info(f"Reassembling video: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"✅ Video saved: {output_file}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Video reassembly failed: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def upscale_video(
        self,
        input_video: str,
        output_video: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None,
        preserve_audio: bool = True,
        cleanup_temp: bool = True
    ) -> Dict[str, Any]:
        """
        Complete video upscaling pipeline.
        
        Args:
            input_video: Input video path
            output_video: Output video path
            start_time: Start time for processing (seconds)
            end_time: End time for processing (seconds)
            max_frames: Maximum frames to process
            preserve_audio: Copy audio from original
            cleanup_temp: Clean up temporary files
            
        Returns:
            Dictionary with processing statistics
        """
        start_total = time.time()
        
        # Get video info
        video_info = VideoInfo(input_video)
        logger.info(f"📽️  Processing: {video_info}")
        
        # Setup temporary directories
        temp_dir = self._setup_temp_dir()
        frames_dir = temp_dir / "frames"
        upscaled_dir = temp_dir / "upscaled"
        
        stats = {
            'input_video': str(input_video),
            'output_video': str(output_video),
            'original_resolution': (video_info.width, video_info.height),
            'fps': video_info.fps,
            'total_frames': video_info.frame_count,
            'processed_frames': 0,
            'extraction_time': 0,
            'upscaling_time': 0,
            'reassembly_time': 0,
            'total_time': 0
        }
        
        try:
            # Step 1: Extract frames
            logger.info("📤 Step 1: Extracting frames...")
            start_step = time.time()
            
            frame_count, _ = self.extract_frames(
                input_video,
                str(frames_dir),
                start_time,
                end_time,
                max_frames
            )
            
            stats['extraction_time'] = time.time() - start_step
            stats['processed_frames'] = frame_count
            
            # Step 2: Upscale frames
            logger.info("🔄 Step 2: Upscaling frames...")
            start_step = time.time()
            
            # Progress tracking
            with Progress(console=console) as progress:
                task = progress.add_task("Upscaling frames...", total=frame_count)
                
                def update_progress(current, total):
                    progress.update(task, completed=current)
                
                processed_count = self.upscale_frames(
                    str(frames_dir),
                    str(upscaled_dir),
                    update_progress
                )
            
            stats['upscaling_time'] = time.time() - start_step
            
            # Step 3: Reassemble video
            logger.info("📥 Step 3: Reassembling video...")
            start_step = time.time()
            
            # Determine output bitrate based on upscaling
            upscale_factor = self.upscaler_config.model_name[-1] if self.upscaler_config.model_name[-1].isdigit() else 4
            original_bitrate = video_info.bitrate
            if original_bitrate > 0:
                # Increase bitrate proportionally to resolution increase
                new_bitrate = f"{int(original_bitrate * (int(upscale_factor) ** 2) / 1000)}k"
            else:
                new_bitrate = None
            
            self.reassemble_video(
                str(upscaled_dir),
                output_video,
                video_info.fps,
                input_video if preserve_audio else None,
                bitrate=new_bitrate,
                copy_audio=preserve_audio
            )
            
            stats['reassembly_time'] = time.time() - start_step
            stats['total_time'] = time.time() - start_total
            
            # Get output video info
            try:
                output_info = VideoInfo(output_video)
                stats['output_resolution'] = (output_info.width, output_info.height)
                stats['upscale_factor'] = output_info.width / video_info.width
            except:
                pass
            
            logger.info(f"✅ Video upscaling completed in {stats['total_time']:.1f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Video upscaling failed: {e}")
            raise
            
        finally:
            if cleanup_temp:
                self._cleanup_temp_dir()


def upscale_video(
    input_video: str,
    output_video: str,
    model_name: str = 'realesrgan_x4',
    device: str = 'auto',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for video upscaling.
    
    Args:
        input_video: Input video path
        output_video: Output video path
        model_name: AI model to use
        device: Processing device
        **kwargs: Additional options
        
    Returns:
        Processing statistics
    """
    config = UpscalerConfig(model_name=model_name, device=device)
    upscaler = VideoUpscaler(config)
    
    return upscaler.upscale_video(input_video, output_video, **kwargs)


if __name__ == "__main__":
    # Test video processing
    import sys
    
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        logger.info("🧪 Testing video upscaling...")
        stats = upscale_video(input_file, output_file)
        
        logger.info("📊 Final statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info("Usage: python video.py <input_video> <output_video>")