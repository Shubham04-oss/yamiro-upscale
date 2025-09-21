"""
Yamiro Upscaler - System Profiler

Cross-platform system monitoring for performance analysis.
Includes CPU, memory, thermal, and power metrics collection.
"""

import time
import threading
import subprocess
import logging
import platform
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)


class SystemProfiler:
    """System performance profiler with platform-specific optimizations."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        self.samples = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        info = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
        
        # macOS specific info
        if platform.system() == 'Darwin':
            try:
                # Get macOS version
                result = subprocess.run(['sw_vers', '-productVersion'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    info['macos_version'] = result.stdout.strip()
                
                # Get hardware model
                result = subprocess.run(['sysctl', '-n', 'hw.model'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    info['hardware_model'] = result.stdout.strip()
                
            except Exception as e:
                logger.warning(f"Failed to get macOS info: {e}")
        
        return info
    
    def _sample_cpu_memory(self) -> Dict[str, float]:
        """Sample CPU and memory usage."""
        # CPU usage (per-core and overall)
        cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        cpu_overall = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            current_freq = cpu_freq.current if cpu_freq else 0
        except:
            current_freq = 0
        
        return {
            'cpu_percent_overall': cpu_overall,
            'cpu_percent_per_core': cpu_percent,
            'cpu_frequency_mhz': current_freq,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'swap_percent': psutil.swap_memory().percent
        }
    
    def _sample_macos_metrics(self) -> Dict[str, Any]:
        """Sample macOS-specific metrics using system tools."""
        metrics = {}
        
        try:
            # Power and thermal metrics using powermetrics (requires sudo for full info)
            # We'll use what's available without sudo
            result = subprocess.run(['powermetrics', '-i', '1000', '-n', '1', '--samplers', 'cpu_power,thermal'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse powermetrics output (simplified)
                output = result.stdout
                if 'CPU die temperature' in output:
                    # Extract temperature if available
                    for line in output.split('\n'):
                        if 'CPU die temperature' in line:
                            temp_str = line.split(':')[-1].strip()
                            if temp_str.endswith(' C'):
                                try:
                                    metrics['cpu_temperature_c'] = float(temp_str[:-2])
                                except:
                                    pass
                            break
        
        except Exception as e:
            logger.debug(f"powermetrics sampling failed: {e}")
        
        try:
            # Memory pressure using vm_stat
            result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                vm_lines = result.stdout.split('\n')
                vm_stats = {}
                
                for line in vm_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().replace(' ', '_').lower()
                        value = value.strip().rstrip('.')
                        
                        try:
                            # Extract numeric value (pages)
                            if value.isdigit():
                                vm_stats[key] = int(value)
                        except:
                            pass
                
                # Calculate memory pressure indicators
                if 'pages_free' in vm_stats and 'pages_active' in vm_stats:
                    page_size = 4096  # 4KB pages on macOS
                    total_pages = vm_stats.get('pages_free', 0) + vm_stats.get('pages_active', 0) + vm_stats.get('pages_inactive', 0)
                    
                    metrics['memory_pressure'] = {
                        'free_pages': vm_stats.get('pages_free', 0),
                        'active_pages': vm_stats.get('pages_active', 0),
                        'inactive_pages': vm_stats.get('pages_inactive', 0),
                        'compressed_pages': vm_stats.get('pages_compressed', 0),
                        'pressure_ratio': 1.0 - (vm_stats.get('pages_free', 0) / max(total_pages, 1))
                    }
        
        except Exception as e:
            logger.debug(f"vm_stat sampling failed: {e}")
        
        try:
            # I/O statistics
            result = subprocess.run(['iostat', '-d'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse iostat output for disk I/O
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'disk0' in line:  # Primary disk
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                metrics['disk_io'] = {
                                    'reads_per_sec': float(parts[1]),
                                    'writes_per_sec': float(parts[2])
                                }
                            except:
                                pass
                        break
        
        except Exception as e:
            logger.debug(f"iostat sampling failed: {e}")
        
        return metrics
    
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        logger.info("🔍 Starting system monitoring...")
        
        while self.monitoring:
            try:
                sample = {
                    'timestamp': time.time(),
                    'datetime': datetime.now().isoformat(),
                    **self._sample_cpu_memory()
                }
                
                # Add platform-specific metrics
                if platform.system() == 'Darwin':
                    macos_metrics = self._sample_macos_metrics()
                    sample.update(macos_metrics)
                
                self.samples.append(sample)
                
                # Limit sample history to prevent memory issues
                if len(self.samples) > 10000:
                    self.samples = self.samples[-5000:]  # Keep recent half
                
            except Exception as e:
                logger.warning(f"Monitoring sample failed: {e}")
            
            time.sleep(self.sample_interval)
        
        logger.info("🛑 System monitoring stopped")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.samples.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"📊 System monitoring started (interval: {self.sample_interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info(f"📊 System monitoring stopped ({len(self.samples)} samples collected)")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics and statistics."""
        if not self.samples:
            logger.warning("No monitoring samples available")
            return {'error': 'No samples collected'}
        
        # Calculate statistics from samples
        cpu_overall = [s['cpu_percent_overall'] for s in self.samples]
        memory_percent = [s['memory_percent'] for s in self.samples]
        memory_used = [s['memory_used_gb'] for s in self.samples]
        
        # CPU frequency stats
        cpu_freq = [s.get('cpu_frequency_mhz', 0) for s in self.samples if s.get('cpu_frequency_mhz', 0) > 0]
        
        # Temperature stats (if available)
        temperatures = [s.get('cpu_temperature_c') for s in self.samples if s.get('cpu_temperature_c')]
        
        stats = {
            'system_info': self.system_info,
            'monitoring_duration': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'],
            'sample_count': len(self.samples),
            'sample_interval': self.sample_interval,
            
            'cpu_stats': {
                'mean_percent': sum(cpu_overall) / len(cpu_overall),
                'max_percent': max(cpu_overall),
                'min_percent': min(cpu_overall),
                'samples': cpu_overall
            },
            
            'memory_stats': {
                'mean_percent': sum(memory_percent) / len(memory_percent),
                'max_percent': max(memory_percent),
                'min_percent': min(memory_percent),
                'mean_used_gb': sum(memory_used) / len(memory_used),
                'max_used_gb': max(memory_used),
                'samples_percent': memory_percent,
                'samples_gb': memory_used
            }
        }
        
        # Add frequency stats if available
        if cpu_freq:
            stats['cpu_frequency_stats'] = {
                'mean_mhz': sum(cpu_freq) / len(cpu_freq),
                'max_mhz': max(cpu_freq),
                'min_mhz': min(cpu_freq),
                'samples': cpu_freq
            }
        
        # Add temperature stats if available
        if temperatures:
            stats['thermal_stats'] = {
                'mean_temperature_c': sum(temperatures) / len(temperatures),
                'max_temperature_c': max(temperatures),
                'min_temperature_c': min(temperatures),
                'samples': temperatures
            }
        
        # Platform-specific stats
        if platform.system() == 'Darwin':
            # Memory pressure stats
            pressure_samples = [s.get('memory_pressure', {}).get('pressure_ratio', 0) for s in self.samples 
                               if s.get('memory_pressure')]
            
            if pressure_samples:
                stats['memory_pressure_stats'] = {
                    'mean_pressure_ratio': sum(pressure_samples) / len(pressure_samples),
                    'max_pressure_ratio': max(pressure_samples),
                    'samples': pressure_samples
                }
        
        return stats
    
    def save_raw_samples(self, filepath: str):
        """Save raw monitoring samples to file."""
        with open(filepath, 'w') as f:
            json.dump({
                'system_info': self.system_info,
                'samples': self.samples
            }, f, indent=2)
        
        logger.info(f"💾 Raw monitoring data saved: {filepath}")
    
    def get_current_snapshot(self) -> Dict[str, Any]:
        """Get a single snapshot of current system state."""
        snapshot = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            **self._sample_cpu_memory()
        }
        
        if platform.system() == 'Darwin':
            macos_metrics = self._sample_macos_metrics()
            snapshot.update(macos_metrics)
        
        return snapshot


def benchmark_profiler():
    """Test the profiler functionality."""
    logger.info("🧪 Testing SystemProfiler...")
    
    profiler = SystemProfiler(sample_interval=0.5)
    
    # Test snapshot
    snapshot = profiler.get_current_snapshot()
    logger.info(f"📸 Current snapshot: CPU {snapshot['cpu_percent_overall']:.1f}%, "
               f"Memory {snapshot['memory_percent']:.1f}%")
    
    # Test monitoring
    profiler.start_monitoring()
    
    # Simulate some work
    time.sleep(5)
    
    profiler.stop_monitoring()
    
    # Get results
    metrics = profiler.get_metrics()
    logger.info(f"📊 Monitoring results: {len(metrics.get('cpu_stats', {}).get('samples', []))} samples")
    logger.info(f"   CPU: {metrics['cpu_stats']['mean_percent']:.1f}% avg, {metrics['cpu_stats']['max_percent']:.1f}% max")
    logger.info(f"   Memory: {metrics['memory_stats']['mean_percent']:.1f}% avg")
    
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark_profiler()