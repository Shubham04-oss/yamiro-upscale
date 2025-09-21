"""Yamiro Upscaler - Web Interface Module"""

try:
    from .app import create_yamiro_interface, launch_yamiro_app, get_system_info
    __all__ = ['create_yamiro_interface', 'launch_yamiro_app', 'get_system_info']
except ImportError:
    # Gradio not available
    __all__ = []