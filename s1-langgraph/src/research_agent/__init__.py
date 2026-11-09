"""Series S1 research agent."""

from .config import Settings, load_settings
from .model import make_model

__all__ = ["Settings", "load_settings", "make_model"]
