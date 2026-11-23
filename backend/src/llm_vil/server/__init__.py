"""Live localhost server: FastAPI REST + WebSocket delta stream over the async SimEngine."""

from .app import create_app

__all__ = ["create_app"]
