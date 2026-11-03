"""I9 changes only its training objective; I8's renderer/adapter are reused."""
from ..covariance_transport.runtime import install, renderer_for

__all__ = ['install', 'renderer_for']
