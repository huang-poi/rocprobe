# ROCProbe Python Bindings
# Provides a Python interface to the ROCProbe GPU profiler for MI300X

"""
ROCProbe — Python bindings for the ROCm GPU profiler.

Usage:
    from rocprobe import Profiler

    profiler = Profiler(device=0)

    @profiler.profile
    def train_step(batch):
        logits = model(batch)
        loss = loss_fn(logits, targets)
        loss.backward()
        return loss

    result = train_step(batch)
    print(result.summary())
"""

__version__ = "0.3.1"
__author__ = "huang-poi"
__license__ = "MIT"

from .client import Profiler, Session, ProfileResult

__all__ = ["Profiler", "Session", "ProfileResult"]
