"""Report the Python, PyTorch, CUDA, and GPU environment."""

from __future__ import annotations

import platform
import sys

import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'not included'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is unavailable; VehicleAI will use the CPU.")
        return 1
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    tensor = torch.rand((1024, 1024), device="cuda")
    result = tensor @ tensor
    torch.cuda.synchronize()
    print(f"CUDA computation test: PASS (mean={result.mean().item():.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

