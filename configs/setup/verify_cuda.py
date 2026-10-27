"""
Diagnostic script to verify PyTorch, CUDA availability, and scientific stack dependencies.
"""

from __future__ import annotations

import platform
import sys


def verify_environment() -> None:
    print("=" * 70)
    print("  Gait360 Environment & CUDA Diagnostic Check")
    print("=" * 70)
    print(f"Python Version : {sys.version.split()[0]} ({platform.platform()})")

    for pkg in ("numpy", "scipy", "pandas", "sklearn", "cv2", "matplotlib", "mediapipe"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"[OK] {pkg:<12}: {ver}")
        except ImportError:
            print(f"[MISSING] {pkg:<12}: not installed in active environment")

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        print(f"[OK] torch       : {torch.__version__} (CUDA available: {cuda_ok})")
        if cuda_ok:
            print(f"     GPU Device  : {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[INFO] torch     : not installed in this environment (use delta_env for GPU inference)")


if __name__ == "__main__":
    verify_environment()
