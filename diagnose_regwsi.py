#!/usr/bin/env python3
"""
Diagnostic script for the RegWSI pipeline.
Checks system configuration, GPU status, dependencies and model files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import json
import tempfile

CHECK = "✓"
CROSS = "✗"


# ----------------------------------------------------------------------------- #
# Utility
# ----------------------------------------------------------------------------- #
def header(title: str) -> None:
    print(f"\n{'=' * 60}\n {title}\n{'=' * 60}")


def run_cmd(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()


# ----------------------------------------------------------------------------- #
# System info
# ----------------------------------------------------------------------------- #
def check_system() -> None:
    header("System information")

    # CPU
    try:
        cpu = run_cmd("lscpu | grep 'Model name' | awk -F ':' '{print $2}'")
        print(f"CPU : {cpu.strip()}")
    except Exception:
        print("CPU : Unable to detect")

    # RAM
    try:
        mem = run_cmd("free -h | awk '/^Mem:/ {print $2, $3, $4}'").split()
        print(f"RAM : {mem[0]} total, {mem[1]} used, {mem[2]} free")
    except Exception:
        print("RAM : Unable to detect")

    # Disk
    try:
        disk = run_cmd("df -h /workspace | tail -1").split()
        print(f"Disk: {disk[1]} total, {disk[2]} used, {disk[3]} free")
    except Exception:
        print("Disk: Unable to detect")


# ----------------------------------------------------------------------------- #
# GPU
# ----------------------------------------------------------------------------- #
def check_gpu() -> None:
    header("GPU information")
    try:
        import torch  # noqa: WPS433

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version    : {torch.version.cuda}")
            cnt = torch.cuda.device_count()
            print(f"GPU count       : {cnt}")

            for i in range(cnt):
                p = torch.cuda.get_device_properties(i)
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"\nGPU {i}: {p.name}")
                print(f"  Memory total  : {p.total_memory / 1024**3:.1f} GB")
                print(f"  Compute cap.  : {p.major}.{p.minor}")
                print(f"  Allocated     : {alloc:.1f} GB")
                print(f"  Reserved      : {reserved:.1f} GB")
        else:
            print("No CUDA devices detected")
    except ImportError:
        print("PyTorch not installed")
    except Exception as exc:
        print(f"GPU check failed: {exc}")

    # nvidia‑smi
    print("\nnvidia‑smi:")
    try:
        subprocess.call(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu", "--format=csv,noheader"]
        )
    except Exception:
        print("nvidia‑smi not available")


# ----------------------------------------------------------------------------- #
# Python dependencies
# ----------------------------------------------------------------------------- #
def check_dependencies() -> None:
    header("Python dependencies")

    required: Dict[str, str | None] = {
        "numpy": "1.24.0",
        "scipy": "1.10.0",
        "scikit-image": "0.19.3",
        "SimpleITK": "2.2.1",
        "opencv-python": "4.7.0",
        "tifffile": "2023.7.10",
        "torch": "2.0.0",
        "deeperhistreg": None,
        "pyvips": None,
        "cupy": None,
        "cucim": None,
    }

    from packaging import version as v  # noqa: WPS433, E402

    for pkg, min_ver in required.items():
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "unknown")
            ok = True
            if min_ver and ver != "unknown":
                ok = v.parse(ver) >= v.parse(min_ver)
            symbol = CHECK if ok else CROSS
            note = "" if ok or min_ver is None else f"(≥ {min_ver})"
            print(f"{pkg:20} {symbol}  {ver} {note}")
        except ImportError:
            print(f"{pkg:20} {CROSS}  Not installed")


# ----------------------------------------------------------------------------- #
# External tools
# ----------------------------------------------------------------------------- #
def check_tools() -> None:
    header("External tools")

    tools = {
        "bfconvert": "/workspace/regwsi/bftools/bfconvert",
        "java": "java",
        "python3": "python3",
        "git": "git",
    }

    for tool, path in tools.items():
        try:
            if tool == "bfconvert":
                status = CHECK if Path(path).exists() else CROSS
                where = f"({path})" if status == CHECK else ""
                print(f"{tool:20} {status}  {where}")
            elif tool == "java":
                ver = run_cmd(f"{path} -version").splitlines()[0]
                print(f"{tool:20} {CHECK}  {ver}")
            else:
                ver = run_cmd(f"{path} --version").splitlines()[0]
                print(f"{tool:20} {CHECK}  {ver}")
        except Exception:
            print(f"{tool:20} {CROSS}  Not found")


# ----------------------------------------------------------------------------- #
# Model files
# ----------------------------------------------------------------------------- #
def check_models() -> None:
    header("DeeperHistReg model files")

    model_dir = Path("/workspace/regwsi/DeeperHistReg/deeperhistreg")
    if not model_dir.exists():
        print(f"{CROSS} Model directory missing: {model_dir}")
        print("Download from:")
        print("https://drive.google.com/drive/folders/1rZca3fKvPLGhoNvaAXJFpfgYYJ9LdKVZ")
        return

    print(f"{CHECK} Model directory: {model_dir}")
    expected = ["superpoint.pth", "superglue.pth", "initial_alignment.pth"]
    for m in expected:
        p = model_dir / m
        if p.exists():
            size = p.stat().st_size / 1024**2
            print(f"  {m:30} {CHECK} ({size:.1f} MB)")
        else:
            print(f"  {m:30} {CROSS} Not found")


# ----------------------------------------------------------------------------- #
# Image format test
# ----------------------------------------------------------------------------- #
def test_image_loading() -> None:
    header("Image format support")

    import numpy as np  # noqa: WPS433
    import tifffile  # noqa: WPS433

    formats = {
        "TIFF": "test.tiff",
        "OME‑TIFF": "test.ome.tiff",
        "PNG": "test.png",
        "JPEG": "test.jpg",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        for name, fname in formats.items():
            try:
                f = tmp / fname
                if "TIFF" in name:
                    tifffile.imwrite(f, data)
                    _ = tifffile.imread(f)
                else:
                    cv2.imwrite(str(f), data)
                    _ = cv2.imread(str(f))
                print(f"{name:15} {CHECK} Read/write OK")
            except Exception as exc:
                print(f"{name:15} {CROSS} {str(exc)[:40]}...")


# ----------------------------------------------------------------------------- #
# GPU memory allocation test
# ----------------------------------------------------------------------------- #
def test_gpu_memory() -> None:
    header("GPU memory test")

    try:
        import torch  # noqa: WPS433

        if not torch.cuda.is_available():
            print("No CUDA device available")
            return

        sizes_gb: List[int] = [1, 10, 50, 100, 120]
        for g in sizes_gb:
            try:
                elems = int(g * 1024**3 / 4)  # float32
                t = torch.zeros(elems, device="cuda")
                print(f"Allocated {g} GB {CHECK}")
                del t
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc):
                    print(f"Reached limit at {g} GB")
                    break
                print(f"Failed at {g} GB: {exc}")
                break
    except ImportError:
        print("PyTorch not installed")


# ----------------------------------------------------------------------------- #
# Entry
# ----------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 60)
    print(" RegWSI diagnostic tool")
    print("=" * 60)

    check_system()
    check_gpu()
    check_dependencies()
    check_tools()
    check_models()
    test_image_loading()
    test_gpu_memory()

    header("Diagnostic complete")
    print("\nIf issues were found:")
    print("  1. Re‑run the setup script to install missing deps")
    print("  2. Download model files if absent")
    print("  3. Verify GPU drivers / CUDA toolkit")
    print("  4. Ensure adequate disk space\n")


if __name__ == "__main__":
    sys.exit(main())
