#!/usr/bin/env python3
"""
Complete RegWSI Pipeline for H&E → mIF registration
Optimized for NVIDIA H200 GPU with 140 GB VRAM.
Handles QPTIFF → OME‑TIFF conversion, registration, and tile extraction for Pix2PixHD.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import psutil
import tifffile
import torch
import zarr
from skimage import exposure, transform
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Optional dependencies
try:
    from deeperhistreg import deeperhistreg
    from deeperhistreg.elastix import Elastix
except ImportError:
    deeperhistreg = None
    Elastix = None
    print("Warning: DeeperHistReg not properly installed")

import SimpleITK as sitk
import pyvips


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #
class Config:
    """Pipeline configuration optimized for H200 GPU."""

    def __init__(self) -> None:
        # Paths
        self.BFCONVERT_PATH = "/workspace/regwsi/bftools/bfconvert"
        self.WORK_DIR = Path("/workspace/data")
        self.INPUT_DIR = self.WORK_DIR / "input"
        self.OUTPUT_DIR = self.WORK_DIR / "output"
        self.TILES_DIR = self.WORK_DIR / "tiles"

        # GPU settings for H200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_memory_limit = 120 * 1024**3  # use 120 GB of 140 GB
        self.tile_batch_size = 64

        # Registration parameters
        self.registration_params: Dict[str, Any] = {
            "initial_alignment": {
                "scales": 8,
                "rotation_angles": 12,
                "use_superpoint": True,
                "use_superglue": True,
            },
            "nonrigid": {
                "similarity_measure": "ncc",
                "regularization": "diffusive",
                "levels": 4,
                "iterations": [100, 100, 50, 50],
                "learning_rates": [0.1, 0.05, 0.01, 0.005],
                "instance_optimization": True,
            },
        }

        # Tile extraction parameters
        self.tile_params: Dict[str, Any] = {
            "tile_size": 512,  # for Pix2PixHD
            "overlap": 0,
            "tissue_threshold": 0.5,
            "quality_threshold": 0.8,
            "min_tissue_area": 0.3,
        }

        # Processing parameters
        self.downsample_he = 2  # downsample H&E from 20× to 10×
        self.if_channel = 0  # DAPI channel (index 0)
        self.num_workers = 16  # parallel workers


# ----------------------------------------------------------------------------- #
# GPU memory manager
# ----------------------------------------------------------------------------- #
class GPUMemoryManager:
    """Manage GPU memory for H200."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.total_memory = props.total_memory
            print(f"\nGPU: {props.name}")
            print(f"Total GPU memory: {self.total_memory / 1024 ** 3:.1f} GB")
            # Use 85 % of GPU memory
            torch.cuda.set_per_process_memory_fraction(0.85)

    def clear_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

    def get_memory_stats(self) -> Dict[str, float]:
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            free = (self.total_memory - torch.cuda.memory_allocated()) / 1024 ** 3
            return {"allocated_gb": allocated, "reserved_gb": reserved, "free_gb": free}
        return {}


# ----------------------------------------------------------------------------- #
# Image preprocessing
# ----------------------------------------------------------------------------- #
class ImagePreprocessor:
    """Handle QPTIFF → OME‑TIFF conversion and preprocessing."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.memory_manager = GPUMemoryManager(config.device)

    # ---------- conversion ---------- #
    def convert_qptiff_to_ometiff(
        self,
        input_path: str | Path,
        output_path: str | Path,
        downsample: int = 1,
    ) -> Path:
        """Convert QPTIFF to pyramidal OME‑TIFF (optionally downsample)."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        print(f"\nConverting {input_path.name} → OME‑TIFF…")

        # Bio‑Formats conversion
        bfconvert = Path(self.config.BFCONVERT_PATH)
        if bfconvert.exists():
            cmd = f'"{bfconvert}" -compression LZW -pyramid-scale 2 "{input_path}" "{output_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"bfconvert error:\n{result.stderr}")
                raise RuntimeError("bfconvert failed")

        # Downsample if requested
        if downsample > 1:
            print(f"Downsampling ×{downsample}…")
            down_path = output_path.with_name(f"{output_path.stem}_down{output_path.suffix}")
            image = pyvips.Image.new_from_file(str(output_path))
            image.resize(1.0 / downsample).write_to_file(
                str(down_path),
                compression="lzw",
                tile=True,
                pyramid=True,
                tile_width=512,
                tile_height=512,
            )
            return down_path

        return output_path

    # ---------- IF channel extraction ---------- #
    def extract_if_channel(
        self,
        input_path: str | Path,
        output_path: str | Path,
        channel_idx: int = 0,
    ) -> Path:
        """Extract a specific channel from a multi‑channel IF image."""
        print(f"\nExtracting channel {channel_idx} from IF image…")

        with tifffile.TiffFile(input_path) as tif:
            data = tif.asarray()

        if data.ndim == 4:  # Z,C,Y,X
            channel_data = data[0, channel_idx]
        elif data.ndim == 3:  # C,Y,X
            channel_data = data[channel_idx]
        else:
            raise ValueError(f"Unexpected image shape: {data.shape}")

        metadata = {
            "axes": "YX",
            "Channel": {"Name": "DAPI"},
            "PhysicalSizeX": 1.0,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        }

        tifffile.imwrite(
            output_path,
            channel_data,
            photometric="minisblack",
            compression="lzw",
            tile=(512, 512),
            metadata=metadata,
            ome=True,
            bigtiff=True,
        )
        return Path(output_path)

    # ---------- preprocessing for registration ---------- #
    @staticmethod
    def prepare_for_registration(image_path: str | Path) -> np.ndarray:
        """Prepare image (grayscale, normalized, CLAHE)."""
        if str(image_path).lower().endswith((".tif", ".tiff")):
            image = tifffile.imread(image_path)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image.ndim == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply((image * 255).astype(np.uint8))

        return image_clahe.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------- #
# Registration
# ----------------------------------------------------------------------------- #
class RegWSIRegistration:
    """Run registration via DeeperHistReg, with a custom fallback."""

    def __init__(self, config: Config, memory_manager: GPUMemoryManager) -> None:
        self.config = config
        self.memory_manager = memory_manager
        self.device = config.device
        self.elastix = Elastix() if Elastix else None

    # ---------- DeeperHistReg ---------- #
    def register_deeperhistreg(
        self,
        source_path: str | Path,
        target_path: str | Path,
        output_dir: str | Path,
    ) -> Tuple[Path, Path]:
        """Attempt registration via DeeperHistReg; fallback to custom."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if deeperhistreg is None:
            print("DeeperHistReg unavailable — using custom registration.")
            return self.register_custom(source_path, target_path, output_dir)

        try:
            config_path = Path(__file__).parent / "configs" / "regwsi_config.yaml"
            if not config_path.exists():
                config_path.parent.mkdir(parents=True, exist_ok=True)
                import yaml

                yaml.dump(
                    {
                        "initial_alignment": {
                            "use_rotation": True,
                            "rotation_step": 30,
                            "use_flipping": True,
                            "use_scale": True,
                            "scale_factors": [0.5, 0.75, 1.0, 1.25, 1.5],
                        },
                        "nonrigid_registration": {
                            "pyramid_levels": 4,
                            "pyramid_schedule": [8, 4, 2, 1],
                            "iterations": 100,
                            "similarity_measure": "ncc",
                        },
                    },
                    config_path.open("w"),
                )

            paths = deeperhistreg.run_registration(
                str(source_path),
                str(target_path),
                str(output_dir),
                config_path=str(config_path),
            )
            displaced_path, displacement_path, registered_path = map(Path, paths)
            return Path(registered_path), Path(displacement_path)
        except Exception as exc:
            print(f"DeeperHistReg failed: {exc}")
            print("Falling back to custom implementation.")
            return self.register_custom(source_path, target_path, output_dir)

    # ---------- custom registration ---------- #
    def register_custom(
        self,
        source_path: str | Path,
        target_path: str | Path,
        output_dir: str | Path,
    ) -> Tuple[Path, Path]:
        """Custom registration using SIFT + Elastix."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n=== Custom RegWSI Registration ===")

        preproc = ImagePreprocessor(self.config)
        src = preproc.prepare_for_registration(source_path)
        tgt = preproc.prepare_for_registration(target_path)

        # Save preprocessed
        (output_dir / "preprocessed").mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "preprocessed/source.png"), (src * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / "preprocessed/target.png"), (tgt * 255).astype(np.uint8))

        # Initial alignment
        print("\n1. Initial alignment…")
        M = self._initial_alignment(src, tgt)
        h, w = tgt.shape
        src_aligned = cv2.warpAffine(src, M, (w, h))
        cv2.imwrite(str(output_dir / "initial_alignment.png"), (src_aligned * 255).astype(np.uint8))

        # Non‑rigid registration with Elastix
        print("\n2. Non‑rigid registration…")
        src_sitk = sitk.GetImageFromArray(src_aligned)
        tgt_sitk = sitk.GetImageFromArray(tgt)

        pmap = sitk.GetDefaultParameterMap("bspline")
        pmap["Metric"] = ["AdvancedNormalizedCorrelation"]
        pmap["NumberOfResolutions"] = ["4"]
        pmap["MaximumNumberOfIterations"] = ["100"]
        pmap["FinalBSplineInterpolationOrder"] = ["3"]

        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(tgt_sitk)
        elastix.SetMovingImage(src_sitk)
        elastix.SetParameterMap(pmap)
        elastix.Execute()

        result_img = sitk.GetArrayFromImage(elastix.GetResultImage())
        registered_path = output_dir / "registered_source.tiff"
        tifffile.imwrite(
            registered_path,
            (result_img * 255).astype(np.uint8),
            photometric="minisblack",
            compression="lzw",
        )

        transform_path = output_dir / "transform_parameters.txt"
        sitk.WriteParameterFile(elastix.GetTransformParameterMap()[0], str(transform_path))

        # Quality metrics
        mse = float(np.mean((tgt - result_img) ** 2))
        ssim_score = float(ssim(tgt, result_img))

        print(f"\nMSE: {mse:.6f}")
        print(f"SSIM: {ssim_score:.4f}")

        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "source": str(source_path),
                "target": str(target_path),
                "initial_transform": M.tolist(),
                "quality_metrics": {"mse": mse, "ssim": ssim_score},
            },
            (output_dir / "registration_report.json").open("w"),
            indent=2,
        )

        return registered_path, transform_path

    # ---------- helpers ---------- #
    @staticmethod
    def _initial_alignment(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Affine (rotation/translation) via SIFT + RANSAC."""
        sift = cv2.SIFT_create(nfeatures=10_000)
        kp1, des1 = sift.detectAndCompute((src * 255).astype(np.uint8), None)
        kp2, des2 = sift.detectAndCompute((tgt * 255).astype(np.uint8), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        print(f"   Good matches: {len(good)}")

        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            return M
        return np.eye(2, 3, dtype=np.float32)

    # Placeholder for full‑WSI transform
    @staticmethod
    def apply_transform_to_wsi(wsi_path: Path, transform_path: Path, output_path: Path) -> Path:
        print("\n[TODO] Apply transform to full‑resolution WSI.")
        return output_path


# ----------------------------------------------------------------------------- #
# Tile extraction
# ----------------------------------------------------------------------------- #
class TileExtractor:
    """Extract paired tiles for Pix2PixHD."""

    def __init__(self, config: Config, memory_manager: GPUMemoryManager) -> None:
        self.config = config
        self.memory_manager = memory_manager

    # ---------- tile extraction ---------- #
    def extract_paired_tiles(
        self,
        registered_src: str | Path,
        target_path: str | Path,
        output_dir: str | Path,
    ) -> int:
        output_dir = Path(output_dir)
        a_dir = output_dir / "train_A"  # H&E
        b_dir = output_dir / "train_B"  # IF

        a_dir.mkdir(parents=True, exist_ok=True)
        b_dir.mkdir(parents=True, exist_ok=True)

        tile_size = self.config.tile_params["tile_size"]
        print(f"\nExtracting {tile_size} × {tile_size} tiles…")

        src_img = tifffile.imread(registered_src)
        tgt_img = tifffile.imread(target_path)

        if src_img.shape != tgt_img.shape:
            print(f"Resizing source {src_img.shape} → {tgt_img.shape}")
            src_img = cv2.resize(src_img, (tgt_img.shape[1], tgt_img.shape[0]))

        h, w = src_img.shape[:2]
        n_tiles_h = h // tile_size
        n_tiles_w = w // tile_size
        total_tiles = n_tiles_h * n_tiles_w

        valid = 0
        with tqdm(total=total_tiles, desc="Tiles") as bar:
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    y, x = i * tile_size, j * tile_size
                    src_tile = src_img[y : y + tile_size, x : x + tile_size]
                    tgt_tile = tgt_img[y : y + tile_size, x : x + tile_size]

                    if self._is_valid_tile(src_tile):
                        name = f"tile_{i:04d}_{j:04d}.png"
                        cv2.imwrite(str(a_dir / name), self._to_rgb(src_tile))
                        cv2.imwrite(str(b_dir / name), self._to_rgb(tgt_tile))
                        valid += 1
                    bar.update(1)

        print(f"\nValid tiles: {valid} / {total_tiles} ({valid / total_tiles * 100:.1f} %)")

        json.dump(
            {
                "source_image": str(registered_src),
                "target_image": str(target_path),
                "tile_size": tile_size,
                "total_tiles": total_tiles,
                "valid_tiles": valid,
                "timestamp": datetime.now().isoformat(),
            },
            (output_dir / "dataset_info.json").open("w"),
            indent=2,
        )
        return valid

    # ---------- helpers ---------- #
    @staticmethod
    def _to_rgb(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img

    def _is_valid_tile(self, tile: np.ndarray) -> bool:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) if tile.ndim == 3 else tile

        # Tissue threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary > 0) / binary.size < self.config.tile_params["min_tissue_area"]:
            return False

        # Variance / brightness checks
        if np.var(gray) < 0.01 or not (10 < np.mean(gray) < 245):
            return False

        return True


# ----------------------------------------------------------------------------- #
# Pipeline orchestrator
# ----------------------------------------------------------------------------- #
class RegWSIPipeline:
    """Orchestrate the full pipeline."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.memory_manager = GPUMemoryManager(self.config.device)
        self.pre = ImagePreprocessor(self.config)
        self.reg = RegWSIRegistration(self.config, self.memory_manager)
        self.tiles = TileExtractor(self.config, self.memory_manager)

    # ---------- run ---------- #
    def run(self, he_qptiff: str | Path, if_qptiff: str | Path, out_base: str | Path) -> Path | None:
        out_base = Path(out_base)
        run_dir = out_base / f"regwsi_run_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"RegWSI Pipeline — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 80)

        try:
            # Step 1 – Image prep
            print("\n### Step 1: Image preparation ###")
            he_ome = run_dir / "he_10x.ome.tiff"
            he_ome = self.pre.convert_qptiff_to_ometiff(he_qptiff, he_ome, self.config.downsample_he)

            if_dapi = run_dir / "if_dapi.ome.tiff"
            if_dapi = self.pre.extract_if_channel(if_qptiff, if_dapi, self.config.if_channel)

            self.memory_manager.clear_cache()

            # Step 2 – Registration
            print("\n### Step 2: Registration ###")
            reg_dir = run_dir / "registration"
            reg_dir.mkdir(exist_ok=True)
            reg_img, transform = self.reg.register_deeperhistreg(he_ome, if_dapi, reg_dir)

            self.memory_manager.clear_cache()

            # Step 3 – Full‑resolution (placeholder)
            print("\n### Step 3: Full‑resolution transform ###")
            full_reg = run_dir / "he_registered_full.ome.tiff"
            shutil.copy(reg_img, full_reg)

            # Step 4 – Tile extraction
            print("\n### Step 4: Tile extraction ###")
            tiles_dir = run_dir / "tiles"
            n_tiles = self.tiles.extract_paired_tiles(full_reg, if_dapi, tiles_dir)

            # Report
            report = {
                "pipeline_version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "input_files": {"he_qptiff": str(he_qptiff), "if_qptiff": str(if_qptiff)},
                "output_files": {
                    "registered_image": str(full_reg),
                    "transform": str(transform),
                    "tiles_directory": str(tiles_dir),
                },
                "statistics": {
                    "tiles_extracted": n_tiles,
                    "gpu_used": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                },
            }
            json.dump(report, (run_dir / "pipeline_report.json").open("w"), indent=2)

            print("\nPipeline complete!")
            print(f"Output: {run_dir}")
            return run_dir

        except Exception as exc:
            print(f"\nPipeline failed: {exc}")
            import traceback

            traceback.print_exc()
            (run_dir / "error_log.txt").write_text(f"{exc}\n\n{traceback.format_exc()}")
            return None


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="RegWSI pipeline (H&E → mIF)")
    parser.add_argument("--he_qptiff", required=True, help="Path to H&E QPTIFF")
    parser.add_argument("--if_qptiff", required=True, help="Path to IF QPTIFF")
    parser.add_argument("--output", default="/workspace/data/output", help="Output directory")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size (default 512)")
    parser.add_argument("--downsample_he", type=int, default=2, help="Downsample factor for H&E")

    args = parser.parse_args()

    cfg = Config()
    cfg.tile_params["tile_size"] = args.tile_size
    cfg.downsample_he = args.downsample_he

    pipeline = RegWSIPipeline(cfg)
    result = pipeline.run(args.he_qptiff, args.if_qptiff, args.output)
    print(f"\nSuccess! Results saved to: {result}") if result else print("\nPipeline failed!")
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
