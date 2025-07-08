#!/usr/bin/env python3
"""
Prepare RegWSI output for Pix2PixHD training.

The script takes the tiles produced by the RegWSI pipeline and
re‑organises them into the directory layout expected by NVIDIA’s
Pix2PixHD implementation.  It also creates convenience option files and
a simple training shell‑script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


# ----------------------------------------------------------------------------- #
# Helper class
# ----------------------------------------------------------------------------- #
class Pix2PixHDPreparator:
    """Prepare paired tiles for Pix2PixHD training."""

    def __init__(self, *, tile_size: int = 512, train_ratio: float = 0.8) -> None:
        self.tile_size = tile_size
        self.train_ratio = train_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def prepare_dataset(
        self,
        regwsi_output_dir: str | Path,
        pix2pixhd_dir: str | Path,
        dataset_name: str = "he2if",
    ) -> Path:
        """
        Convert RegWSI tiles → Pix2PixHD dataset.

        Expected Pix2PixHD layout::

            datasets/
            └── he2if/
                ├── train_A/   (H&E)
                ├── train_B/   (IF)
                ├── test_A/
                └── test_B/
        """
        regwsi_output_dir = Path(regwsi_output_dir)
        pix2pixhd_dir = Path(pix2pixhd_dir)

        # Locate tiles directory (either given or inside latest run)
        tiles_dir = regwsi_output_dir / "tiles"
        if not tiles_dir.exists():
            runs = sorted(regwsi_output_dir.glob("regwsi_run_*"))
            if runs:
                tiles_dir = runs[-1] / "tiles"
        if not tiles_dir.exists():
            raise FileNotFoundError(f"No tiles directory found in {regwsi_output_dir}")

        src_a = tiles_dir / "train_A"
        src_b = tiles_dir / "train_B"
        if not src_a.exists() or not src_b.exists():
            raise FileNotFoundError(f"Source tile folders missing in {tiles_dir}")

        # ------------------------------------------------------------------ #
        # Create destination structure
        # ------------------------------------------------------------------ #
        ds_dir = pix2pixhd_dir / "datasets" / dataset_name
        (ds_dir / "train_A").mkdir(parents=True, exist_ok=True)
        (ds_dir / "train_B").mkdir(exist_ok=True)
        (ds_dir / "test_A").mkdir(exist_ok=True)
        (ds_dir / "test_B").mkdir(exist_ok=True)

        # ------------------------------------------------------------------ #
        # Train / test split
        # ------------------------------------------------------------------ #
        tile_names = sorted(f.name for f in src_a.glob("*.png"))
        print(f"Found {len(tile_names)} tile pairs")

        train_tiles, test_tiles = train_test_split(
            tile_names, train_size=self.train_ratio, random_state=42
        )
        print(f"Train tiles: {len(train_tiles)}")
        print(f"Test  tiles: {len(test_tiles)}")

        # ------------------------------------------------------------------ #
        # Copy & preprocess
        # ------------------------------------------------------------------ #
        print("\nProcessing training tiles …")
        for t in tqdm(train_tiles, desc="train"):
            self._process_pair(
                src_a / t, src_b / t, ds_dir / "train_A" / t, ds_dir / "train_B" / t
            )

        print("\nProcessing test tiles …")
        for t in tqdm(test_tiles, desc="test"):
            self._process_pair(
                src_a / t, src_b / t, ds_dir / "test_A" / t, ds_dir / "test_B" / t
            )

        # Metadata + option files
        self._write_dataset_info(ds_dir, len(train_tiles), len(test_tiles))
        self._write_pix2pixhd_options(pix2pixhd_dir, dataset_name)

        print("\nDataset prepared:")
        print(ds_dir)
        return ds_dir

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _process_pair(self, src_a: Path, src_b: Path, dst_a: Path, dst_b: Path) -> None:
        """Load, resize / preprocess, save."""
        img_a = cv2.imread(str(src_a))
        img_b = cv2.imread(str(src_b))

        img_a = cv2.resize(img_a, (self.tile_size, self.tile_size))
        img_b = cv2.resize(img_b, (self.tile_size, self.tile_size))

        cv2.imwrite(str(dst_a), self._preprocess_he(img_a))
        cv2.imwrite(str(dst_b), self._preprocess_if(img_b))

    @staticmethod
    def _preprocess_he(img: np.ndarray) -> np.ndarray:
        """Placeholder H&E preprocessing (RGB ensure)."""
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img

    @staticmethod
    def _preprocess_if(img: np.ndarray) -> np.ndarray:
        """Enhance IF‑DAPI channel (CLAHE)."""
        if img.ndim == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    # ------------------------------------------------------------------ #
    # Metadata / option files
    # ------------------------------------------------------------------ #
    def _write_dataset_info(self, ds_dir: Path, n_train: int, n_test: int) -> None:
        info = {
            "dataset_name": ds_dir.name,
            "tile_size": self.tile_size,
            "n_train": n_train,
            "n_test": n_test,
            "train_ratio": self.train_ratio,
            "modality_A": "H&E (10×)",
            "modality_B": "IF DAPI (10×)",
            "preprocessing": {
                "he": "RGB conversion",
                "if": "CLAHE, RGB conversion",
            },
        }
        (ds_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

    def _write_pix2pixhd_options(self, root: Path, dataset: str) -> None:
        opt_dir = root / "options"
        opt_dir.mkdir(parents=True, exist_ok=True)

        train_txt = f"""\
# Pix2PixHD training options for {dataset}
# usage: python train.py --name he2if_512 --dataroot ./datasets/{dataset} --gpu_ids 0

--name          he2if_512
--dataroot      ./datasets/{dataset}
--checkpoints_dir ./checkpoints
--gpu_ids       0

# model
--model         pix2pixHD
--netG          global
--ngf           64

# data
--input_nc      3
--output_nc     3
--loadSize      512
--fineSize      512
--resize_or_crop none

# training
--batchSize     4
--niter         50
--niter_decay   50
--save_epoch_freq 5
"""

        test_txt = f"""\
# Pix2PixHD test options for {dataset}
# usage: python test.py --name he2if_512 --dataroot ./datasets/{dataset} --gpu_ids 0

--name          he2if_512
--dataroot      ./datasets/{dataset}
--checkpoints_dir ./checkpoints
--results_dir   ./results
--gpu_ids       0

# model
--model         pix2pixHD
--netG          global
--ngf           64

# data
--input_nc      3
--output_nc     3
--loadSize      512
--fineSize      512
--resize_or_crop none

# inference
--phase         test
--how_many      100
--which_epoch   latest
"""

        (opt_dir / f"train_{dataset}.txt").write_text(train_txt)
        (opt_dir / f"test_{dataset}.txt").write_text(test_txt)
        print(f"\nPix2PixHD option files → {opt_dir}")

        self._create_training_script(root, dataset)

    @staticmethod
    def _create_training_script(root: Path, dataset: str) -> None:
        """Convenience shell script for starting training."""
        script = f"""#!/bin/bash
# Train Pix2PixHD on {dataset}

set -e
[ ! -d pix2pixHD ] && git clone https://github.com/NVIDIA/pix2pixHD.git
cd pix2pixHD

pip install dominate  # Pix2PixHD requirement

# copy dataset (if outside repo)
[ ! -d datasets/{dataset} ] && cp -r ../datasets/{dataset} datasets/

python train.py \\
  --name he2if_512 \\
  --dataroot ./datasets/{dataset} \\
  --gpu_ids 0 \\
  --batchSize 4 \\
  --niter 50 \\
  --niter_decay 50 \\
  --save_epoch_freq 5

echo "Training complete → checkpoints/he2if_512"
"""
        path = root / f"train_{dataset}.sh"
        path.write_text(script)
        path.chmod(0o755)
        print(f"Training script created: {path}")


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RegWSI tiles for Pix2PixHD.")
    parser.add_argument("--regwsi_output", required=True, help="RegWSI output directory")
    parser.add_argument("--pix2pixhd_dir", default="/workspace/pix2pixhd", help="Pix2PixHD root")
    parser.add_argument("--dataset_name", default="he2if", help="Dataset name")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train / test split ratio")

    args = parser.parse_args()
    prep = Pix2PixHDPreparator(tile_size=args.tile_size, train_ratio=args.train_ratio)
    prep.prepare_dataset(args.regwsi_output, args.pix2pixhd_dir, args.dataset_name)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  cd {args.pix2pixhd_dir}")
    print(f"  ./train_{args.dataset_name}.sh\n")


if __name__ == "__main__":
    sys.exit(main())
