#!/usr/bin/env python3\
"""\
Prepare RegWSI output for Pix2PixHD training\
Creates proper directory structure and performs final preprocessing\
"""\
\
import os\
import sys\
from pathlib import Path\
import cv2\
import numpy as np\
from tqdm import tqdm\
import json\
import shutil\
from sklearn.model_selection import train_test_split\
import argparse\
\
\
class Pix2PixHDPreparator:\
    """Prepare tiles for Pix2PixHD training"""\
    \
    def __init__(self, tile_size=512, train_ratio=0.8):\
        self.tile_size = tile_size\
        self.train_ratio = train_ratio\
        \
    def prepare_dataset(self, regwsi_output_dir, pix2pixhd_dir, dataset_name="he2if"):\
        """\
        Prepare dataset in Pix2PixHD format\
        \
        Expected structure:\
        datasets/\
        \uc0\u9492 \u9472 \u9472  he2if/\
            \uc0\u9500 \u9472 \u9472  train_A/    (H&E images)\
            \uc0\u9500 \u9472 \u9472  train_B/    (IF images)\
            \uc0\u9500 \u9472 \u9472  test_A/\
            \uc0\u9492 \u9472 \u9472  test_B/\
        """\
        regwsi_output_dir = Path(regwsi_output_dir)\
        pix2pixhd_dir = Path(pix2pixhd_dir)\
        \
        # Find the tiles directory\
        tiles_dir = regwsi_output_dir / "tiles"\
        if not tiles_dir.exists():\
            # Try to find in latest run\
            runs = sorted(regwsi_output_dir.glob("regwsi_run_*"))\
            if runs:\
                tiles_dir = runs[-1] / "tiles"\
        \
        if not tiles_dir.exists():\
            raise ValueError(f"Tiles directory not found in \{regwsi_output_dir\}")\
        \
        train_a_src = tiles_dir / "train_A"\
        train_b_src = tiles_dir / "train_B"\
        \
        if not train_a_src.exists() or not train_b_src.exists():\
            raise ValueError(f"Source tile directories not found in \{tiles_dir\}")\
        \
        # Create output directory structure\
        dataset_dir = pix2pixhd_dir / "datasets" / dataset_name\
        dataset_dir.mkdir(parents=True, exist_ok=True)\
        \
        train_a_dst = dataset_dir / "train_A"\
        train_b_dst = dataset_dir / "train_B"\
        test_a_dst = dataset_dir / "test_A"\
        test_b_dst = dataset_dir / "test_B"\
        \
        for dir_path in [train_a_dst, train_b_dst, test_a_dst, test_b_dst]:\
            dir_path.mkdir(exist_ok=True)\
        \
        # Get all tile pairs\
        tile_names = sorted([f.name for f in train_a_src.glob("*.png")])\
        print(f"Found \{len(tile_names)\} tile pairs")\
        \
        # Split into train/test\
        train_tiles, test_tiles = train_test_split(\
            tile_names, \
            train_size=self.train_ratio, \
            random_state=42\
        )\
        \
        print(f"Train tiles: \{len(train_tiles)\}")\
        print(f"Test tiles: \{len(test_tiles)\}")\
        \
        # Process and copy tiles\
        print("\\nProcessing training tiles...")\
        for tile_name in tqdm(train_tiles, desc="Train"):\
            self._process_tile_pair(\
                train_a_src / tile_name,\
                train_b_src / tile_name,\
                train_a_dst / tile_name,\
                train_b_dst / tile_name\
            )\
        \
        print("\\nProcessing test tiles...")\
        for tile_name in tqdm(test_tiles, desc="Test"):\
            self._process_tile_pair(\
                train_a_src / tile_name,\
                train_b_src / tile_name,\
                test_a_dst / tile_name,\
                test_b_dst / tile_name\
            )\
        \
        # Create dataset info\
        self._create_dataset_info(dataset_dir, len(train_tiles), len(test_tiles))\
        \
        # Create Pix2PixHD options file\
        self._create_options_file(pix2pixhd_dir, dataset_name)\
        \
        print(f"\\nDataset prepared successfully!")\
        print(f"Location: \{dataset_dir\}")\
        \
        return dataset_dir\
    \
    def _process_tile_pair(self, src_a, src_b, dst_a, dst_b):\
        """Process and save a tile pair with optional preprocessing"""\
        # Load images\
        img_a = cv2.imread(str(src_a))\
        img_b = cv2.imread(str(src_b))\
        \
        # Ensure correct size\
        if img_a.shape[:2] != (self.tile_size, self.tile_size):\
            img_a = cv2.resize(img_a, (self.tile_size, self.tile_size))\
        if img_b.shape[:2] != (self.tile_size, self.tile_size):\
            img_b = cv2.resize(img_b, (self.tile_size, self.tile_size))\
        \
        # Optional: Apply preprocessing\
        img_a = self._preprocess_he(img_a)\
        img_b = self._preprocess_if(img_b)\
        \
        # Save\
        cv2.imwrite(str(dst_a), img_a)\
        cv2.imwrite(str(dst_b), img_b)\
    \
    def _preprocess_he(self, img):\
        """Preprocess H&E image"""\
        # Optional: Apply stain normalization or other preprocessing\
        # For now, just ensure it's RGB\
        if img.ndim == 2:\
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)\
        return img\
    \
    def _preprocess_if(self, img):\
        """Preprocess IF image"""\
        # Optional: Enhance contrast for DAPI\
        if img.ndim == 2:\
            # Apply CLAHE to grayscale\
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))\
            img = clahe.apply(img)\
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)\
        return img\
    \
    def _create_dataset_info(self, dataset_dir, n_train, n_test):\
        """Create dataset information file"""\
        info = \{\
            "dataset_name": dataset_dir.name,\
            "tile_size": self.tile_size,\
            "n_train": n_train,\
            "n_test": n_test,\
            "train_ratio": self.train_ratio,\
            "modality_A": "H&E (20x downsampled to 10x)",\
            "modality_B": "IF DAPI (10x)",\
            "preprocessing": \{\
                "he": "RGB conversion",\
                "if": "CLAHE enhancement, RGB conversion"\
            \}\
        \}\
        \
        with open(dataset_dir / "dataset_info.json", 'w') as f:\
            json.dump(info, f, indent=2)\
    \
    def _create_options_file(self, pix2pixhd_dir, dataset_name):\
        """Create sample options file for Pix2PixHD training"""\
        options_dir = pix2pixhd_dir / "options"\
        options_dir.mkdir(exist_ok=True)\
        \
        train_options = f"""# Pix2PixHD training options for \{dataset_name\}\
# Usage: python train.py --name he2if_512 --dataroot ./datasets/\{dataset_name\} --gpu_ids 0\
\
--name he2if_512\
--dataroot ./datasets/\{dataset_name\}\
--checkpoints_dir ./checkpoints\
--gpu_ids 0\
\
# Model parameters\
--model pix2pixHD\
--netG global\
--ngf 64\
--niter 50\
--niter_decay 50\
\
# Input/output parameters\
--input_nc 3\
--output_nc 3\
--loadSize 512\
--fineSize 512\
--resize_or_crop none\
\
# Training parameters\
--batchSize 4\
--nThreads 8\
--save_latest_freq 1000\
--save_epoch_freq 5\
--print_freq 100\
--display_freq 100\
\
# Loss parameters\
--lambda_feat 10\
--no_vgg_loss\
\
# For high-res (1024x1024), uncomment:\
# --netG local\
# --loadSize 1024\
# --fineSize 1024\
# --batchSize 1\
"""\
        \
        with open(options_dir / f"train_\{dataset_name\}.txt", 'w') as f:\
            f.write(train_options)\
        \
        test_options = f"""# Pix2PixHD test options for \{dataset_name\}\
# Usage: python test.py --name he2if_512 --dataroot ./datasets/\{dataset_name\} --gpu_ids 0\
\
--name he2if_512\
--dataroot ./datasets/\{dataset_name\}\
--checkpoints_dir ./checkpoints\
--results_dir ./results\
--gpu_ids 0\
\
# Model parameters\
--model pix2pixHD\
--netG global\
--ngf 64\
\
# Input/output parameters  \
--input_nc 3\
--output_nc 3\
--loadSize 512\
--fineSize 512\
--resize_or_crop none\
\
# Test parameters\
--phase test\
--how_many 100\
--which_epoch latest\
"""\
        \
        with open(options_dir / f"test_\{dataset_name\}.txt", 'w') as f:\
            f.write(test_options)\
        \
        print(f"\\nCreated Pix2PixHD option files in: \{options_dir\}")\
\
\
def create_training_script(pix2pixhd_dir, dataset_name):\
    """Create a simple training script"""\
    script_content = f"""#!/bin/bash\
# Training script for Pix2PixHD on \{dataset_name\}\
\
# Clone Pix2PixHD if not present\
if [ ! -d "pix2pixHD" ]; then\
    git clone https://github.com/NVIDIA/pix2pixHD.git\
    cd pix2pixHD\
else\
    cd pix2pixHD\
fi\
\
# Install requirements\
pip install dominate\
\
# Copy dataset\
if [ ! -d "datasets/\{dataset_name\}" ]; then\
    cp -r ../datasets/\{dataset_name\} datasets/\
fi\
\
# Train model\
python train.py \\\\\
    --name he2if_512 \\\\\
    --dataroot ./datasets/\{dataset_name\} \\\\\
    --gpu_ids 0 \\\\\
    --batchSize 4 \\\\\
    --niter 50 \\\\\
    --niter_decay 50 \\\\\
    --save_epoch_freq 5\
\
echo "Training complete! Model saved in checkpoints/he2if_512/"\
"""\
    \
    script_path = pix2pixhd_dir / f"train_\{dataset_name\}.sh"\
    with open(script_path, 'w') as f:\
        f.write(script_content)\
    \
    os.chmod(script_path, 0o755)\
    print(f"Created training script: \{script_path\}")\
\
\
def main():\
    parser = argparse.ArgumentParser(\
        description='Prepare RegWSI output for Pix2PixHD training'\
    )\
    parser.add_argument(\
        '--regwsi_output',\
        type=str,\
        required=True,\
        help='Path to RegWSI output directory'\
    )\
    parser.add_argument(\
        '--pix2pixhd_dir',\
        type=str,\
        default='/workspace/pix2pixhd',\
        help='Pix2PixHD directory (default: /workspace/pix2pixhd)'\
    )\
    parser.add_argument(\
        '--dataset_name',\
        type=str,\
        default='he2if',\
        help='Dataset name (default: he2if)'\
    )\
    parser.add_argument(\
        '--tile_size',\
        type=int,\
        default=512,\
        help='Tile size (default: 512)'\
    )\
    parser.add_argument(\
        '--train_ratio',\
        type=float,\
        default=0.8,\
        help='Train/test split ratio (default: 0.8)'\
    )\
    \
    args = parser.parse_args()\
    \
    # Prepare dataset\
    preparator = Pix2PixHDPreparator(\
        tile_size=args.tile_size,\
        train_ratio=args.train_ratio\
    )\
    \
    dataset_dir = preparator.prepare_dataset(\
        args.regwsi_output,\
        args.pix2pixhd_dir,\
        args.dataset_name\
    )\
    \
    # Create training script\
    create_training_script(Path(args.pix2pixhd_dir), args.dataset_name)\
    \
    print("\\n" + "="*60)\
    print("Dataset preparation complete!")\
    print("="*60)\
    print(f"\\nNext steps:")\
    print(f"1. cd \{args.pix2pixhd_dir\}")\
    print(f"2. ./train_\{args.dataset_name\}.sh")\
    print(f"\\nOr manually:")\
    print(f"1. git clone https://github.com/NVIDIA/pix2pixHD.git")\
    print(f"2. cd pix2pixHD")\
    print(f"3. python train.py --name he2if_512 --dataroot ./datasets/\{args.dataset_name\}")\
\
\
if __name__ == "__main__":\
    main()}
