#!/usr/bin/env python3\
"""\
Complete RegWSI Pipeline for H&E to mIF Registration\
Optimized for NVIDIA H200 GPU with 140GB VRAM\
Handles QPTIFF to OME-TIFF conversion, registration, and tile extraction for Pix2PixHD\
"""\
\
import os\
import sys\
import subprocess\
import time\
import shutil\
from pathlib import Path\
import numpy as np\
import torch\
import cv2\
import tifffile\
import zarr\
from datetime import datetime\
from tqdm import tqdm\
import json\
import psutil\
import gc\
\
# Import registration frameworks\
try:\
    from deeperhistreg import deeperhistreg\
    from deeperhistreg.elastix import Elastix\
except ImportError:\
    print("Warning: DeeperHistReg not properly installed")\
    \
import SimpleITK as sitk\
from skimage import exposure, transform\
from skimage.metrics import structural_similarity as ssim\
import pyvips\
\
# Configuration\
class Config:\
    """Pipeline configuration optimized for H200 GPU"""\
    def __init__(self):\
        # Paths\
        self.BFCONVERT_PATH = "/workspace/regwsi/bftools/bfconvert"\
        self.WORK_DIR = Path("/workspace/data")\
        self.INPUT_DIR = self.WORK_DIR / "input"\
        self.OUTPUT_DIR = self.WORK_DIR / "output"\
        self.TILES_DIR = self.WORK_DIR / "tiles"\
        \
        # GPU settings for H200\
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\
        self.gpu_memory_limit = 120 * 1024**3  # Use 120GB of 140GB\
        self.tile_batch_size = 64  # Large batch size for H200\
        \
        # Registration parameters\
        self.registration_params = \{\
            'initial_alignment': \{\
                'scales': 8,\
                'rotation_angles': 12,\
                'use_superpoint': True,\
                'use_superglue': True\
            \},\
            'nonrigid': \{\
                'similarity_measure': 'ncc',\
                'regularization': 'diffusive',\
                'levels': 4,\
                'iterations': [100, 100, 50, 50],\
                'learning_rates': [0.1, 0.05, 0.01, 0.005],\
                'instance_optimization': True\
            \}\
        \}\
        \
        # Tile extraction parameters\
        self.tile_params = \{\
            'tile_size': 512,  # For Pix2PixHD\
            'overlap': 0,\
            'tissue_threshold': 0.5,\
            'quality_threshold': 0.8,\
            'min_tissue_area': 0.3\
        \}\
        \
        # Processing parameters\
        self.downsample_he = 2  # Downsample H&E from 20x to 10x\
        self.if_channel = 0  # DAPI channel (index 0)\
        self.num_workers = 16  # Parallel processing workers\
        \
\
class GPUMemoryManager:\
    """Manage GPU memory for H200"""\
    def __init__(self, device=None):\
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')\
        \
        if self.device.type == 'cuda':\
            self.gpu_properties = torch.cuda.get_device_properties(0)\
            self.total_memory = self.gpu_properties.total_memory\
            print(f"\\nGPU: \{self.gpu_properties.name\}")\
            print(f"Total GPU Memory: \{self.total_memory / 1024**3:.1f\} GB")\
            \
            # Set memory fraction\
            torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory\
    \
    def clear_cache(self):\
        if self.device.type == 'cuda':\
            torch.cuda.empty_cache()\
            torch.cuda.synchronize()\
            gc.collect()\
    \
    def get_memory_stats(self):\
        if self.device.type == 'cuda':\
            allocated = torch.cuda.memory_allocated() / 1024**3\
            reserved = torch.cuda.memory_reserved() / 1024**3\
            free = (self.total_memory - torch.cuda.memory_allocated()) / 1024**3\
            return \{\
                'allocated_gb': allocated,\
                'reserved_gb': reserved,\
                'free_gb': free\
            \}\
        return \{\}\
\
\
class ImagePreprocessor:\
    """Handle QPTIFF to OME-TIFF conversion and preprocessing"""\
    \
    def __init__(self, config):\
        self.config = config\
        self.memory_manager = GPUMemoryManager(config.device)\
    \
    def convert_qptiff_to_ometiff(self, input_path, output_path, downsample=1):\
        """Convert QPTIFF to pyramidal OME-TIFF"""\
        input_path = Path(input_path)\
        output_path = Path(output_path)\
        \
        print(f"\\nConverting \{input_path.name\} to OME-TIFF...")\
        \
        # Use bfconvert for initial conversion\
        if self.config.BFCONVERT_PATH and Path(self.config.BFCONVERT_PATH).exists():\
            cmd = f'"\{self.config.BFCONVERT_PATH\}" -compression LZW -pyramid-scale 2 "\{input_path\}" "\{output_path\}"'\
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)\
            \
            if result.returncode != 0:\
                print(f"Error in bfconvert: \{result.stderr\}")\
                return False\
        \
        # If downsampling is needed\
        if downsample > 1:\
            print(f"Downsampling by factor of \{downsample\}...")\
            downsampled_path = output_path.parent / f"\{output_path.stem\}_downsampled\{output_path.suffix\}"\
            \
            # Use pyvips for efficient downsampling\
            image = pyvips.Image.new_from_file(str(output_path))\
            downsampled = image.resize(1.0 / downsample)\
            downsampled.write_to_file(str(downsampled_path), compression='lzw', tile=True, \
                                     pyramid=True, tile_width=512, tile_height=512)\
            \
            return downsampled_path\
        \
        return output_path\
    \
    def extract_if_channel(self, input_path, output_path, channel_idx=0):\
        """Extract specific channel from multi-channel IF image"""\
        print(f"\\nExtracting channel \{channel_idx\} from IF image...")\
        \
        with tifffile.TiffFile(input_path) as tif:\
            # Read the image\
            data = tif.asarray()\
            \
            # Handle different dimension orders\
            if data.ndim == 4:  # Z, C, Y, X\
                channel_data = data[0, channel_idx, :, :]\
            elif data.ndim == 3:  # C, Y, X\
                channel_data = data[channel_idx, :, :]\
            else:\
                raise ValueError(f"Unexpected data shape: \{data.shape\}")\
            \
            # Save as OME-TIFF\
            metadata = \{\
                'axes': 'YX',\
                'Channel': \{'Name': 'DAPI'\},\
                'PhysicalSizeX': 1.0,\
                'PhysicalSizeY': 1.0,\
                'PhysicalSizeXUnit': '\'b5m',\
                'PhysicalSizeYUnit': '\'b5m'\
            \}\
            \
            tifffile.imwrite(\
                output_path,\
                channel_data,\
                photometric='minisblack',\
                compression='lzw',\
                tile=(512, 512),\
                metadata=metadata,\
                ome=True,\
                bigtiff=True\
            )\
        \
        return output_path\
    \
    def prepare_for_registration(self, image_path):\
        """Prepare image for registration (grayscale, normalized)"""\
        # Load image\
        if str(image_path).endswith(('.tif', '.tiff')):\
            image = tifffile.imread(image_path)\
        else:\
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)\
        \
        # Convert to grayscale if needed\
        if image.ndim == 3:\
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)\
        \
        # Normalize to [0, 1]\
        image = image.astype(np.float32)\
        image = (image - image.min()) / (image.max() - image.min())\
        \
        # Apply CLAHE for better contrast\
        image_uint8 = (image * 255).astype(np.uint8)\
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))\
        image_enhanced = clahe.apply(image_uint8)\
        \
        return image_enhanced.astype(np.float32) / 255.0\
\
\
class RegWSIRegistration:\
    """Main RegWSI registration using DeeperHistReg framework"""\
    \
    def __init__(self, config, memory_manager):\
        self.config = config\
        self.memory_manager = memory_manager\
        self.device = config.device\
        \
        # Initialize Elastix for nonrigid registration\
        self.elastix = Elastix()\
    \
    def register_deeperhistreg(self, source_path, target_path, output_dir):\
        """Perform registration using DeeperHistReg"""\
        output_dir = Path(output_dir)\
        output_dir.mkdir(exist_ok=True, parents=True)\
        \
        try:\
            # Load configuration\
            config_path = Path(__file__).parent / "configs" / "regwsi_config.yaml"\
            \
            # Create custom config if doesn't exist\
            if not config_path.exists():\
                config_path.parent.mkdir(exist_ok=True, parents=True)\
                config_data = \{\
                    'initial_alignment': \{\
                        'use_rotation': True,\
                        'rotation_step': 30,\
                        'use_flipping': True,\
                        'use_scale': True,\
                        'scale_factors': [0.5, 0.75, 1.0, 1.25, 1.5]\
                    \},\
                    'nonrigid_registration': \{\
                        'pyramid_levels': 4,\
                        'pyramid_schedule': [8, 4, 2, 1],\
                        'iterations': 100,\
                        'similarity_measure': 'ncc'\
                    \}\
                \}\
                import yaml\
                with open(config_path, 'w') as f:\
                    yaml.dump(config_data, f)\
            \
            # Run DeeperHistReg\
            displaced_path, displacement_path, registered_path = deeperhistreg.run_registration(\
                str(source_path),\
                str(target_path),\
                str(output_dir),\
                config_path=str(config_path) if config_path.exists() else None\
            )\
            \
            return registered_path, displacement_path\
            \
        except Exception as e:\
            print(f"DeeperHistReg failed: \{e\}")\
            print("Falling back to custom implementation...")\
            return self.register_custom(source_path, target_path, output_dir)\
    \
    def register_custom(self, source_path, target_path, output_dir):\
        """Custom RegWSI implementation"""\
        output_dir = Path(output_dir)\
        output_dir.mkdir(exist_ok=True, parents=True)\
        \
        print("\\n=== RegWSI Registration Pipeline ===")\
        \
        # Step 1: Load and preprocess images\
        print("\\n1. Loading and preprocessing images...")\
        preprocessor = ImagePreprocessor(self.config)\
        \
        source_img = preprocessor.prepare_for_registration(source_path)\
        target_img = preprocessor.prepare_for_registration(target_path)\
        \
        # Save preprocessed images\
        preprocessed_dir = output_dir / "preprocessed"\
        preprocessed_dir.mkdir(exist_ok=True)\
        \
        cv2.imwrite(str(preprocessed_dir / "source_preprocessed.png"), \
                   (source_img * 255).astype(np.uint8))\
        cv2.imwrite(str(preprocessed_dir / "target_preprocessed.png"), \
                   (target_img * 255).astype(np.uint8))\
        \
        # Step 2: Initial alignment (rotation and translation)\
        print("\\n2. Performing initial alignment...")\
        initial_transform = self._initial_alignment(source_img, target_img)\
        \
        # Apply initial transformation\
        h, w = target_img.shape\
        source_aligned = cv2.warpAffine(source_img, initial_transform, (w, h))\
        \
        cv2.imwrite(str(output_dir / "initial_alignment.png"), \
                   (source_aligned * 255).astype(np.uint8))\
        \
        # Step 3: Nonrigid registration\
        print("\\n3. Performing nonrigid registration...")\
        \
        # Convert to SimpleITK images\
        source_sitk = sitk.GetImageFromArray(source_aligned)\
        target_sitk = sitk.GetImageFromArray(target_img)\
        \
        # Configure Elastix parameters\
        parameter_map = sitk.GetDefaultParameterMap('bspline')\
        parameter_map['Metric'] = ['AdvancedNormalizedCorrelation']\
        parameter_map['NumberOfResolutions'] = ['4']\
        parameter_map['MaximumNumberOfIterations'] = ['100']\
        parameter_map['FinalBSplineInterpolationOrder'] = ['3']\
        \
        # Set up Elastix\
        elastix_filter = sitk.ElastixImageFilter()\
        elastix_filter.SetFixedImage(target_sitk)\
        elastix_filter.SetMovingImage(source_sitk)\
        elastix_filter.SetParameterMap(parameter_map)\
        \
        # Execute registration\
        elastix_filter.Execute()\
        \
        # Get results\
        result_image = elastix_filter.GetResultImage()\
        transform_parameters = elastix_filter.GetTransformParameterMap()\
        \
        # Convert back to numpy\
        registered_img = sitk.GetArrayFromImage(result_image)\
        \
        # Save results\
        registered_path = output_dir / "registered_source.tiff"\
        tifffile.imwrite(\
            registered_path,\
            (registered_img * 255).astype(np.uint8),\
            photometric='minisblack',\
            compression='lzw'\
        )\
        \
        # Save transform\
        transform_path = output_dir / "transform_parameters.txt"\
        sitk.WriteParameterFile(transform_parameters[0], str(transform_path))\
        \
        # Calculate quality metrics\
        print("\\n4. Calculating registration quality...")\
        mse = np.mean((target_img - registered_img) ** 2)\
        ssim_score = ssim(target_img, registered_img)\
        \
        print(f"   MSE: \{mse:.6f\}")\
        print(f"   SSIM: \{ssim_score:.4f\}")\
        \
        # Save registration report\
        report = \{\
            'timestamp': datetime.now().isoformat(),\
            'source': str(source_path),\
            'target': str(target_path),\
            'initial_transform': initial_transform.tolist(),\
            'quality_metrics': \{\
                'mse': float(mse),\
                'ssim': float(ssim_score)\
            \}\
        \}\
        \
        with open(output_dir / "registration_report.json", 'w') as f:\
            json.dump(report, f, indent=2)\
        \
        return registered_path, transform_path\
    \
    def _initial_alignment(self, source, target):\
        """Perform initial alignment using feature matching"""\
        # Detect features using SIFT\
        sift = cv2.SIFT_create(nfeatures=10000)\
        \
        kp1, desc1 = sift.detectAndCompute(\
            (source * 255).astype(np.uint8), None\
        )\
        kp2, desc2 = sift.detectAndCompute(\
            (target * 255).astype(np.uint8), None\
        )\
        \
        # Match features\
        matcher = cv2.BFMatcher()\
        matches = matcher.knnMatch(desc1, desc2, k=2)\
        \
        # Apply Lowe's ratio test\
        good_matches = []\
        for m, n in matches:\
            if m.distance < 0.7 * n.distance:\
                good_matches.append(m)\
        \
        print(f"   Found \{len(good_matches)\} good matches")\
        \
        if len(good_matches) >= 4:\
            # Extract matched points\
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])\
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])\
            \
            # Find transformation\
            M, mask = cv2.estimateAffinePartial2D(\
                src_pts, dst_pts, cv2.RANSAC, 5.0\
            )\
            \
            return M\
        else:\
            # Return identity transform if not enough matches\
            return np.eye(2, 3, dtype=np.float32)\
    \
    def apply_transform_to_wsi(self, wsi_path, transform_path, output_path):\
        """Apply registration transform to full WSI"""\
        print("\\n5. Applying transform to full resolution WSI...")\
        \
        # This would use the transform parameters to warp the entire WSI\
        # For now, we'll use the registered image directly\
        \
        # In production, you would:\
        # 1. Load the transform parameters\
        # 2. Apply to each tile of the WSI\
        # 3. Save as pyramidal OME-TIFF\
        \
        return output_path\
\
\
class TileExtractor:\
    """Extract tiles for Pix2PixHD training"""\
    \
    def __init__(self, config, memory_manager):\
        self.config = config\
        self.memory_manager = memory_manager\
        self.device = config.device\
    \
    def extract_paired_tiles(self, registered_source_path, target_path, output_dir):\
        """Extract corresponding tiles from registered images"""\
        output_dir = Path(output_dir)\
        train_a_dir = output_dir / "train_A"  # H&E tiles\
        train_b_dir = output_dir / "train_B"  # IF tiles\
        \
        train_a_dir.mkdir(parents=True, exist_ok=True)\
        train_b_dir.mkdir(parents=True, exist_ok=True)\
        \
        print(f"\\nExtracting \{self.config.tile_params['tile_size']\}x\{self.config.tile_params['tile_size']\} tiles...")\
        \
        # Load images\
        source_img = tifffile.imread(registered_source_path)\
        target_img = tifffile.imread(target_path)\
        \
        # Ensure same dimensions\
        if source_img.shape != target_img.shape:\
            print(f"Resizing source from \{source_img.shape\} to \{target_img.shape\}")\
            source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))\
        \
        h, w = source_img.shape[:2]\
        tile_size = self.config.tile_params['tile_size']\
        \
        # Calculate number of tiles\
        n_tiles_h = h // tile_size\
        n_tiles_w = w // tile_size\
        total_tiles = n_tiles_h * n_tiles_w\
        \
        print(f"Image size: \{w\}x\{h\}")\
        print(f"Extracting \{total_tiles\} potential tiles (\{n_tiles_w\}x\{n_tiles_h\} grid)")\
        \
        valid_tiles = 0\
        \
        with tqdm(total=total_tiles, desc="Extracting tiles") as pbar:\
            for i in range(n_tiles_h):\
                for j in range(n_tiles_w):\
                    y = i * tile_size\
                    x = j * tile_size\
                    \
                    # Extract tiles\
                    source_tile = source_img[y:y+tile_size, x:x+tile_size]\
                    target_tile = target_img[y:y+tile_size, x:x+tile_size]\
                    \
                    # Quality check\
                    if self._is_valid_tile(source_tile):\
                        # Save tiles\
                        tile_name = f"tile_\{i:04d\}_\{j:04d\}.png"\
                        \
                        # Convert to RGB if needed\
                        if source_tile.ndim == 2:\
                            source_tile_rgb = cv2.cvtColor(source_tile, cv2.COLOR_GRAY2RGB)\
                        else:\
                            source_tile_rgb = source_tile\
                        \
                        if target_tile.ndim == 2:\
                            target_tile_rgb = cv2.cvtColor(target_tile, cv2.COLOR_GRAY2RGB)\
                        else:\
                            target_tile_rgb = target_tile\
                        \
                        cv2.imwrite(str(train_a_dir / tile_name), source_tile_rgb)\
                        cv2.imwrite(str(train_b_dir / tile_name), target_tile_rgb)\
                        \
                        valid_tiles += 1\
                    \
                    pbar.update(1)\
        \
        print(f"\\nExtracted \{valid_tiles\} valid tiles (\{valid_tiles/total_tiles*100:.1f\}% of total)")\
        \
        # Create dataset info\
        dataset_info = \{\
            'source_image': str(registered_source_path),\
            'target_image': str(target_path),\
            'tile_size': tile_size,\
            'total_tiles': total_tiles,\
            'valid_tiles': valid_tiles,\
            'timestamp': datetime.now().isoformat()\
        \}\
        \
        with open(output_dir / "dataset_info.json", 'w') as f:\
            json.dump(dataset_info, f, indent=2)\
        \
        return valid_tiles\
    \
    def _is_valid_tile(self, tile):\
        """Check if tile meets quality criteria"""\
        # Convert to grayscale if needed\
        if tile.ndim == 3:\
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)\
        else:\
            gray = tile\
        \
        # Check tissue content\
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\
        tissue_ratio = np.sum(binary > 0) / binary.size\
        \
        if tissue_ratio < self.config.tile_params['min_tissue_area']:\
            return False\
        \
        # Check variance (avoid blank tiles)\
        if np.var(gray) < 0.01:\
            return False\
        \
        # Check for artifacts (very bright or dark areas)\
        if np.mean(gray) < 10 or np.mean(gray) > 245:\
            return False\
        \
        return True\
\
\
class RegWSIPipeline:\
    """Main pipeline orchestrator"""\
    \
    def __init__(self, config=None):\
        self.config = config or Config()\
        self.memory_manager = GPUMemoryManager(self.config.device)\
        self.preprocessor = ImagePreprocessor(self.config)\
        self.registration = RegWSIRegistration(self.config, self.memory_manager)\
        self.tile_extractor = TileExtractor(self.config, self.memory_manager)\
    \
    def run(self, he_qptiff_path, if_qptiff_path, output_base_dir):\
        """Run complete pipeline"""\
        output_base_dir = Path(output_base_dir)\
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\
        run_dir = output_base_dir / f"regwsi_run_\{timestamp\}"\
        run_dir.mkdir(parents=True, exist_ok=True)\
        \
        print("\\n" + "="*80)\
        print(f"RegWSI Pipeline - Started at \{datetime.now().strftime('%Y-%m-%d %H:%M:%S')\}")\
        print("="*80)\
        \
        # Log configuration\
        log_file = run_dir / "pipeline_log.txt"\
        \
        try:\
            # Step 1: Convert and prepare images\
            print("\\n### Step 1: Image Preparation ###")\
            \
            # Convert H&E QPTIFF to OME-TIFF and downsample\
            he_ometiff = run_dir / "he_10x.ome.tiff"\
            he_converted = self.preprocessor.convert_qptiff_to_ometiff(\
                he_qptiff_path, he_ometiff, downsample=self.config.downsample_he\
            )\
            \
            # Extract DAPI channel from IF\
            if_dapi = run_dir / "if_dapi.ome.tiff"\
            if_channel = self.preprocessor.extract_if_channel(\
                if_qptiff_path, if_dapi, channel_idx=self.config.if_channel\
            )\
            \
            self.memory_manager.clear_cache()\
            \
            # Step 2: Registration\
            print("\\n### Step 2: Registration ###")\
            registration_dir = run_dir / "registration"\
            \
            registered_path, transform_path = self.registration.register_deeperhistreg(\
                he_converted, if_channel, registration_dir\
            )\
            \
            self.memory_manager.clear_cache()\
            \
            # Step 3: Apply to full resolution\
            print("\\n### Step 3: Full Resolution Transform ###")\
            full_registered = run_dir / "he_registered_full.ome.tiff"\
            \
            # For now, we'll use the registered result directly\
            # In production, apply transform to original resolution\
            shutil.copy(registered_path, full_registered)\
            \
            # Step 4: Tile extraction\
            print("\\n### Step 4: Tile Extraction for Pix2PixHD ###")\
            tiles_dir = run_dir / "tiles"\
            \
            n_tiles = self.tile_extractor.extract_paired_tiles(\
                full_registered, if_channel, tiles_dir\
            )\
            \
            # Final report\
            print("\\n### Pipeline Complete ###")\
            print(f"Output directory: \{run_dir\}")\
            print(f"Registered image: \{full_registered\}")\
            print(f"Tiles extracted: \{n_tiles\}")\
            print(f"Ready for Pix2PixHD training: \{tiles_dir\}")\
            \
            # Save final report\
            final_report = \{\
                'pipeline_version': '1.0',\
                'timestamp': datetime.now().isoformat(),\
                'input_files': \{\
                    'he_qptiff': str(he_qptiff_path),\
                    'if_qptiff': str(if_qptiff_path)\
                \},\
                'output_files': \{\
                    'registered_image': str(full_registered),\
                    'transform': str(transform_path),\
                    'tiles_directory': str(tiles_dir)\
                \},\
                'statistics': \{\
                    'tiles_extracted': n_tiles,\
                    'gpu_used': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'\
                \}\
            \}\
            \
            with open(run_dir / "pipeline_report.json", 'w') as f:\
                json.dump(final_report, f, indent=2)\
            \
            return run_dir\
            \
        except Exception as e:\
            print(f"\\nPipeline failed with error: \{e\}")\
            import traceback\
            traceback.print_exc()\
            \
            # Save error log\
            with open(run_dir / "error_log.txt", 'w') as f:\
                f.write(f"Error: \{str(e)\}\\n\\n")\
                f.write(traceback.format_exc())\
            \
            return None\
\
\
def main():\
    """Main entry point"""\
    import argparse\
    \
    parser = argparse.ArgumentParser(\
        description='RegWSI Pipeline for H&E to mIF Registration'\
    )\
    parser.add_argument(\
        '--he_qptiff', \
        type=str, \
        required=True,\
        help='Path to H&E QPTIFF file'\
    )\
    parser.add_argument(\
        '--if_qptiff', \
        type=str, \
        required=True,\
        help='Path to IF QPTIFF file'\
    )\
    parser.add_argument(\
        '--output', \
        type=str, \
        default='/workspace/data/output',\
        help='Output directory'\
    )\
    parser.add_argument(\
        '--tile_size', \
        type=int, \
        default=512,\
        help='Tile size for Pix2PixHD (default: 512)'\
    )\
    parser.add_argument(\
        '--downsample_he', \
        type=int, \
        default=2,\
        help='Downsample factor for H&E (default: 2)'\
    )\
    \
    args = parser.parse_args()\
    \
    # Create custom config\
    config = Config()\
    config.tile_params['tile_size'] = args.tile_size\
    config.downsample_he = args.downsample_he\
    \
    # Run pipeline\
    pipeline = RegWSIPipeline(config)\
    result = pipeline.run(\
        args.he_qptiff,\
        args.if_qptiff,\
        args.output\
    )\
    \
    if result:\
        print(f"\\nSuccess! Results saved to: \{result\}")\
        return 0\
    else:\
        print("\\nPipeline failed!")\
        return 1\
\
\
if __name__ == "__main__":\
    sys.exit(main())}
