{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww12720\viewh7240\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/bin/bash\
# Setup script for RegWSI on vast.ai Ubuntu container\
# Optimized for H200 GPU with CUDA 12.4\
\
echo "==================================================================================="\
echo "RegWSI Setup for H&E to mIF Registration on vast.ai"\
echo "==================================================================================="\
echo "Started at: $(date)"\
echo "==================================================================================="\
\
# System update and essential packages\
echo "Installing system dependencies..."\
apt-get update && apt-get install -y \\\
    python3.10 python3.10-dev python3-pip \\\
    build-essential cmake git wget \\\
    libvips-dev libvips-tools \\\
    libopenslide-dev \\\
    libopenjp2-7-dev \\\
    libtiff-dev \\\
    libxml2-dev libxslt1-dev \\\
    openjdk-11-jdk \\\
    ffmpeg libsm6 libxext6 \\\
    curl unzip\
\
# Create working directory\
mkdir -p /workspace/regwsi\
cd /workspace/regwsi\
\
# Install Bio-Formats tools for QPTIFF conversion\
echo "Installing Bio-Formats tools..."\
wget https://downloads.openmicroscopy.org/bio-formats/7.0.1/artifacts/bftools.zip\
unzip bftools.zip\
rm bftools.zip\
export PATH="/workspace/regwsi/bftools:$PATH"\
\
# Create Python virtual environment\
echo "Creating Python environment..."\
python3.10 -m venv regwsi_env\
source regwsi_env/bin/activate\
\
# Upgrade pip\
pip install --upgrade pip setuptools wheel\
\
# Install CUDA-specific PyTorch for H200\
echo "Installing PyTorch with CUDA 12.4 support..."\
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121\
\
# Install core dependencies\
echo "Installing core dependencies..."\
pip install numpy==1.24.3 \\\
    scipy>=1.10.0 \\\
    scikit-image>=0.19.3 \\\
    SimpleITK>=2.2.1 \\\
    opencv-python>=4.7.0 \\\
    matplotlib>=3.5.0 \\\
    tifffile>=2023.7.10 \\\
    zarr>=2.14.2 \\\
    imagecodecs>=2023.1.23 \\\
    Pillow>=9.4.0 \\\
    tqdm>=4.65.0 \\\
    psutil>=5.9.0\
\
# Install registration frameworks\
echo "Installing registration frameworks..."\
pip install deeperhistreg\
pip install pyvips\
pip install openslide-python\
\
# Install for OME-TIFF support\
pip install ome-types python-bioformats\
\
# Install for GPU acceleration\
pip install cupy-cuda12x\
pip install cucim\
\
# Install for tile extraction and quality control\
pip install slideflow\
pip install histolab\
\
# Clone DeeperHistReg for models\
echo "Setting up DeeperHistReg models..."\
cd /workspace/regwsi\
git clone https://github.com/MWod/DeeperHistReg.git\
cd DeeperHistReg\
\
# Download pretrained models (you'll need to get these from the Google Drive link)\
echo "IMPORTANT: Download models from https://drive.google.com/drive/folders/1rZca3fKvPLGhoNvaAXJFpfgYYJ9LdKVZ"\
echo "Place them in: /workspace/regwsi/DeeperHistReg/deeperhistreg/"\
\
# Create directories for processing\
mkdir -p /workspace/data/\{input,output,tiles,models\}\
\
# Set environment variables\
echo "Setting environment variables..."\
cat >> ~/.bashrc << 'EOF'\
export CUDA_VISIBLE_DEVICES=0\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512\
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64\
export PATH="/workspace/regwsi/bftools:$PATH"\
source /workspace/regwsi/regwsi_env/bin/activate\
EOF\
\
# Test GPU availability\
echo "Testing GPU setup..."\
python3 -c "\
import torch\
import cv2\
print(f'PyTorch version: \{torch.__version__\}')\
print(f'CUDA available: \{torch.cuda.is_available()\}')\
if torch.cuda.is_available():\
    print(f'GPU: \{torch.cuda.get_device_name(0)\}')\
    print(f'GPU Memory: \{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f\} GB')\
print(f'OpenCV version: \{cv2.__version__\}')\
"\
\
echo "==================================================================================="\
echo "Setup complete! Activate environment with: source /workspace/regwsi/regwsi_env/bin/activate"\
echo "==================================================================================="}