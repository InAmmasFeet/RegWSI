#!/usr/bin/env python3\
"""\
Diagnostic script for RegWSI pipeline\
Checks system configuration and dependencies\
"""\
\
import sys\
import os\
import subprocess\
from pathlib import Path\
\
\
def print_header(title):\
    print(f"\\n\{'='*60\}")\
    print(f" \{title\}")\
    print(f"\{'='*60\}")\
\
\
def check_system():\
    """Check system specifications"""\
    print_header("System Information")\
    \
    # CPU info\
    try:\
        cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()\
        print(f"CPU: \{cpu_info.split(':')[1].strip()\}")\
    except:\
        print("CPU: Unable to detect")\
    \
    # Memory info\
    try:\
        mem_info = subprocess.check_output("free -h | grep Mem", shell=True).decode().strip().split()\
        print(f"RAM: \{mem_info[1]\} total, \{mem_info[2]\} used, \{mem_info[3]\} free")\
    except:\
        print("RAM: Unable to detect")\
    \
    # Disk info\
    try:\
        disk_info = subprocess.check_output("df -h /workspace | tail -1", shell=True).decode().strip().split()\
        print(f"Disk: \{disk_info[1]\} total, \{disk_info[2]\} used, \{disk_info[3]\} available")\
    except:\
        print("Disk: Unable to detect")\
\
\
def check_gpu():\
    """Check GPU configuration"""\
    print_header("GPU Information")\
    \
    try:\
        import torch\
        print(f"PyTorch version: \{torch.__version__\}")\
        print(f"CUDA available: \{torch.cuda.is_available()\}")\
        \
        if torch.cuda.is_available():\
            print(f"CUDA version: \{torch.version.cuda\}")\
            print(f"GPU count: \{torch.cuda.device_count()\}")\
            for i in range(torch.cuda.device_count()):\
                props = torch.cuda.get_device_properties(i)\
                print(f"\\nGPU \{i\}: \{props.name\}")\
                print(f"  Memory: \{props.total_memory / 1024**3:.1f\} GB")\
                print(f"  Compute capability: \{props.major\}.\{props.minor\}")\
                \
                # Current memory usage\
                allocated = torch.cuda.memory_allocated(i) / 1024**3\
                reserved = torch.cuda.memory_reserved(i) / 1024**3\
                print(f"  Currently allocated: \{allocated:.1f\} GB")\
                print(f"  Currently reserved: \{reserved:.1f\} GB")\
        else:\
            print("No CUDA devices available!")\
            \
    except ImportError:\
        print("PyTorch not installed!")\
    except Exception as e:\
        print(f"Error checking GPU: \{e\}")\
    \
    # Also check nvidia-smi\
    try:\
        print("\\nnvidia-smi output:")\
        subprocess.call(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu", \
                        "--format=csv,noheader"])\
    except:\
        print("nvidia-smi not available")\
\
\
def check_dependencies():\
    """Check Python dependencies"""\
    print_header("Python Dependencies")\
    \
    required_packages = \{\
        'numpy': '1.24.0',\
        'scipy': '1.10.0',\
        'scikit-image': '0.19.3',\
        'SimpleITK': '2.2.1',\
        'opencv-python': '4.7.0',\
        'tifffile': '2023.7.10',\
        'torch': '2.0.0',\
        'deeperhistreg': None,\
        'pyvips': None,\
        'cupy': None,\
        'cucim': None\
    \}\
    \
    for package, min_version in required_packages.items():\
        try:\
            module = __import__(package.replace('-', '_'))\
            version = getattr(module, '__version__', 'unknown')\
            status = "\uc0\u10003 "\
            \
            # Version check if specified\
            if min_version and version != 'unknown':\
                from packaging import version as pkg_version\
                if pkg_version.parse(version) < pkg_version.parse(min_version):\
                    status = f"\uc0\u9888  (version \{version\} < \{min_version\})"\
                    \
            print(f"\{package:20\} \{status:10\} \{version\}")\
            \
        except ImportError:\
            print(f"\{package:20\} \uc0\u10007  Not installed")\
\
\
def check_tools():\
    """Check external tools"""\
    print_header("External Tools")\
    \
    tools = \{\
        'bfconvert': '/workspace/regwsi/bftools/bfconvert',\
        'java': 'java',\
        'python3': 'python3',\
        'git': 'git'\
    \}\
    \
    for tool, path in tools.items():\
        try:\
            if tool == 'java':\
                result = subprocess.check_output([path, '-version'], stderr=subprocess.STDOUT).decode()\
                version = result.split('\\n')[0]\
                print(f"\{tool:20\} \uc0\u10003  \{version\}")\
            elif tool == 'bfconvert':\
                if Path(path).exists():\
                    print(f"\{tool:20\} \uc0\u10003  Found at \{path\}")\
                else:\
                    print(f"\{tool:20\} \uc0\u10007  Not found at \{path\}")\
            else:\
                result = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT).decode()\
                print(f"\{tool:20\} \uc0\u10003  \{result.strip()\}")\
        except:\
            print(f"\{tool:20\} \uc0\u10007  Not found")\
\
\
def check_models():\
    """Check for DeeperHistReg models"""\
    print_header("Model Files")\
    \
    model_dir = Path("/workspace/regwsi/DeeperHistReg/deeperhistreg")\
    \
    if model_dir.exists():\
        print(f"Model directory exists: \{model_dir\}")\
        \
        # Check for specific model files\
        expected_models = [\
            "superpoint.pth",\
            "superglue.pth",\
            "initial_alignment.pth"\
        ]\
        \
        for model in expected_models:\
            model_path = model_dir / model\
            if model_path.exists():\
                size = model_path.stat().st_size / 1024**2\
                print(f"  \{model:30\} \uc0\u10003  (\{size:.1f\} MB)")\
            else:\
                print(f"  \{model:30\} \uc0\u10007  Not found")\
    else:\
        print(f"Model directory not found: \{model_dir\}")\
        print("Please download models from:")\
        print("https://drive.google.com/drive/folders/1rZca3fKvPLGhoNvaAXJFpfgYYJ9LdKVZ")\
\
\
def test_image_loading():\
    """Test image loading capabilities"""\
    print_header("Image Format Support")\
    \
    test_formats = \{\
        'TIFF': 'test.tiff',\
        'OME-TIFF': 'test.ome.tiff',\
        'PNG': 'test.png',\
        'JPEG': 'test.jpg'\
    \}\
    \
    import tempfile\
    import numpy as np\
    \
    with tempfile.TemporaryDirectory() as tmpdir:\
        tmpdir = Path(tmpdir)\
        \
        # Create test images\
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)\
        \
        for format_name, filename in test_formats.items():\
            try:\
                test_path = tmpdir / filename\
                \
                if format_name in ['TIFF', 'OME-TIFF']:\
                    import tifffile\
                    tifffile.imwrite(test_path, test_data)\
                else:\
                    import cv2\
                    cv2.imwrite(str(test_path), test_data)\
                \
                # Try to read back\
                if format_name in ['TIFF', 'OME-TIFF']:\
                    data = tifffile.imread(test_path)\
                else:\
                    data = cv2.imread(str(test_path))\
                \
                if data is not None:\
                    print(f"\{format_name:15\} \uc0\u10003  Read/write successful")\
                else:\
                    print(f"\{format_name:15\} \uc0\u10007  Read failed")\
                    \
            except Exception as e:\
                print(f"\{format_name:15\} \uc0\u10007  \{str(e)[:40]\}...")\
\
\
def test_gpu_memory():\
    """Test GPU memory allocation"""\
    print_header("GPU Memory Test")\
    \
    try:\
        import torch\
        \
        if not torch.cuda.is_available():\
            print("No GPU available for testing")\
            return\
        \
        device = torch.device('cuda')\
        \
        # Try allocating increasingly large tensors\
        sizes_gb = [1, 10, 50, 100, 120]\
        \
        for size_gb in sizes_gb:\
            try:\
                # Calculate tensor size for GB allocation\
                elements = int(size_gb * 1024**3 / 4)  # float32 = 4 bytes\
                tensor = torch.zeros(elements, device=device)\
                actual_gb = tensor.element_size() * tensor.nelement() / 1024**3\
                print(f"Allocated \{actual_gb:.1f\} GB \uc0\u10003 ")\
                del tensor\
                torch.cuda.empty_cache()\
            except RuntimeError as e:\
                if "out of memory" in str(e):\
                    print(f"Failed at \{size_gb\} GB - Maximum reached")\
                    break\
                else:\
                    print(f"Failed at \{size_gb\} GB - \{e\}")\
                    break\
                    \
    except ImportError:\
        print("PyTorch not available for GPU testing")\
\
\
def main():\
    print("="*60)\
    print(" RegWSI Pipeline Diagnostic Tool")\
    print("="*60)\
    \
    # Run all checks\
    check_system()\
    check_gpu()\
    check_dependencies()\
    check_tools()\
    check_models()\
    test_image_loading()\
    test_gpu_memory()\
    \
    print_header("Diagnostic Complete")\
    print("\\nIf any issues were found, please:")\
    print("1. Run setup_regwsi_vastai.sh to install missing dependencies")\
    print("2. Download model files if missing")\
    print("3. Check GPU drivers if CUDA is not available")\
    print("4. Ensure sufficient disk space for processing")\
\
\
if __name__ == "__main__":\
    main()}
