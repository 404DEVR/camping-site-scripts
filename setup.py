"""
Setup Script for ResNet-CBAM SAR Landcover Classification

This script helps with the initial setup and installation of dependencies.

Author: Kiro AI Assistant
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    print("\nSetting up virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activation_command():
    """Get the command to activate virtual environment"""
    system = platform.system().lower()
    
    if system == "windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Determine pip executable
    system = platform.system().lower()
    if system == "windows":
        pip_executable = "venv\\Scripts\\pip"
    else:
        pip_executable = "venv/bin/pip"
    
    # Check if virtual environment exists
    if not Path(pip_executable).exists():
        print("⚠ Virtual environment not found. Installing globally...")
        pip_executable = "pip"
    
    try:
        # Upgrade pip first
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
        print("✅ Pip upgraded successfully")
        
        # Install requirements
        subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gpu_support():
    """Check for GPU support"""
    print("\nChecking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_count} GPU(s) detected")
            print(f"   Primary GPU: {gpu_name}")
            
            # Check CUDA version
            cuda_version = torch.version.cuda
            print(f"   CUDA Version: {cuda_version}")
            return True
        else:
            print("⚠ CUDA not available. Training will use CPU (slower)")
            return False
            
    except ImportError:
        print("⚠ PyTorch not installed yet. GPU check will be performed after installation.")
        return None

def create_directory_structure():
    """Create necessary directories"""
    print("\nCreating directory structure...")
    
    directories = [
        "checkpoints",
        "logs", 
        "results",
        "runs",
        "dataset",
        "dataset/train",
        "dataset/test"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_basic_test():
    """Run basic functionality test"""
    print("\nRunning basic tests...")
    
    try:
        result = subprocess.run([sys.executable, "simple_test.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Basic tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Basic tests failed: {e}")
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return False

def run_model_test():
    """Run model architecture test"""
    print("\nRunning model tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_model.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Model tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model tests failed: {e}")
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return False

def print_next_steps():
    """Print next steps for the user"""
    activation_cmd = get_activation_command()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Activate virtual environment: {activation_cmd}")
    print("2. Prepare your SAR dataset in the following structure:")
    print("   dataset/")
    print("   ├── train/")
    print("   │   ├── Bare_Soil/")
    print("   │   ├── Forest/")
    print("   │   ├── Water/")
    print("   │   ├── Snow_Ice/")
    print("   │   └── Grassland/")
    print("   └── test/")
    print("       ├── Bare_Soil/")
    print("       ├── Forest/")
    print("       ├── Water/")
    print("       ├── Snow_Ice/")
    print("       └── Grassland/")
    print("\n3. Start training:")
    print("   python train.py --data_dir dataset")
    print("\n4. Evaluate model:")
    print("   python evaluate.py --checkpoint_path checkpoints/resnet_cbam_sar.pth")
    print("\n5. Run inference:")
    print("   python inference.py --model_path checkpoints/resnet_cbam_sar.pth --input path/to/images")
    print("\nFor more details, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print("="*60)
    print("ResNet-CBAM SAR Classification - Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("⚠ Continuing without virtual environment...")
    
    # Create directory structure
    create_directory_structure()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Run basic tests
    if not run_basic_test():
        print("❌ Setup failed during basic tests")
        sys.exit(1)
    
    # Check GPU support (after PyTorch installation)
    check_gpu_support()
    
    # Run model tests
    if not run_model_test():
        print("⚠ Model tests failed. Check PyTorch installation.")
        print("You can still proceed, but verify dependencies manually.")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()