"""
Script to help install PyTorch with CUDA support for NVIDIA Quadro RTX 4000.
This will guide you through the installation process.
"""

import subprocess
import sys
import platform

print("="*70)
print("PYTORCH CUDA INSTALLATION GUIDE FOR NVIDIA QUADRO RTX 4000")
print("="*70)

print("\nStep 1: Checking system information...")
print(f"  Python version: {sys.version}")
print(f"  Platform: {platform.platform()}")

print("\nStep 2: Checking for NVIDIA drivers...")
print("  Attempting to run 'nvidia-smi'...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("  ✓ NVIDIA drivers detected!")
        print("\n  GPU Information:")
        # Extract GPU info from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line or 'Driver Version' in line or 'CUDA Version' in line:
                print(f"    {line.strip()}")
    else:
        print("  ⚠ nvidia-smi returned an error")
        print("  This might indicate driver issues")
except FileNotFoundError:
    print("  ✗ nvidia-smi not found!")
    print("\n  ⚠ CRITICAL: NVIDIA drivers may not be installed.")
    print("  Please install NVIDIA drivers first:")
    print("    1. Visit: https://www.nvidia.com/Download/index.aspx")
    print("    2. Select your Quadro RTX 4000 and download drivers")
    print("    3. Install the drivers and restart your computer")
    print("    4. Then run this script again")
    sys.exit(1)
except Exception as e:
    print(f"  ⚠ Error running nvidia-smi: {e}")
    print("  You may need to install NVIDIA drivers first")

print("\n" + "="*70)
print("Step 3: Installing PyTorch with CUDA support")
print("="*70)

print("\nFor Python 3.12 on Windows, you have two options:")
print("\nOption A: Install via pip (Recommended)")
print("  Run this command:")
print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("\n  This installs PyTorch with CUDA 12.1 support")

print("\nOption B: Install via conda (if using conda)")
print("  Run this command:")
print("    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")

print("\n" + "="*70)
print("RECOMMENDED INSTALLATION COMMAND")
print("="*70)
print("\nFor your Quadro RTX 4000, run:")
print("\n  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("\nOr if you prefer CUDA 11.8 (more stable):")
print("\n  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*70)
print("After installation, verify with:")
print("="*70)
print("  python check_gpu.py")
print("\nYou should see:")
print("  ✓ GPU DETECTED!")
print("  GPU Name: NVIDIA Quadro RTX 4000")

print("\n" + "="*70)
print("Would you like to install PyTorch with CUDA now?")
print("="*70)
response = input("\nType 'yes' to proceed with installation, or 'no' to exit: ")

if response.lower() in ['yes', 'y']:
    print("\nInstalling PyTorch with CUDA 12.1...")
    print("(This may take several minutes)")
    
    try:
        # Try CUDA 12.1 first
        cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', 
               '--index-url', 'https://download.pytorch.org/whl/cu121']
        result = subprocess.run(cmd, check=True)
        print("\n✓ Installation complete!")
        print("\nVerifying installation...")
        
        # Test import
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✓ SUCCESS! CUDA is now available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠ CUDA still not available. You may need to:")
            print(f"     1. Restart your Python environment")
            print(f"     2. Check NVIDIA driver installation")
            print(f"     3. Try CUDA 11.8 instead")
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        print("\nTry installing manually:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except Exception as e:
        print(f"\n✗ Error: {e}")
else:
    print("\nInstallation cancelled. Run the pip command manually when ready.")

print("\n" + "="*70)
