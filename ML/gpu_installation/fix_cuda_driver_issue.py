"""
Fix CUDA driver compatibility issue.
PyTorch 2.7.1+cu118 requires CUDA driver 11.8+, but you have 11.4.
This script will help you install a compatible PyTorch version.
"""

import subprocess
import sys

print("="*70)
print("FIXING CUDA DRIVER COMPATIBILITY ISSUE")
print("="*70)

print("\nProblem:")
print("  • Your NVIDIA driver supports CUDA 11.4")
print("  • PyTorch 2.7.1+cu118 requires CUDA driver 11.8+")
print("  • This causes CUDA to be unavailable")

print("\n" + "="*70)
print("SOLUTION OPTIONS")
print("="*70)

print("\nOption 1: Update NVIDIA Drivers (RECOMMENDED)")
print("  This will allow you to use the latest PyTorch with CUDA 11.8")
print("  1. Visit: https://www.nvidia.com/Download/index.aspx")
print("  2. Select: Quadro RTX 4000, Windows, your OS version")
print("  3. Download and install the latest drivers")
print("  4. Restart your computer")
print("  5. PyTorch 2.7.1+cu118 will then work")

print("\nOption 2: Install Older PyTorch Compatible with CUDA 11.4")
print("  This installs PyTorch 2.0.1 which works with CUDA 11.4 drivers")
print("  (Note: This is an older version, but should work)")

response = input("\nWhich option? (1 for update drivers, 2 for install older PyTorch): ")

if response == "2":
    print("\nInstalling PyTorch 2.0.1 with CUDA 11.8 (compatible with 11.4 drivers)...")
    print("(This version should work with your CUDA 11.4 driver)")
    
    try:
        # Uninstall current PyTorch
        print("\nUninstalling current PyTorch...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'], check=False)
        
        # Install PyTorch 2.0.1 with CUDA 11.8 (should work with 11.4 driver)
        print("\nInstalling PyTorch 2.0.1+cu118...")
        cmd = [sys.executable, '-m', 'pip', 'install', 
               'torch==2.0.1', 'torchvision==0.15.2', 'torchaudio==2.0.2',
               '--index-url', 'https://download.pytorch.org/whl/cu118']
        result = subprocess.run(cmd, check=True)
        
        print("\n✓ Installation complete!")
        print("\nVerifying...")
        
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✓ SUCCESS! CUDA is now available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠ CUDA still not available.")
            print(f"  You may need to update your NVIDIA drivers (Option 1)")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nYou may need to update your NVIDIA drivers instead.")
        
elif response == "1":
    print("\n" + "="*70)
    print("DRIVER UPDATE INSTRUCTIONS")
    print("="*70)
    print("\n1. Visit: https://www.nvidia.com/Download/index.aspx")
    print("2. Select:")
    print("   - Product Type: Quadro")
    print("   - Product Series: Quadro RTX Series")
    print("   - Product: Quadro RTX 4000")
    print("   - Operating System: Windows 11 (or your version)")
    print("   - Download Type: Standard")
    print("3. Download and run the installer")
    print("4. Restart your computer")
    print("5. After restart, run: python check_gpu.py")
    print("\nAfter updating drivers, PyTorch 2.7.1+cu118 should work!")
    
else:
    print("\nInvalid choice. Please run this script again and choose 1 or 2.")

print("\n" + "="*70)
