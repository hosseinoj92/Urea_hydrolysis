"""
Quick script to verify GPU availability and PyTorch CUDA setup.
Run this before training to ensure your GPU is detected.
"""

import torch
import sys
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*60)
print("GPU VERIFICATION FOR DEEPONET TRAINING")
print("="*60)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"\n✓ GPU DETECTED!")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    
    # GPU properties
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU Specifications:")
    print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Multiprocessors: {props.multi_processor_count}")
    
    # Current memory usage
    print(f"\nCurrent GPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.3f} GB")
    print(f"  Free: {(props.total_memory - torch.cuda.memory_reserved(0)) / 1e9:.3f} GB")
    
    # Test GPU computation
    print(f"\nTesting GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"  ✓ GPU computation test passed!")
        print(f"  Result tensor device: {z.device}")
        print(f"  Result tensor shape: {z.shape}")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ GPU computation test failed: {e}")
        sys.exit(1)
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS FOR TRAINING")
    print("="*60)
    print(f"Your GPU: NVIDIA Quadro RTX 4000")
    print(f"  • Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"  • Recommended batch_size: 512-1024 (depending on model size)")
    print(f"  • With architecture [256, 256, 256], batch_size=1024 should fit")
    print(f"  • Monitor GPU memory during training with: nvidia-smi")
    print(f"\nTo verify GPU usage during training:")
    print(f"  1. Open a separate terminal")
    print(f"  2. Run: nvidia-smi -l 1  (updates every 1 second)")
    print(f"  3. You should see GPU utilization ~80-100% during training")
    
else:
    print(f"\n⚠ NO GPU DETECTED!")
    print(f"  Training will use CPU (much slower)")
    print(f"\nTroubleshooting:")
    print(f"  1. Check if NVIDIA drivers are installed: nvidia-smi")
    print(f"  2. Check if CUDA toolkit is installed")
    print(f"  3. Verify PyTorch was installed with CUDA support:")
    print(f"     Run: python -c 'import torch; print(torch.cuda.is_available())'")
    print(f"  4. If False, reinstall PyTorch with CUDA:")
    print(f"     Visit: https://pytorch.org/get-started/locally/")

print("\n" + "="*60)
