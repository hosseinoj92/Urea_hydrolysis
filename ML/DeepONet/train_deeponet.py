"""
Train DeepONet surrogate model on synthetic mechanistic simulation data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from deeponet import create_deeponet, physics_loss_regularizer
from mechanistic_simulator import UreaseSimulator

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Data paths
    "data_dir": "Generated_Data_6params_50000",              # Directory containing training_data.npz
    "output_dir": "models",          # Directory to save trained models
    
    # Training hyperparameters
    "batch_size": 512,              # Batch size for training (increased for large dataset)
    "epochs": 3,                   # Number of training epochs (with early stopping, this is max)
    "lr": 1e-3,                      # Learning rate (slightly higher for faster initial learning)
    "val_split": 0.2,                # Validation split fraction (0.2 = 20% validation)
    "early_stopping_patience": 2,    # Stop if val loss doesn't improve for N epochs
    
    # Model architecture (increased capacity for 5 inputs and complex dynamics)
    "hidden_dims": [256, 256, 256],  # Hidden layer dimensions for branch/trunk networks
    "branch_output_dim": 256,        # Output dimension of branch/trunk (must match)
    
    # Training options
    "use_physics_reg": True,         # Use physics regularizers in loss
    "device": "auto",                # Device: "auto", "cpu", or "cuda"
}


class UreaseDataset(Dataset):
    """Dataset for (x, t) -> y pairs with optional normalization."""
    
    def __init__(
        self, 
        params: np.ndarray, 
        outputs: np.ndarray, 
        t_grid: np.ndarray, 
        use_totals: bool = True,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
        param_mean: np.ndarray = None,
        param_std: np.ndarray = None,
        output_mean: np.ndarray = None,
        output_std: np.ndarray = None,
        t_max: float = None
    ):
        """
        Parameters:
        -----------
        params: (n_samples, n_params) parameter vectors
        outputs: (n_samples, n_times, n_outputs) output trajectories
        t_grid: (n_times,) time points
        use_totals: if True, outputs are [S, Ntot, Ctot]; else [pH]
        normalize_inputs: if True, normalize params and time
        normalize_outputs: if True, normalize outputs
        param_mean, param_std: normalization stats for params (if None, computed from data)
        output_mean, output_std: normalization stats for outputs (if None, computed from data)
        t_max: max time for normalization (if None, uses max of t_grid)
        """
        self.use_totals = use_totals
        self.n_samples, self.n_times, self.n_outputs = outputs.shape
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        
        # Normalize parameters
        if normalize_inputs:
            if param_mean is None or param_std is None:
                self.param_mean = np.mean(params, axis=0, keepdims=True)
                self.param_std = np.std(params, axis=0, keepdims=True) + 1e-8  # Avoid division by zero
            else:
                self.param_mean = param_mean
                self.param_std = param_std
            self.params = torch.FloatTensor((params - self.param_mean) / self.param_std)
        else:
            self.params = torch.FloatTensor(params)
            self.param_mean = np.zeros((1, params.shape[1]))
            self.param_std = np.ones((1, params.shape[1]))
        
        # Normalize time
        if normalize_inputs:
            if t_max is None:
                t_max = np.max(t_grid)
            self.t_max = t_max
            self.t_grid = torch.FloatTensor(t_grid / t_max)  # Normalize to [0, 1]
        else:
            self.t_grid = torch.FloatTensor(t_grid)
            self.t_max = np.max(t_grid) if t_max is None else t_max
        
        # Normalize outputs
        if normalize_outputs:
            if output_mean is None or output_std is None:
                # Compute stats over all samples and times
                self.output_mean = np.mean(outputs, axis=(0, 1), keepdims=True)
                self.output_std = np.std(outputs, axis=(0, 1), keepdims=True) + 1e-8
            else:
                self.output_mean = output_mean
                self.output_std = output_std
            self.outputs = torch.FloatTensor((outputs - self.output_mean) / self.output_std)
        else:
            self.outputs = torch.FloatTensor(outputs)
            self.output_mean = np.zeros((1, 1, self.n_outputs))
            self.output_std = np.ones((1, 1, self.n_outputs))
    
    def __len__(self):
        return self.n_samples * self.n_times
    
    def __getitem__(self, idx):
        # Flatten (sample, time) -> single index
        sample_idx = idx // self.n_times
        time_idx = idx % self.n_times
        
        x = self.params[sample_idx]  # (n_params,)
        t = self.t_grid[time_idx].unsqueeze(0)  # (1,)
        y = self.outputs[sample_idx, time_idx]  # (n_outputs,)
        
        return x, t, y


def train_epoch(model, dataloader, optimizer, criterion, device, use_physics_reg: bool = True, epoch: int = 0, total_epochs: int = 1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    # Reset peak memory stats at start of first epoch
    if epoch == 0 and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(0)
    
    # Create progress bar for batches
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", 
                unit="batch", leave=False, ncols=100)
    
    for batch_idx, (x, t, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)  # non_blocking for faster CPU->GPU transfer
        t = t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x, t)
        
        # MSE loss
        loss = criterion(y_pred, y)
        
        # Physics regularizer
        if use_physics_reg:
            reg_loss = physics_loss_regularizer(
                y_pred,
                t,
                mode="totals" if y.shape[1] > 1 else "pH"
            )
            loss = loss + reg_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_val:.6f}',
            'avg_loss': f'{total_loss/n_batches:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Print GPU memory usage after first epoch
    if device.type == "cuda" and epoch == 0:
        peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
        current_memory = torch.cuda.memory_allocated(0) / 1e9
        reserved_memory = torch.cuda.memory_reserved(0) / 1e9
        print(f"  GPU Memory Usage:")
        print(f"    Allocated: {current_memory:.2f} GB")
        print(f"    Reserved: {reserved_memory:.2f} GB")
        print(f"    Peak (this epoch): {peak_memory:.2f} GB")
    
    return total_loss / n_batches


def validate(model, dataloader, criterion, device, epoch: int = 0, total_epochs: int = 1):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    # Create progress bar for validation
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", 
                unit="batch", leave=False, ncols=100)
    
    with torch.no_grad():
        for x, t, y in pbar:
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)
            
            y_pred = model(x, t)
            loss = criterion(y_pred, y)
            
            loss_val = loss.item()
            total_loss += loss_val
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_val:.6f}',
                'avg_loss': f'{total_loss/n_batches:.6f}'
            })
    
    return total_loss / n_batches


def main():
    # Use CONFIG dictionary for all parameters
    data_dir = Path(CONFIG["data_dir"])
    data_file = data_dir / "training_data.npz"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}. Run generate_training_data.py first.")
    
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    params = data["params"]
    outputs = data["outputs"]
    t_grid = data["t_grid"]
    
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    use_totals = metadata.get("use_totals", True)
    n_outputs = outputs.shape[2]
    
    # Display parameter configuration
    variable_params = metadata.get("variable_params", [])
    fixed_params = metadata.get("fixed_params", {})
    n_variable_params = metadata.get("n_variable_params", params.shape[1])
    
    print(f"Data shape: params={params.shape}, outputs={outputs.shape}, t_grid={t_grid.shape}")
    print(f"Using totals: {use_totals}, n_outputs: {n_outputs}")
    print(f"Variable parameters ({len(variable_params)}): {variable_params}")
    if fixed_params:
        print(f"Fixed parameters: {fixed_params}")
    print(f"Model input dimension: {n_variable_params} (from variable parameters only)")
    
    # Device selection and verification
    if CONFIG["device"] == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            print("⚠ CUDA not available, using CPU")
    else:
        device = torch.device(CONFIG["device"])
        if device.type == "cuda":
            if torch.cuda.is_available():
                print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("⚠ CUDA requested but not available, falling back to CPU")
                device = torch.device("cpu")
        else:
            print(f"Using device: {device}")
    
    # Verify GPU is actually being used
    if device.type == "cuda":
        print(f"  Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
        print(f"  Current GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.3f} GB")
    
    # Train/val split
    n_samples = params.shape[0]
    n_train = int(n_samples * (1 - CONFIG["val_split"]))
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_params = params[train_indices]
    train_outputs = outputs[train_indices]
    val_params = params[val_indices]
    val_outputs = outputs[val_indices]
    
    print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
    
    # Create datasets
    # Create datasets with normalization
    # Use training data to compute normalization stats (to avoid data leakage)
    train_dataset = UreaseDataset(
        train_params, train_outputs, t_grid, 
        use_totals=use_totals,
        normalize_inputs=True,
        normalize_outputs=True
    )
    
    # Use same normalization stats for validation set
    val_dataset = UreaseDataset(
        val_params, val_outputs, t_grid,
        use_totals=use_totals,
        normalize_inputs=True,
        normalize_outputs=True,
        param_mean=train_dataset.param_mean,
        param_std=train_dataset.param_std,
        output_mean=train_dataset.output_mean,
        output_std=train_dataset.output_std,
        t_max=train_dataset.t_max
    )
    
    # Save normalization stats to metadata
    normalization_stats = {
        'param_mean': train_dataset.param_mean.squeeze().tolist(),
        'param_std': train_dataset.param_std.squeeze().tolist(),
        'output_mean': train_dataset.output_mean.squeeze().tolist(),
        'output_std': train_dataset.output_std.squeeze().tolist(),
        't_max': float(train_dataset.t_max),
        'normalize_inputs': True,
        'normalize_outputs': True,
    }
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # Create model
    input_dim = params.shape[1]
    model = create_deeponet(
        input_dim=input_dim,
        n_outputs=n_outputs,
        hidden_dims=CONFIG["hidden_dims"],
        branch_output_dim=CONFIG["branch_output_dim"],
        use_bias=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {n_params:,} parameters")
    if device.type == "cuda":
        model_size_mb = n_params * 4 / 1e6  # 4 bytes per float32
        print(f"  Model size: ~{model_size_mb:.2f} MB (float32)")
        print(f"  Model moved to GPU: {next(model.parameters()).is_cuda}")
        print(f"  Model device: {next(model.parameters()).device}")
        
        # Estimate memory needed for training
        batch_size = CONFIG["batch_size"]
        input_dim = params.shape[1]
        n_outputs = outputs.shape[2]
        # Rough estimate: batch_size * (input_dim + n_outputs) * 4 bytes per float
        batch_memory_mb = batch_size * (input_dim + n_outputs + 100) * 4 / 1e6  # +100 for intermediate activations
        print(f"  Estimated memory per batch: ~{batch_memory_mb:.2f} MB")
        print(f"  Total estimated GPU memory needed: ~{model_size_mb + batch_memory_mb * 2:.2f} MB")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training loop
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = CONFIG.get("early_stopping_patience", 20)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {len(train_indices):,} train samples × {len(t_grid)} time points = {len(train_indices) * len(t_grid):,} training pairs")
    print(f"Batches per epoch: {len(train_loader):,}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Model architecture: {CONFIG['hidden_dims']}, branch_output_dim={CONFIG['branch_output_dim']}")
    print(f"Learning rate: {CONFIG['lr']}")
    print(f"Early stopping: will stop if val loss doesn't improve for {early_stopping_patience} epochs")
    print("="*60)
    print("\nStarting training...")
    
    # Create outer progress bar for epochs
    epoch_pbar = tqdm(range(CONFIG["epochs"]), desc="Training Progress", unit="epoch", ncols=120)
    
    for epoch in epoch_pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                CONFIG["use_physics_reg"], epoch=epoch, total_epochs=CONFIG["epochs"])
        val_loss = validate(model, val_loader, criterion, device, epoch=epoch, total_epochs=CONFIG["epochs"])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Update epoch progress bar
        best_val_str = f'{best_val_loss:.6f}' if best_val_loss != float('inf') else 'N/A'
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'best_val': best_val_str,
            'patience': f'{patience_counter}/{early_stopping_patience}'
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': CONFIG,  # Save CONFIG instead of args
                'normalization_stats': normalization_stats,  # Save normalization stats
                'metadata': metadata
            }, output_dir / "best_model.pt")
            epoch_pbar.write(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                epoch_pbar.write(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
                epoch_pbar.write(f"Best validation loss: {best_val_loss:.6f} at epoch {epoch - early_stopping_patience + 1}")
                break
    
    epoch_pbar.close()
    
    # Save final model
    torch.save({
        'epoch': CONFIG["epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': CONFIG,  # Save CONFIG instead of args
        'normalization_stats': normalization_stats,  # Save normalization stats
        'metadata': metadata
    }, output_dir / "final_model.pt")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.yscale('log')
    plt.title('Training Curves')
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Models saved to: {output_dir}")
    print(f"  Training curves: {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
