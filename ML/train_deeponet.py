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
    "data_dir": "data",              # Directory containing training_data.npz
    "output_dir": "models",          # Directory to save trained models
    
    # Training hyperparameters
    "batch_size": 256,               # Batch size for training
    "epochs": 100,                   # Number of training epochs
    "lr": 1e-3,                      # Learning rate
    "val_split": 0.2,                # Validation split fraction (0.2 = 20% validation)
    
    # Model architecture
    "hidden_dims": [128, 128, 128],  # Hidden layer dimensions for branch/trunk networks
    "branch_output_dim": 128,        # Output dimension of branch/trunk (must match)
    
    # Training options
    "use_physics_reg": True,         # Use physics regularizers in loss
    "device": "auto",                # Device: "auto", "cpu", or "cuda"
}


class UreaseDataset(Dataset):
    """Dataset for (x, t) -> y pairs."""
    
    def __init__(self, params: np.ndarray, outputs: np.ndarray, t_grid: np.ndarray, use_totals: bool = True):
        """
        Parameters:
        -----------
        params: (n_samples, n_params) parameter vectors
        outputs: (n_samples, n_times, n_outputs) output trajectories
        t_grid: (n_times,) time points
        use_totals: if True, outputs are [S, Ntot, Ctot]; else [pH]
        """
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.t_grid = torch.FloatTensor(t_grid)
        self.use_totals = use_totals
        
        self.n_samples, self.n_times, self.n_outputs = outputs.shape
    
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


def train_epoch(model, dataloader, optimizer, criterion, device, use_physics_reg: bool = True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, t, y in dataloader:
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)
        
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
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for x, t, y in dataloader:
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)
            
            y_pred = model(x, t)
            loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            n_batches += 1
    
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
    
    print(f"Data shape: params={params.shape}, outputs={outputs.shape}, t_grid={t_grid.shape}")
    print(f"Using totals: {use_totals}, n_outputs: {n_outputs}")
    
    # Device
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
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
    train_dataset = UreaseDataset(train_params, train_outputs, t_grid, use_totals=use_totals)
    val_dataset = UreaseDataset(val_params, val_outputs, t_grid, use_totals=use_totals)
    
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
    
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
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
    
    print("\nStarting training...")
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, CONFIG["use_physics_reg"])
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': CONFIG,  # Save CONFIG instead of args
                'metadata': metadata
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
    
    # Save final model
    torch.save({
        'epoch': CONFIG["epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': CONFIG,  # Save CONFIG instead of args
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
