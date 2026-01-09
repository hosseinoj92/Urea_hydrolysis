"""
Train early inference model for parameter estimation from pH prefix sequences.
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

from early_inference_model import create_early_inference_model, gaussian_nll_loss

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Data paths
    "data_dir": "Generated_Data_EarlyInference_20000",
    "output_dir": "models_early_inference",
    "prefix_length": 60.0,  # Which prefix length to train on (10, 30, or 60 seconds)
    
    # Training hyperparameters
    "batch_size": 128,
    "epochs": 100,
    "lr": 1e-3,
    "val_split": 0.2,
    "early_stopping_patience": 20,
    
    # Model architecture
    "tcn_channels": [64, 128, 256],
    "tcn_kernel_size": 3,
    "tcn_dropout": 0.2,
    "mlp_hidden_dims": [128, 64],
    "output_dropout": 0.1,
    "use_uncertainty": True,
    
    # Training options
    "device": "auto",
    "normalize_inputs": True,
    "normalize_outputs": True,
}


class EarlyInferenceDataset(Dataset):
    """Dataset for early inference training."""
    
    def __init__(
        self,
        pH_prefix: np.ndarray,
        t_prefix: np.ndarray,
        known_inputs: np.ndarray,
        target_params: np.ndarray,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True,
        input_stats: dict = None,
        output_stats: dict = None,
    ):
        """
        Parameters
        ----------
        pH_prefix: (n_samples, seq_len) array
        t_prefix: (n_samples, seq_len) array
        known_inputs: (n_samples, n_known) array
        target_params: (n_samples, n_params) array
        """
        self.n_samples = pH_prefix.shape[0]
        self.seq_length = pH_prefix.shape[1]
        self.n_known = known_inputs.shape[1]
        self.n_params = target_params.shape[1]
        
        # Normalize inputs
        if normalize_inputs:
            if input_stats is None:
                # Compute stats from data
                self.pH_mean = np.mean(pH_prefix)
                self.pH_std = np.std(pH_prefix) + 1e-8
                self.known_mean = np.mean(known_inputs, axis=0, keepdims=True)
                self.known_std = np.std(known_inputs, axis=0, keepdims=True) + 1e-8
            else:
                self.pH_mean = input_stats["pH_mean"]
                self.pH_std = input_stats["pH_std"]
                self.known_mean = input_stats["known_mean"]
                self.known_std = input_stats["known_std"]
            
            self.pH_prefix = (pH_prefix - self.pH_mean) / self.pH_std
            self.known_inputs = (known_inputs - self.known_mean) / self.known_std
        else:
            self.pH_prefix = pH_prefix
            self.known_inputs = known_inputs
            self.pH_mean = 0.0
            self.pH_std = 1.0
            self.known_mean = np.zeros((1, self.n_known))
            self.known_std = np.ones((1, self.n_known))
        
        # Normalize outputs
        if normalize_outputs:
            if output_stats is None:
                self.param_mean = np.mean(target_params, axis=0, keepdims=True)
                self.param_std = np.std(target_params, axis=0, keepdims=True) + 1e-8
            else:
                self.param_mean = output_stats["param_mean"]
                self.param_std = output_stats["param_std"]
            
            self.target_params = (target_params - self.param_mean) / self.param_std
        else:
            self.target_params = target_params
            self.param_mean = np.zeros((1, self.n_params))
            self.param_std = np.ones((1, self.n_params))
        
        # Convert to tensors
        self.pH_prefix = torch.FloatTensor(self.pH_prefix)
        self.known_inputs = torch.FloatTensor(self.known_inputs)
        self.target_params = torch.FloatTensor(self.target_params)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return (
            self.pH_prefix[idx],      # (seq_len,)
            self.known_inputs[idx],   # (n_known,)
            self.target_params[idx],  # (n_params,)
        )
    
    def get_normalization_stats(self):
        """Get normalization statistics."""
        return {
            "input": {
                "pH_mean": self.pH_mean,
                "pH_std": self.pH_std,
                "known_mean": self.known_mean.squeeze().tolist(),
                "known_std": self.known_std.squeeze().tolist(),
            },
            "output": {
                "param_mean": self.param_mean.squeeze().tolist(),
                "param_std": self.param_std.squeeze().tolist(),
            },
        }


def train_epoch(model, dataloader, optimizer, criterion, device, use_uncertainty: bool, epoch: int, total_epochs: int):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", 
                unit="batch", leave=False, ncols=100)
    
    for pH_seq, known_inputs, target_params in pbar:
        pH_seq = pH_seq.to(device)
        known_inputs = known_inputs.to(device)
        target_params = target_params.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_uncertainty:
            mean, logvar = model(pH_seq, known_inputs)
            loss = gaussian_nll_loss(mean, logvar, target_params)
        else:
            mean = model.predict(pH_seq, known_inputs)
            loss = criterion(mean, target_params)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss_val:.6f}',
            'avg_loss': f'{total_loss/n_batches:.6f}',
        })
    
    return total_loss / n_batches


def validate(model, dataloader, criterion, device, use_uncertainty: bool, epoch: int, total_epochs: int):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", 
                unit="batch", leave=False, ncols=100)
    
    with torch.no_grad():
        for pH_seq, known_inputs, target_params in pbar:
            pH_seq = pH_seq.to(device)
            known_inputs = known_inputs.to(device)
            target_params = target_params.to(device)
            
            if use_uncertainty:
                mean, logvar = model(pH_seq, known_inputs)
                loss = gaussian_nll_loss(mean, logvar, target_params)
            else:
                mean = model.predict(pH_seq, known_inputs)
                loss = criterion(mean, target_params)
            
            loss_val = loss.item()
            total_loss += loss_val
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss_val:.6f}',
                'avg_loss': f'{total_loss/n_batches:.6f}',
            })
    
    return total_loss / n_batches


def main():
    # Use CONFIG dictionary
    data_dir = Path(CONFIG["data_dir"])
    data_file = data_dir / "training_data.npz"
    prefix_length = CONFIG["prefix_length"]
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}. Run generate_early_inference_data.py first.")
    
    print(f"Loading data from {data_file}...")
    data = np.load(data_file, allow_pickle=True)
    
    # Load data for specified prefix length
    prefix_key = f"prefix_{int(prefix_length)}s"
    if prefix_key not in data:
        raise ValueError(f"Prefix length {prefix_length}s not found in data. Available: {list(data.keys())}")
    
    prefix_data = data[prefix_key].item()
    pH_prefix = prefix_data["pH_prefix"]
    known_inputs = prefix_data["known_inputs"]
    target_params = prefix_data["target_params"]
    
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"Data shape: pH_prefix={pH_prefix.shape}, known_inputs={known_inputs.shape}, target_params={target_params.shape}")
    print(f"Infer params: {metadata['infer_params']}")
    
    # Device selection
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    # Train/val split
    n_samples = pH_prefix.shape[0]
    n_train = int(n_samples * (1 - CONFIG["val_split"]))
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_pH = pH_prefix[train_indices]
    train_known = known_inputs[train_indices]
    train_targets = target_params[train_indices]
    val_pH = pH_prefix[val_indices]
    val_known = known_inputs[val_indices]
    val_targets = target_params[val_indices]
    
    print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
    
    # Extract time arrays from data (for future use - currently using zeros for backward compatibility)
    # Note: To use actual time as features, load t_prefix from dataset and pass here
    train_t = np.zeros_like(train_pH)  # TODO: Replace with actual t_prefix when ready
    val_t = np.zeros_like(val_pH)      # TODO: Replace with actual t_prefix when ready
    
    # Create datasets
    train_dataset = EarlyInferenceDataset(
        train_pH, train_t, train_known, train_targets,
        normalize_inputs=CONFIG["normalize_inputs"],
        normalize_outputs=CONFIG["normalize_outputs"],
    )
    
    val_dataset = EarlyInferenceDataset(
        val_pH, val_t, val_known, val_targets,
        normalize_inputs=CONFIG["normalize_inputs"],
        normalize_outputs=CONFIG["normalize_outputs"],
        input_stats={
            "pH_mean": train_dataset.pH_mean,
            "pH_std": train_dataset.pH_std,
            "known_mean": train_dataset.known_mean,
            "known_std": train_dataset.known_std,
        },
        output_stats={
            "param_mean": train_dataset.param_mean,
            "param_std": train_dataset.param_std,
        },
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # Create model
    seq_length = train_pH.shape[1]
    n_known = train_known.shape[1]
    n_params = train_targets.shape[1]
    
    model = create_early_inference_model(
        seq_length=seq_length,
        n_known_inputs=n_known,
        n_output_params=n_params,
        tcn_channels=CONFIG["tcn_channels"],
        tcn_kernel_size=CONFIG["tcn_kernel_size"],
        tcn_dropout=CONFIG["tcn_dropout"],
        mlp_hidden_dims=CONFIG["mlp_hidden_dims"],
        output_dropout=CONFIG["output_dropout"],
        use_uncertainty=CONFIG["use_uncertainty"],
    ).to(device)
    
    n_model_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {n_model_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Prefix length: {prefix_length}s")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['lr']}")
    print(f"Early stopping patience: {CONFIG['early_stopping_patience']}")
    print("="*60)
    print("\nStarting training...")
    
    epoch_pbar = tqdm(range(CONFIG["epochs"]), desc="Training Progress", unit="epoch", ncols=120)
    
    for epoch in epoch_pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                CONFIG["use_uncertainty"], epoch=epoch, total_epochs=CONFIG["epochs"])
        val_loss = validate(model, val_loader, criterion, device, 
                           CONFIG["use_uncertainty"], epoch=epoch, total_epochs=CONFIG["epochs"])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Track learning rate before scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print message if learning rate changed
        if old_lr != new_lr:
            epoch_pbar.write(f"  → Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        best_val_str = f'{best_val_loss:.6f}' if best_val_loss != float('inf') else 'N/A'
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'best_val': best_val_str,
            'patience': f'{patience_counter}/{CONFIG["early_stopping_patience"]}'
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            normalization_stats = train_dataset.get_normalization_stats()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
                'normalization_stats': normalization_stats,
                'metadata': metadata,
                'prefix_length': prefix_length,
            }, output_dir / f"best_model_prefix_{int(prefix_length)}s.pt")
            epoch_pbar.write(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                epoch_pbar.write(f"\nEarly stopping triggered!")
                break
    
    epoch_pbar.close()
    
    # Save final model
    normalization_stats = train_dataset.get_normalization_stats()
    torch.save({
        'epoch': CONFIG["epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': CONFIG,
        'normalization_stats': normalization_stats,
        'metadata': metadata,
        'prefix_length': prefix_length,
    }, output_dir / f"final_model_prefix_{int(prefix_length)}s.pt")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Training Curves (Prefix {prefix_length}s)')
    plt.savefig(output_dir / f"training_curves_prefix_{int(prefix_length)}s.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
