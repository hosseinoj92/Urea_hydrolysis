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
    "data_dir": r"C:\Users\vt4ho\Simulations\simulation_data\generated_data\imperfect\version_2\Generated_Data_EarlyInference_100000",
    "output_dir": r"C:\Users\vt4ho\Simulations\simulation_data\models\imperfect\verion_2\models_early_inference_100000_30s",
    "prefix_length": 30.0,  # Which prefix length to train on (10, 30, or 60 seconds)
    
    # Training hyperparameters
    "batch_size": 512,  # Increased for better GPU utilization and stability
    "epochs": 2000,
    "lr": 8e-3,  # Increased and scaled with batch size (2x batch = 2x LR)
    "val_split": 0.2,
    "early_stopping_patience": 50,
    "weight_decay": 1e-5,  # L2 regularization
    "grad_clip_norm": 5.0,  # Increased from 1.0 for better gradient flow
    
    # Learning rate scheduler settings
    "scheduler_factor": 0.7,  # Less aggressive reduction (was 0.5)
    "scheduler_patience": 15,  # More patience (was 5)
    "scheduler_min_lr": 1e-5,  # Minimum LR floor
    "warmup_epochs": 5,  # Warmup period for stable training
    
    # Model architecture
    # Increased depth to cover sequence length 300: [128, 256, 512, 512, 512] gives RF=373
    "tcn_channels": [128, 256, 512, 512, 512],  # Added 5th level for RF >= 300
    "tcn_kernel_size": 7,
    "tcn_dropout": 0.2,
    "mlp_hidden_dims": [256, 128],
    "output_dropout": 0.1,
    "use_uncertainty": True,
    
    # Training options
    "device": "auto",
    "normalize_inputs": True,
    "normalize_outputs": True,
    "seed": 42,  # Random seed for reproducibility (train/val split, etc.)
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
                # B1: Normalize time per-sequence to preserve dt relationships (CRITICAL FIX)
                # This preserves the relative spacing (dt) within each sequence
                t_prefix_norm = []
                for i in range(t_prefix.shape[0]):
                    t_seq = t_prefix[i]
                    if len(t_seq) > 1:
                        t_mean_i = np.mean(t_seq)
                        t_std_i = np.std(t_seq) + 1e-8
                        t_prefix_norm.append((t_seq - t_mean_i) / t_std_i)
                    else:
                        t_prefix_norm.append(t_seq)
                self.t_prefix = np.array(t_prefix_norm)
                # Store dummy stats for compatibility (not used for per-sequence norm)
                self.t_mean = 0.0
                self.t_std = 1.0
                self.known_mean = np.mean(known_inputs, axis=0, keepdims=True)
                self.known_std = np.std(known_inputs, axis=0, keepdims=True) + 1e-8
            else:
                self.pH_mean = input_stats["pH_mean"]
                self.pH_std = input_stats["pH_std"]
                # For validation, also normalize time per-sequence using stored approach
                t_prefix_norm = []
                for i in range(t_prefix.shape[0]):
                    t_seq = t_prefix[i]
                    if len(t_seq) > 1:
                        t_mean_i = np.mean(t_seq)
                        t_std_i = np.std(t_seq) + 1e-8
                        t_prefix_norm.append((t_seq - t_mean_i) / t_std_i)
                    else:
                        t_prefix_norm.append(t_seq)
                self.t_prefix = np.array(t_prefix_norm)
                self.t_mean = input_stats.get("t_mean", 0.0)
                self.t_std = input_stats.get("t_std", 1.0)
                self.known_mean = input_stats["known_mean"]
                self.known_std = input_stats["known_std"]
            
            self.pH_prefix = (pH_prefix - self.pH_mean) / self.pH_std
            # t_prefix already normalized per-sequence above
            self.known_inputs = (known_inputs - self.known_mean) / self.known_std
        else:
            self.pH_prefix = pH_prefix
            self.t_prefix = t_prefix  # Store time even if not normalizing
            self.known_inputs = known_inputs
            self.pH_mean = 0.0
            self.pH_std = 1.0
            self.t_mean = 0.0
            self.t_std = 1.0
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
        self.t_prefix = torch.FloatTensor(self.t_prefix)  # Store time as tensor!
        self.known_inputs = torch.FloatTensor(self.known_inputs)
        self.target_params = torch.FloatTensor(self.target_params)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return (
            self.pH_prefix[idx],      # (seq_len,)
            self.t_prefix[idx],       # (seq_len,) - return time!
            self.known_inputs[idx],   # (n_known,)
            self.target_params[idx],  # (n_params,)
        )
    
    def get_normalization_stats(self):
        """Get normalization statistics."""
        return {
            "input": {
                "pH_mean": self.pH_mean,
                "pH_std": self.pH_std,
                "t_mean": self.t_mean,
                "t_std": self.t_std,
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
    
    for pH_seq, t_seq, known_inputs, target_params in pbar:
        pH_seq = pH_seq.to(device)
        t_seq = t_seq.to(device)  # Pass time to device!
        known_inputs = known_inputs.to(device)
        target_params = target_params.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - now includes time!
        if use_uncertainty:
            mean, logvar = model(pH_seq, t_seq, known_inputs)
            loss = gaussian_nll_loss(mean, logvar, target_params)
        else:
            mean = model.predict(pH_seq, t_seq, known_inputs)
            loss = criterion(mean, target_params)
        
        # F1: Monitor gradient norms for debugging (check if learning is happening)
        # Enable this if loss isn't decreasing
        if epoch == 0 and n_batches == 0:  # First batch of first epoch only
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            pbar.write(f"Initial gradient norm: {total_norm:.6f} (should be > 0.01)")
        
        # Backward pass
        loss.backward()
        
        # F1: Monitor gradient norms for debugging (check if learning is happening)
        # Check gradients after backward, before clipping
        if epoch == 0 and n_batches == 0:  # First batch of first epoch only
            total_norm = 0
            n_params_with_grad = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    n_params_with_grad += 1
            total_norm = total_norm ** (1. / 2)
            pbar.write(f"Gradient check: norm={total_norm:.6f}, params_with_grad={n_params_with_grad}")
            if total_norm < 1e-6:
                pbar.write("WARNING: Gradients are very small! Learning may not happen.")
        
        # F2: Add gradient clipping to prevent exploding gradients
        # Increased max_norm for better gradient flow (was 1.0)
        grad_clip_norm = CONFIG.get("grad_clip_norm", 5.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
        for pH_seq, t_seq, known_inputs, target_params in pbar:
            pH_seq = pH_seq.to(device)
            t_seq = t_seq.to(device)  # Pass time to device!
            known_inputs = known_inputs.to(device)
            target_params = target_params.to(device)
            
            if use_uncertainty:
                mean, logvar = model(pH_seq, t_seq, known_inputs)  # Include time!
                loss = gaussian_nll_loss(mean, logvar, target_params)
                # D1: Check for negative loss (variance collapse/explosion)
                if loss.item() < 0:
                    pbar.write(f"WARNING: Negative loss {loss.item():.6f}, logvar range: [{logvar.min():.4f}, {logvar.max():.4f}]")
            else:
                mean = model.predict(pH_seq, t_seq, known_inputs)  # Include time!
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
        available = [k for k in data.keys() if k.startswith('prefix_')]
        raise ValueError(f"Prefix length {prefix_length}s not found in data. Available: {available}")
    
    prefix_data = data[prefix_key].item()
    pH_prefix = prefix_data["pH_prefix"]
    t_prefix = prefix_data["t_prefix"]  # Load actual time grid!
    known_inputs = prefix_data["known_inputs"]
    target_params = prefix_data["target_params"]
    
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"Data shape: pH_prefix={pH_prefix.shape}, t_prefix={t_prefix.shape}, known_inputs={known_inputs.shape}, target_params={target_params.shape}")
    print(f"Infer params: {metadata['infer_params']}")
    
    # A3: Verify parameter ordering
    assert list(metadata['infer_params']) == ['E0_g_per_L', 'k_d'], \
        f"Param order mismatch! Expected ['E0_g_per_L', 'k_d'], got {metadata['infer_params']}"
    
    # I1: Add shape assertions
    assert pH_prefix.shape[0] == t_prefix.shape[0] == known_inputs.shape[0] == target_params.shape[0], \
        "Sample count mismatch across arrays"
    assert pH_prefix.shape[1] == t_prefix.shape[1], "pH and time sequence length mismatch"
    assert known_inputs.shape[1] == 5, f"Expected 5 known inputs, got {known_inputs.shape[1]}"
    assert target_params.shape[1] == 2, f"Expected 2 target params, got {target_params.shape[1]}"
    
    # I2: Add range checks for target parameters
    assert np.all(target_params[:, 0] >= 0.01), f"E0_g_per_L too small: min={target_params[:, 0].min():.6f}"
    assert np.all(target_params[:, 0] <= 2.0), f"E0_g_per_L too large: max={target_params[:, 0].max():.6f}"
    assert np.all(target_params[:, 1] >= 0), f"k_d negative: min={target_params[:, 1].min():.6f}"
    assert np.all(target_params[:, 1] <= 0.01), f"k_d too large: max={target_params[:, 1].max():.6f}"
    
    # Device selection
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    # Train/val split (with fixed seed for reproducibility)
    n_samples = pH_prefix.shape[0]
    n_train = int(n_samples * (1 - CONFIG["val_split"]))
    # Use fixed seed for reproducible train/val split
    rng = np.random.default_rng(CONFIG.get("seed", 42))
    indices = rng.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_pH = pH_prefix[train_indices]
    train_t = t_prefix[train_indices]  # Load actual time grid!
    train_known = known_inputs[train_indices]
    train_targets = target_params[train_indices]
    val_pH = pH_prefix[val_indices]
    val_t = t_prefix[val_indices]  # Load actual time grid!
    val_known = known_inputs[val_indices]
    val_targets = target_params[val_indices]
    
    print(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
    
    # Create datasets
    train_dataset = EarlyInferenceDataset(
        train_pH, train_t, train_known, train_targets,  # Use actual time grid!
        normalize_inputs=CONFIG["normalize_inputs"],
        normalize_outputs=CONFIG["normalize_outputs"],
    )
    
    val_dataset = EarlyInferenceDataset(
        val_pH, val_t, val_known, val_targets,  # Use actual time grid!
        normalize_inputs=CONFIG["normalize_inputs"],
        normalize_outputs=CONFIG["normalize_outputs"],
        input_stats={
            "pH_mean": train_dataset.pH_mean,
            "pH_std": train_dataset.pH_std,
            "t_mean": train_dataset.t_mean,
            "t_std": train_dataset.t_std,
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
    
    # E2: Check TCN receptive field is sufficient
    kernel_size = CONFIG["tcn_kernel_size"]
    num_levels = len(CONFIG["tcn_channels"])
    receptive_field = 1 + sum(2 * (kernel_size - 1) * (2 ** i) for i in range(num_levels))
    print(f"TCN receptive field: {receptive_field}, Sequence length: {seq_length}")
    if receptive_field < seq_length:
        print(f"WARNING: Receptive field ({receptive_field}) < sequence length ({seq_length})")
        print("  Model may not see full sequence. Consider increasing TCN depth or kernel size.")
        print("  However, with weighted pooling, this may be acceptable as pooling can aggregate information.")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    weight_decay = CONFIG.get("weight_decay", 0.0)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    # Note: Cannot use SequentialLR with ReduceLROnPlateau (requires metric in step())
    # Instead, manually handle warmup and plateau separately
    warmup_epochs = CONFIG.get("warmup_epochs", 0)
    scheduler_factor = CONFIG.get("scheduler_factor", 0.7)
    scheduler_patience = CONFIG.get("scheduler_patience", 10)
    scheduler_min_lr = CONFIG.get("scheduler_min_lr", 1e-6)
    
    # Main scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
        verbose=False  # Set to False to avoid deprecation warning
    )
    
    # Store warmup info for manual handling
    use_warmup = (warmup_epochs > 0)
    base_lr = CONFIG["lr"]  # Store base LR for warmup calculation
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
        
        # Handle warmup manually (before plateau scheduler)
        if use_warmup and epoch < warmup_epochs:
            # Linear warmup: LR = base_lr * (epoch + 1) / warmup_epochs
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            new_lr = warmup_lr
        else:
            # After warmup, use plateau scheduler
            scheduler.step(val_loss)  # Plateau scheduler steps on validation loss
            new_lr = optimizer.param_groups[0]['lr']
        
        # Print message if learning rate changed
        if old_lr != new_lr:
            if use_warmup and epoch < warmup_epochs:
                epoch_pbar.write(f"  → Learning rate (warmup): {old_lr:.2e} → {new_lr:.2e}")
            else:
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
