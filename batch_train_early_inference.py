"""
Batch training script for running multiple training configurations.

This script allows you to define multiple training configurations and run them sequentially.
Each configuration will be saved in a separate folder with parameters in the folder name.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import itertools

# Import the main training function
from train_early_inference import main as train_main, CONFIG as BASE_CONFIG

# ╔══════════════════════════════════════════════════════════════╗
# ║                   BATCH TRAINING CONFIG                       ║
# ╚══════════════════════════════════════════════════════════════╝

# Base data directory
BASE_DATA_DIR = r"C:\Users\vt4ho\Simulations\simulation_data\generated_data\imperfect\version_2\Generated_Data_EarlyInference_100000"
BASE_OUTPUT_DIR = r"C:\Users\vt4ho\Simulations\simulation_data\models\imperfect\version_2\batch_ML_training_models"

# Define parameter variations
TRAINING_CONFIGS = {
    # Prefix lengths to train on
    "prefix_lengths": [30.0, 120.0],
    
    # Training hyperparameters variations
    "batch_sizes": [512],
    "learning_rates": [2e-3, 5e-3],
    "epochs": [1000],
    "early_stopping_patience": [20],
    "warmup_epochs": [0, 5],  # Test with/without warmup
    
    # Model architecture variations
    "tcn_channels": [
        [128, 256, 512, 512],
        #[128, 256, 512, 512, 512],  # Deeper network
    ],
    "tcn_kernel_sizes": [7],
    
    # Optional features
    "use_weighted_pooling": [False, True],  # Test both pooling methods
    "use_variance_regularization": [False, True],  # Test with/without variance reg
}

# Fixed parameters (same for all runs)
FIXED_CONFIG = {
    "val_split": 0.2,
    "weight_decay": 1e-5,  # Keep - good
    "grad_clip_norm": 5.0,  # Keep - good
    "scheduler_factor": 0.7,  # Keep - good
    "scheduler_patience": 15,  # Keep - good
    "scheduler_min_lr": 1e-5,  # Keep - good
    # Note: warmup_epochs is now in TRAINING_CONFIGS for variation
    
    # REDUCE DROPOUT (main change)
    "tcn_dropout": 0.1,  # Reduced from 0.2 - less underfitting risk
    "output_dropout": 0.05,  # Reduced from 0.1 - less aggressive
    
    "mlp_hidden_dims": [256, 128],  # Keep - good architecture
    
    "use_uncertainty": True,  # Keep - good
    "variance_penalty_weight": 0.01,  # Keep - only used if enabled
    "target_variance": 0.1,  # Keep - only used if enabled
    
    "device": "auto",
    "normalize_inputs": True,
    "normalize_outputs": True,
    "seed": 42,
}


def generate_config_name(config):
    """Generate a folder name from configuration parameters."""
    parts = []
    
    # Add prefix length
    parts.append(f"pref{int(config['prefix_length'])}s")
    
    # Add batch size
    parts.append(f"bs{config['batch_size']}")
    
    # Add learning rate (scientific notation without e)
    lr_str = f"{config['lr']:.0e}".replace("e-0", "e").replace("e-", "em")
    parts.append(f"lr{lr_str}")
    
    # Add TCN channels (abbreviated)
    tcn_str = "_".join(str(c) for c in config['tcn_channels'])
    parts.append(f"tcn{len(config['tcn_channels'])}l")
    
    # Add kernel size
    parts.append(f"ks{config['tcn_kernel_size']}")
    
    # Add optional features
    if config.get("use_weighted_pooling", False):
        parts.append("wpool")
    if config.get("use_variance_regularization", False):
        parts.append("vreg")
    
    return "_".join(parts)


def generate_all_configs():
    """Generate all combinations of training configurations."""
    configs = []
    
    # Generate all combinations
    keys = list(TRAINING_CONFIGS.keys())
    values = list(TRAINING_CONFIGS.values())
    
    for combination in itertools.product(*values):
        config = FIXED_CONFIG.copy()
        config["data_dir"] = BASE_DATA_DIR
        
        # Map combination to config dict
        for key, value in zip(keys, combination):
            # Handle special case: "prefix_lengths" -> "prefix_length" (singular)
            if key == "prefix_lengths":
                config["prefix_length"] = value
            # Handle special case: "learning_rates" -> "lr"
            elif key == "learning_rates":
                config["lr"] = value
            # Handle special case: "batch_sizes" -> "batch_size"
            elif key == "batch_sizes":
                config["batch_size"] = value
            # Handle special case: "tcn_kernel_sizes" -> "tcn_kernel_size"
            elif key == "tcn_kernel_sizes":
                config["tcn_kernel_size"] = value
            # Handle special case: "early_stopping_patience" -> already correct
            else:
                config[key] = value
        
        # Generate output directory name
        config_name = generate_config_name(config)
        config["output_dir"] = str(Path(BASE_OUTPUT_DIR) / config_name)
        
        configs.append((config_name, config))
    
    return configs


def save_config_json(config, output_path):
    """Save configuration to JSON file."""
    # Convert numpy arrays/lists to serializable format
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            config_serializable[key] = list(value)
        elif isinstance(value, Path):
            config_serializable[key] = str(value)
        else:
            config_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)


def run_training(config_name, config):
    """Run training with a specific configuration."""
    print("\n" + "="*80)
    print(f"STARTING TRAINING: {config_name}")
    print("="*80)
    print(f"Configuration:")
    for key in sorted(config.keys()):
        if key not in ["data_dir", "output_dir"]:
            print(f"  {key}: {config[key]}")
    print(f"Output directory: {config['output_dir']}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration JSON
    config_json_path = output_dir / "training_config.json"
    save_config_json(config, config_json_path)
    print(f"Saved configuration to: {config_json_path}")
    
    # Run training with the specific config
    try:
        train_main(config=config)
        print(f"\n✓ Training completed successfully: {config_name}")
        return True
    except Exception as e:
        print(f"\n✗ Training failed: {config_name}")
        print(f"  Error: {str(e)}")
        # Save error to file
        error_file = output_dir / "training_error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Training failed for configuration: {config_name}\n")
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        return False


def main():
    """Main function to run batch training."""
    print("="*80)
    print("BATCH TRAINING FOR EARLY INFERENCE MODEL")
    print("="*80)
    
    # Generate all configurations
    print("\nGenerating training configurations...")
    configs = generate_all_configs()
    
    print(f"\nTotal number of configurations: {len(configs)}")
    print("\nConfigurations to run:")
    for i, (name, config) in enumerate(configs, 1):
        print(f"  {i}. {name}")
        print(f"     Prefix: {config['prefix_length']}s, "
              f"Batch: {config['batch_size']}, "
              f"LR: {config['lr']}, "
              f"Warmup: {config.get('warmup_epochs', 0)} epochs, "
              f"TCN levels: {len(config['tcn_channels'])}, "
              f"Pooling: {'Weighted' if config.get('use_weighted_pooling') else 'Average'}, "
              f"Var Reg: {config.get('use_variance_regularization', False)}")
    
    # Ask for confirmation
    response = input(f"\nProceed with {len(configs)} training runs? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Batch training cancelled.")
        return
    
    # Run all configurations
    successful = []
    failed = []
    
    start_time = datetime.now()
    
    for i, (config_name, config) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing configuration: {config_name}")
        
        success = run_training(config_name, config)
        
        if success:
            successful.append(config_name)
        else:
            failed.append(config_name)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH TRAINING SUMMARY")
    print("="*80)
    print(f"Total configurations: {len(configs)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {duration}")
    print(f"\nSuccessful runs:")
    for name in successful:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed runs:")
        for name in failed:
            print(f"  ✗ {name}")
    
    # Save summary to file
    summary_file = Path(BASE_OUTPUT_DIR) / "batch_training_summary.json"
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "total_configs": len(configs),
        "successful": successful,
        "failed": failed,
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
