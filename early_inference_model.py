"""
Early inference model for parameter estimation from pH prefix sequences.

Architecture: TCN (Temporal Convolutional Network) with uncertainty support.
Takes fixed-length window of pH samples + known inputs and outputs parameter estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TemporalBlock(nn.Module):
    """Temporal block with dilated convolution, normalization, and dropout."""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, 
            padding=padding, dilation=dilation
        )
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)  # Remove padding from right
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride,
            padding=padding, dilation=dilation
        )
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class WeightedTemporalPooling(nn.Module):
    """
    Weighted temporal pooling that can emphasize early-time information.
    
    Uses learnable attention mechanism to weight different time points,
    allowing the model to focus on early dynamics even in long sequences.
    """
    def __init__(self, dim: int, seq_length: int):
        super(WeightedTemporalPooling, self).__init__()
        self.dim = dim
        self.seq_length = seq_length
        
        # Learnable attention weights
        # Input: TCN output features, Output: attention weights per time step
        self.attention = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(dim // 2, 1, kernel_size=1),
        )
        
        # Optional: Fixed exponential decay weights as initialization bias
        # This encourages the model to initially focus on early time points
        with torch.no_grad():
            decay_weights = torch.exp(-torch.linspace(0, 2, seq_length))
            decay_weights = decay_weights / decay_weights.sum()
            self.register_buffer('decay_weights', decay_weights.view(1, 1, -1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (batch_size, dim, seq_len) tensor
        
        Returns
        -------
        pooled: (batch_size, dim) tensor
        """
        # Compute attention weights
        attn_logits = self.attention(x)  # (batch, 1, seq_len)
        attn_weights = F.softmax(attn_logits, dim=2)  # (batch, 1, seq_len)
        
        # Optional: Blend with decay weights (helps with initialization)
        # attn_weights = 0.5 * attn_weights + 0.5 * self.decay_weights
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=2)  # (batch, dim)
        return pooled


class TCN(nn.Module):
    """Temporal Convolutional Network for sequence processing."""
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: list,
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            ]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (batch_size, n_inputs, seq_len) tensor
        
        Returns
        -------
        output: (batch_size, num_channels[-1], seq_len) tensor
        """
        return self.network(x)


class EarlyInferenceModel(nn.Module):
    """
    Early inference model that estimates parameters from pH prefix sequences.
    
    Architecture:
    - TCN processes pH sequence, time grid, and dt (3-channel input: pH + time + dt)
    - MLP processes known inputs
    - Concatenate and pass through output head
    - Outputs mean and log-variance for uncertainty quantification
    
    The time grid and dt (sampling interval) are critical for learning sampling
    rate differences (1s vs 10s) and temporal dynamics of the pH trajectory.
    dt is computed explicitly as t[i] - t[i-1] to help the model distinguish
    different sampling cadences.
    """
    
    def __init__(
        self,
        seq_length: int,
        n_known_inputs: int = 5,  # substrate_mM, grams_urease_powder, temperature_C, initial_pH, volume_L
        n_output_params: int = 2,  # powder_activity_frac, k_d
        tcn_channels: list = [64, 128, 256],
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        mlp_hidden_dims: list = [128, 64],
        output_dropout: float = 0.1,
        use_uncertainty: bool = True,
        use_weighted_pooling: bool = False,  # Option to use weighted temporal pooling
    ):
        super(EarlyInferenceModel, self).__init__()
        
        self.seq_length = seq_length
        self.n_known_inputs = n_known_inputs
        self.n_output_params = n_output_params
        self.use_uncertainty = use_uncertainty
        
        # TCN for pH sequence processing
        # B3: Input: (batch, 3, seq_len) - pH values, time grid, and dt (sampling interval)
        # Adding dt as explicit feature helps model learn sampling rate differences
        self.tcn = TCN(
            num_inputs=3,  # pH + time + dt
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )
        tcn_output_dim = tcn_channels[-1]
        self.use_weighted_pooling = use_weighted_pooling
        
        # Pooling over sequence dimension (optional: weighted or simple average)
        if use_weighted_pooling:
            # Weighted pooling that emphasizes early-time information
            self.pool = WeightedTemporalPooling(tcn_output_dim, seq_length)
        else:
            # Global pooling over sequence dimension
            self.pool = nn.AdaptiveAvgPool1d(1)  # (batch, tcn_output_dim, 1)
        
        # MLP for known inputs
        mlp_layers = []
        input_dim = n_known_inputs
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(output_dropout))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        mlp_output_dim = mlp_hidden_dims[-1] if mlp_hidden_dims else n_known_inputs
        
        # Combine TCN and MLP outputs
        combined_dim = tcn_output_dim + mlp_output_dim
        
        # Output head
        if use_uncertainty:
            # Output mean and log-variance
            self.mean_head = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(128, n_output_params)
            )
            self.logvar_head = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(128, n_output_params)
            )
        else:
            # Output only mean
            self.output_head = nn.Sequential(
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(128, n_output_params)
            )
    
    def forward(
        self, 
        pH_seq: torch.Tensor,
        t_seq: torch.Tensor,
        known_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Parameters
        ----------
        pH_seq: (batch_size, seq_length) tensor of pH values
        t_seq: (batch_size, seq_length) tensor of time values
        known_inputs: (batch_size, n_known_inputs) tensor of known inputs
        
        Returns
        -------
        mean: (batch_size, n_output_params) tensor of parameter means
        logvar: (batch_size, n_output_params) tensor of log-variances (if use_uncertainty)
        """
        batch_size = pH_seq.shape[0]
        
        # Process pH sequence, time grid, and dt with TCN
        # B3: Compute dt (sampling interval) explicitly as third channel
        # dt[i] = t[i] - t[i-1], with dt[0] = dt[1] (pad first element)
        dt = t_seq[:, 1:] - t_seq[:, :-1]  # (batch, seq_len-1)
        dt_padded = torch.cat([dt[:, 0:1], dt], dim=1)  # (batch, seq_len) - pad first element
        
        # Concatenate pH, time, and dt: (batch, 3, seq_len)
        pH_seq = pH_seq.unsqueeze(1)  # (batch, 1, seq_len)
        t_seq = t_seq.unsqueeze(1)    # (batch, 1, seq_len)
        dt_padded = dt_padded.unsqueeze(1)  # (batch, 1, seq_len)
        seq_input = torch.cat([pH_seq, t_seq, dt_padded], dim=1)  # (batch, 3, seq_len)
        
        tcn_out = self.tcn(seq_input)  # (batch, tcn_channels[-1], seq_len)
        if self.use_weighted_pooling:
            tcn_pooled = self.pool(tcn_out)  # (batch, tcn_channels[-1]) - weighted pooling
        else:
            tcn_pooled = self.pool(tcn_out).squeeze(-1)  # (batch, tcn_channels[-1]) - simple average
        
        # Process known inputs with MLP
        mlp_out = self.mlp(known_inputs)  # (batch, mlp_output_dim)
        
        # Concatenate
        combined = torch.cat([tcn_pooled, mlp_out], dim=1)  # (batch, combined_dim)
        
        # Output
        if self.use_uncertainty:
            mean = self.mean_head(combined)
            logvar = self.logvar_head(combined)
            return mean, logvar
        else:
            mean = self.output_head(combined)
            return mean, None
    
    def predict(self, pH_seq: torch.Tensor, t_seq: torch.Tensor, known_inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict parameters (returns mean only).
        
        Parameters
        ----------
        pH_seq: (batch_size, seq_length) tensor
        t_seq: (batch_size, seq_length) tensor of time values
        known_inputs: (batch_size, n_known_inputs) tensor
        
        Returns
        -------
        params: (batch_size, n_output_params) tensor
        """
        mean, _ = self.forward(pH_seq, t_seq, known_inputs)
        return mean
    
    def sample(self, pH_seq: torch.Tensor, t_seq: torch.Tensor, known_inputs: torch.Tensor, 
               n_samples: int = 1) -> torch.Tensor:
        """
        Sample parameters from predicted distribution (if uncertainty enabled).
        
        Parameters
        ----------
        pH_seq: (batch_size, seq_length) tensor
        t_seq: (batch_size, seq_length) tensor of time values
        known_inputs: (batch_size, n_known_inputs) tensor
        n_samples: number of samples to draw
        
        Returns
        -------
        samples: (batch_size, n_samples, n_output_params) tensor
        """
        if not self.use_uncertainty:
            mean = self.predict(pH_seq, t_seq, known_inputs)
            return mean.unsqueeze(1).expand(-1, n_samples, -1)
        
        mean, logvar = self.forward(pH_seq, t_seq, known_inputs)
        std = torch.exp(0.5 * logvar)
        batch_size = mean.shape[0]
        
        # Sample
        eps = torch.randn(batch_size, n_samples, self.n_output_params, 
                         device=mean.device, dtype=mean.dtype)
        samples = mean.unsqueeze(1) + std.unsqueeze(1) * eps
        
        return samples


def create_early_inference_model(
    seq_length: int,
    n_known_inputs: int = 5,  # substrate_mM, grams_urease_powder, temperature_C, initial_pH, volume_L
    n_output_params: int = 2,  # powder_activity_frac, k_d
    tcn_channels: list = [64, 128, 256],
    tcn_kernel_size: int = 3,
    tcn_dropout: float = 0.2,
    mlp_hidden_dims: list = [128, 64],
    output_dropout: float = 0.1,
    use_uncertainty: bool = True,
    use_weighted_pooling: bool = False,  # Option to use weighted temporal pooling
) -> EarlyInferenceModel:
    """
    Create early inference model.
    
    Parameters
    ----------
    seq_length: length of input pH sequence
    n_known_inputs: number of known input features (e.g., substrate, temperature, etc.)
    n_output_params: number of parameters to predict
    tcn_channels: list of channel dimensions for TCN layers
    tcn_kernel_size: kernel size for TCN
    tcn_dropout: dropout rate for TCN
    mlp_hidden_dims: hidden dimensions for MLP processing known inputs
    output_dropout: dropout rate for output head
    use_uncertainty: if True, output mean and log-variance
    use_weighted_pooling: if True, use learnable attention-based pooling instead of simple average
    
    Returns
    -------
    model: EarlyInferenceModel instance
    """
    model = EarlyInferenceModel(
        seq_length=seq_length,
        n_known_inputs=n_known_inputs,
        n_output_params=n_output_params,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        output_dropout=output_dropout,
        use_uncertainty=use_uncertainty,
        use_weighted_pooling=use_weighted_pooling,
    )
    return model


def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, 
                     target: torch.Tensor, 
                     use_variance_regularization: bool = False,
                     variance_penalty_weight: float = 0.01,
                     target_variance: float = 0.1) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.
    
    D1: Clamp logvar to prevent extreme values that cause negative loss or numerical issues.
    
    Parameters
    ----------
    mean: (batch_size, n_params) predicted means
    logvar: (batch_size, n_params) predicted log-variances
    target: (batch_size, n_params) target values
    use_variance_regularization: if True, adds penalty to encourage reasonable variance
    variance_penalty_weight: weight for variance regularization penalty
    target_variance: target variance value for regularization
    
    Returns
    -------
    loss: scalar tensor
    """
    # D1: Fix to prevent negative loss while preserving gradients
    # The issue: when logvar is very negative (small variance) and error is small,
    # loss = 0.5 * (logvar + precision * error) can be negative
    # Solution: Apply minimum variance floor and include log(2*pi) term
    
    # Apply minimum variance floor: ensures variance >= 0.01
    # This prevents infinite precision while allowing model to learn
    # log(0.01) ≈ -4.6, so clamp logvar to [-4.6, 10]
    min_logvar = -4.6  # Minimum variance of 0.01
    logvar = torch.clamp(logvar, min=min_logvar, max=10.0)
    
    # Compute precision (inverse variance) and squared error
    precision = torch.exp(-logvar)
    sq_error = (target - mean) ** 2
    
    # Full NLL formula: 0.5 * (log(2*pi) + logvar + precision * sq_error)
    # Including log(2*pi) ≈ 1.84 ensures loss is typically positive
    # This is the standard formulation and preserves all gradients
    log_2pi = 1.8378770664093453  # log(2*pi)
    nll = 0.5 * (log_2pi + logvar + precision * sq_error)
    
    # With min_logvar = -4.6 and log_2pi = 1.84:
    # Worst case (logvar=-4.6, error=0): nll = 0.5 * (1.84 - 4.6) = -1.38
    # But in practice, precision * sq_error is usually > 0, so loss is positive
    # If it occasionally goes slightly negative, that's fine for optimization
    # (it just means model is very confident and correct)
    
    loss = nll.mean()
    
    # Optional: Add variance regularization to prevent variance collapse/explosion
    if use_variance_regularization:
        var = torch.exp(logvar)
        # Penalize variance that deviates from target
        variance_penalty = variance_penalty_weight * torch.mean((var - target_variance) ** 2)
        loss = loss + variance_penalty
    
    return loss
