"""
Early inference model for parameter estimation from pH prefix sequences.

Architecture: TCN (Temporal Convolutional Network) with uncertainty support.
Takes fixed-length window of pH samples + known inputs and outputs parameter estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
    - TCN processes pH sequence
    - MLP processes known inputs
    - Concatenate and pass through output head
    - Outputs mean and log-variance for uncertainty quantification
    """
    
    def __init__(
        self,
        seq_length: int,
        n_known_inputs: int = 6,
        n_output_params: int = 4,
        tcn_channels: list = [64, 128, 256],
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        mlp_hidden_dims: list = [128, 64],
        output_dropout: float = 0.1,
        use_uncertainty: bool = True,
    ):
        super(EarlyInferenceModel, self).__init__()
        
        self.seq_length = seq_length
        self.n_known_inputs = n_known_inputs
        self.n_output_params = n_output_params
        self.use_uncertainty = use_uncertainty
        
        # TCN for pH sequence processing
        # Input: (batch, 1, seq_len) - pH values
        self.tcn = TCN(
            num_inputs=1,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )
        tcn_output_dim = tcn_channels[-1]
        
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
        known_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Parameters
        ----------
        pH_seq: (batch_size, seq_length) tensor of pH values
        known_inputs: (batch_size, n_known_inputs) tensor of known inputs
        
        Returns
        -------
        mean: (batch_size, n_output_params) tensor of parameter means
        logvar: (batch_size, n_output_params) tensor of log-variances (if use_uncertainty)
        """
        batch_size = pH_seq.shape[0]
        
        # Process pH sequence with TCN
        # Reshape: (batch, seq_len) -> (batch, 1, seq_len)
        pH_seq = pH_seq.unsqueeze(1)  # (batch, 1, seq_len)
        
        tcn_out = self.tcn(pH_seq)  # (batch, tcn_channels[-1], seq_len)
        tcn_pooled = self.pool(tcn_out).squeeze(-1)  # (batch, tcn_channels[-1])
        
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
    
    def predict(self, pH_seq: torch.Tensor, known_inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict parameters (returns mean only).
        
        Parameters
        ----------
        pH_seq: (batch_size, seq_length) tensor
        known_inputs: (batch_size, n_known_inputs) tensor
        
        Returns
        -------
        params: (batch_size, n_output_params) tensor
        """
        mean, _ = self.forward(pH_seq, known_inputs)
        return mean
    
    def sample(self, pH_seq: torch.Tensor, known_inputs: torch.Tensor, 
               n_samples: int = 1) -> torch.Tensor:
        """
        Sample parameters from predicted distribution (if uncertainty enabled).
        
        Parameters
        ----------
        pH_seq: (batch_size, seq_length) tensor
        known_inputs: (batch_size, n_known_inputs) tensor
        n_samples: number of samples to draw
        
        Returns
        -------
        samples: (batch_size, n_samples, n_output_params) tensor
        """
        if not self.use_uncertainty:
            mean = self.predict(pH_seq, known_inputs)
            return mean.unsqueeze(1).expand(-1, n_samples, -1)
        
        mean, logvar = self.forward(pH_seq, known_inputs)
        std = torch.exp(0.5 * logvar)
        batch_size = mean.shape[0]
        
        # Sample
        eps = torch.randn(batch_size, n_samples, self.n_output_params, 
                         device=mean.device, dtype=mean.dtype)
        samples = mean.unsqueeze(1) + std.unsqueeze(1) * eps
        
        return samples


def create_early_inference_model(
    seq_length: int,
    n_known_inputs: int = 6,
    n_output_params: int = 4,
    tcn_channels: list = [64, 128, 256],
    tcn_kernel_size: int = 3,
    tcn_dropout: float = 0.2,
    mlp_hidden_dims: list = [128, 64],
    output_dropout: float = 0.1,
    use_uncertainty: bool = True,
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
    )
    return model


def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, 
                     target: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.
    
    Parameters
    ----------
    mean: (batch_size, n_params) predicted means
    logvar: (batch_size, n_params) predicted log-variances
    target: (batch_size, n_params) target values
    
    Returns
    -------
    loss: scalar tensor
    """
    precision = torch.exp(-logvar)
    return 0.5 * (logvar + precision * (target - mean) ** 2).mean()
