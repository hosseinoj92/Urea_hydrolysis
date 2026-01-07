"""
DeepONet (Deep Operator Network) implementation for urease kinetics surrogate.
Architecture: branch network (encodes parameters) + trunk network (encodes time) + inner product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BranchNet(nn.Module):
    """Branch network: encodes parameter vector x into feature space."""
    
    def __init__(self, input_dim: int = 5, hidden_dims: list = [128, 128, 128], output_dim: int = 128):
        super(BranchNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x: (batch_size, input_dim) parameter vector
        
        Returns:
        --------
        features: (batch_size, output_dim) encoded features
        """
        return self.net(x)


class TrunkNet(nn.Module):
    """Trunk network: encodes time t into feature space."""
    
    def __init__(self, input_dim: int = 1, hidden_dims: list = [128, 128, 128], output_dim: int = 128):
        super(TrunkNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        t: (batch_size, input_dim) time vector (can be 1D or 2D)
        
        Returns:
        --------
        features: (batch_size, output_dim) encoded features
        """
        # Handle both (batch_size,) and (batch_size, 1) inputs
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(t)


class DeepONet(nn.Module):
    """
    DeepONet: combines branch and trunk networks via inner product.
    
    Architecture:
    - Branch: x (parameters) → b(x) ∈ R^p
    - Trunk: t (time) → t(t) ∈ R^p
    - Output: G(x, t) = b(x)^T t(t) + bias
    """
    
    def __init__(
        self,
        branch_input_dim: int = 5,
        branch_hidden_dims: list = [128, 128, 128],
        trunk_input_dim: int = 1,
        trunk_hidden_dims: list = [128, 128, 128],
        branch_output_dim: int = 128,
        trunk_output_dim: int = 128,
        output_dim: int = 1,  # For single output (pH or one component of totals)
        use_bias: bool = True
    ):
        super(DeepONet, self).__init__()
        
        assert branch_output_dim == trunk_output_dim, "Branch and trunk output dims must match"
        
        self.branch = BranchNet(branch_input_dim, branch_hidden_dims, branch_output_dim)
        self.trunk = TrunkNet(trunk_input_dim, trunk_hidden_dims, trunk_output_dim)
        self.use_bias = use_bias
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x: (batch_size, branch_input_dim) parameter vector
        t: (batch_size, trunk_input_dim) time vector
        
        Returns:
        --------
        output: (batch_size, output_dim) predicted value
        """
        # Encode parameters and time
        b = self.branch(x)  # (batch_size, branch_output_dim)
        t_enc = self.trunk(t)  # (batch_size, trunk_output_dim)
        
        # Inner product
        output = torch.sum(b * t_enc, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Add bias if used
        if self.use_bias:
            output = output + self.bias
        
        return output


class MultiOutputDeepONet(nn.Module):
    """
    DeepONet for multiple outputs (e.g., S, Ntot, Ctot or pH).
    Uses separate DeepONets for each output.
    """
    
    def __init__(
        self,
        branch_input_dim: int = 5,
        branch_hidden_dims: list = [128, 128, 128],
        trunk_input_dim: int = 1,
        trunk_hidden_dims: list = [128, 128, 128],
        branch_output_dim: int = 128,
        trunk_output_dim: int = 128,
        n_outputs: int = 3,  # e.g., S, Ntot, Ctot
        use_bias: bool = True
    ):
        super(MultiOutputDeepONet, self).__init__()
        
        self.n_outputs = n_outputs
        self.networks = nn.ModuleList([
            DeepONet(
                branch_input_dim=branch_input_dim,
                branch_hidden_dims=branch_hidden_dims,
                trunk_input_dim=trunk_input_dim,
                trunk_hidden_dims=trunk_hidden_dims,
                branch_output_dim=branch_output_dim,
                trunk_output_dim=trunk_output_dim,
                output_dim=1,
                use_bias=use_bias
            )
            for _ in range(n_outputs)
        ])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x: (batch_size, branch_input_dim) parameter vector
        t: (batch_size, trunk_input_dim) time vector
        
        Returns:
        --------
        output: (batch_size, n_outputs) predicted values
        """
        outputs = []
        for net in self.networks:
            outputs.append(net(x, t))
        return torch.cat(outputs, dim=1)  # (batch_size, n_outputs)


def physics_loss_regularizer(
    outputs: torch.Tensor,
    t: torch.Tensor,
    mode: str = "totals"
) -> torch.Tensor:
    """
    Physics-informed regularizers:
    - Nonnegativity: ensure outputs are non-negative (via softplus)
    - Monotonicity: urea should decrease, ammonia should increase
    
    Parameters:
    -----------
    outputs: (batch_size, n_outputs) model outputs
    t: (batch_size, 1) time
    mode: "totals" or "pH"
    
    Returns:
    --------
    loss: scalar regularization term
    """
    loss = 0.0
    
    if mode == "totals":
        # outputs: [S, Ntot, Ctot]
        S = outputs[:, 0:1]  # Urea
        Ntot = outputs[:, 1:2]  # Total ammonia
        Ctot = outputs[:, 2:3]  # Total carbon
        
        # Nonnegativity (softplus ensures positive)
        # Already handled by using softplus activation if needed
        
        # Monotonicity: dS/dt < 0, dNtot/dt > 0 (approximate via finite differences)
        # This is a simplified check - in practice, we'd need sorted time points
        # For now, just penalize negative urea or negative ammonia
        loss += 0.01 * F.relu(-S).mean()  # Penalize negative urea
        loss += 0.01 * F.relu(-Ntot).mean()  # Penalize negative ammonia
        loss += 0.01 * F.relu(-Ctot).mean()  # Penalize negative carbon
    
    elif mode == "pH":
        # pH should be in reasonable range [0, 14]
        pH = outputs[:, 0:1]
        loss += 0.01 * F.relu(pH - 14.0).mean()  # Penalize pH > 14
        loss += 0.01 * F.relu(-pH).mean()  # Penalize pH < 0
    
    return loss


def create_deeponet(
    input_dim: int = 5,
    n_outputs: int = 3,
    hidden_dims: list = [128, 128, 128],
    branch_output_dim: int = 128,
    use_bias: bool = True
) -> nn.Module:
    """
    Factory function to create DeepONet model.
    
    Parameters:
    -----------
    input_dim: dimension of parameter vector (default 5: a, E_eff0, k_d, t_shift, tau_probe)
    n_outputs: number of outputs (1 for pH, 3 for totals)
    hidden_dims: hidden layer dimensions for branch and trunk
    branch_output_dim: output dimension of branch/trunk (must match)
    use_bias: whether to use bias in output
    
    Returns:
    --------
    model: DeepONet or MultiOutputDeepONet
    """
    if n_outputs == 1:
        return DeepONet(
            branch_input_dim=input_dim,
            branch_hidden_dims=hidden_dims,
            trunk_input_dim=1,
            trunk_hidden_dims=hidden_dims,
            branch_output_dim=branch_output_dim,
            trunk_output_dim=branch_output_dim,
            output_dim=1,
            use_bias=use_bias
        )
    else:
        return MultiOutputDeepONet(
            branch_input_dim=input_dim,
            branch_hidden_dims=hidden_dims,
            trunk_input_dim=1,
            trunk_hidden_dims=hidden_dims,
            branch_output_dim=branch_output_dim,
            trunk_output_dim=branch_output_dim,
            n_outputs=n_outputs,
            use_bias=use_bias
        )
