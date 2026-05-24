import torch
from torch import nn, optim

class TwinsBarlow(nn.Module):
    def __init__(self, args, backbone=None):
        """
        Twins Barlow model constructor.

        Args:
            args: Model arguments.
            backbone: Backbone model (default is None).
        """
        super().__init__()
        self.args = args
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        self.bt_loss = BarlowTwinsLoss(self.args)

    def forward(self, y1, y2):
        """
        Forward pass of the model.

        Args:
            y1: Input tensor 1.
            y2: Input tensor 2.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = self.bt_loss(self.backbone(y1), self.backbone(y2))
        return loss

class BarlowTwinsLoss(nn.Module):
    def __init__(self, feature_size=2048, layer_sizes=None, final_reduction='mean_on_diag'):
        """
        Barlow Twins loss constructor.

        Args:
            feature_size (int): Feature size.
            layer_sizes (list): List of layer sizes (default is None).
            final_reduction (str): Final reduction operation (default is 'mean_on_diag').
        """
        super().__init__()
        if layer_sizes is None:
            layer_sizes = 3*[8192]
        sizes = [feature_size] + layer_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1])) 
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bt_loss_headless = BarlowTwinsLossHeadless(sizes[-1], final_reduction=final_reduction)

    def forward(self, y1, y2, batch_size=None, ring_size=None):
        """
        Forward pass of the loss function.

        Args:
            y1: Input tensor 1.
            y2: Input tensor 2.
            batch_size: Batch size (default is None).
            ring_size: Ring size (default is None).

        Returns:
            torch.Tensor: Loss value.
        """
        if self.projector is not None:
            z1 = self.projector(y1)
            z2 = self.projector(y2)
        else:
            z1 = y1
            z2 = y2
        loss = self.bt_loss_headless(z1, z2, batch_size=batch_size, ring_size=ring_size)
        return loss

class BarlowTwinsLossHeadless(nn.Module):
    def __init__(self, feature_size, batch_size=None, lambd=0.005, final_reduction='mean_on_diag'):
        """
        Barlow Twins loss headless constructor.

        Args:
            feature_size (int): Feature size.
            batch_size: Batch size (default is None).
            lambd (float): Lambda parameter (default is 0.005).
            final_reduction (str): Final reduction operation (default is 'mean_on_diag').
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(feature_size, affine=False)
        self.lambd = lambd
        self.batch_size = batch_size
        if final_reduction not in ["sum", "mean", "mean_on_diag", "mean_off_diag"]:
            raise ValueError(f"Invalid reduction operation for Barlow Twins: '{self.final_reduction}'")
        self.final_reduction = final_reduction

    def forward(self, z1, z2, batch_size=None, ring_size=None):
        """
        Forward pass of the loss function.

        Args:
            z1: Input tensor 1.
            z2: Input tensor 2.
            batch_size: Batch size (default is None).
            ring_size: Ring size (default is None).

        Returns:
            torch.Tensor: Loss value.
        """
        assert not (batch_size is not None and self.batch_size is not None)
        if ring_size is not None and ring_size > 1:
            raise NotImplementedError("Barlow Twins with rings are not yet supported.")
        if batch_size is None:
            if self.batch_size is not None:
                batch_size = self.batch_size
            else:
                print("[WARNING] Batch size for Barlow Twins loss not explicitly set. "
                      "This can make problems in multi-gpu training.")
                batch_size = z1.shape[0]
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(batch_size)
        if torch.distributed.is_initialized():
            torch.distributed.nn.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2)
        off_diag = off_diagonal(c).pow_(2)
        if self.final_reduction == 'sum':
            on_diag = on_diag.sum()
            off_diag = off_diag.sum()
        elif self.final_reduction == 'mean':
            on_diag = on_diag.mean()
            off_diag = off_diag.mean()
        elif self.final_reduction == 'mean_on_diag':
            n = on_diag.numel()
            on_diag = on_diag.mean()
            off_diag = off_diag.sum() / n
        elif self.final_reduction == 'mean_off_diag':
            n = off_diag.numel()
            on_diag = on_diag.sum() / n
            off_diag = off_diag.mean()
        else:
            raise ValueError(f"Invalid reduction operation for Barlow Twins: '{self.final_reduction}'")
        loss = on_diag + self.lambd * off_diag
        return loss

def off_diagonal(x):
    """
    Get the off-diagonal elements of a matrix.

    Args:
        x: Input matrix.

    Returns:
        torch.Tensor: Off-diagonal elements.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:]
