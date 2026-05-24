import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # Instance normalization layer (IN) without learnable parameters
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # Fully connected layer to compute gamma and beta parameters for AdaIN
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        # Applying the fully connected layer to style vector s
        h = self.fc(s)
        # Reshaping the output to split into gamma and beta
        h = h.view(h.size(0), h.size(1), 1, 1)
        # Splitting gamma and beta
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # Applying AdaIN normalization
        return (1 + gamma) * self.norm(x) + beta
