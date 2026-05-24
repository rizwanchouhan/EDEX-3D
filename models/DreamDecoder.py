import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for generating images from random noise.
    """

    def __init__(self, latent_dim=100, out_channels=1, out_scale=1, sample_mode='bilinear'):
        """
        Initializes the Generator network.
        Args:
            latent_dim (int): Dimension of the input noise vector.
            out_channels (int): Number of output channels in the generated images.
            out_scale (float): Scale factor for the output images.
            sample_mode (str): Upsampling mode.
        """
        super(Generator, self).__init__()
        self.out_scale = out_scale

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 64
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 128
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode=sample_mode),  # 256
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        """
        Forward pass of the Generator network.
        Args:
            z (torch.Tensor): Input noise tensor.

        Returns:
            torch.Tensor: Generated image tensor.
        """
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img * self.out_scale


from losses.AdaIN import AdaIN


class AdaInUpConvBlock(nn.Module):
    """
    AdaIN (Adaptive Instance Normalization) Upsampling Convolutional Block.
    """

    def __init__(self, dim_in, dim_out, cond_dim, kernel_size=3, scale_factor=2, sample_mode='bilinear'):
        """
        Initializes the AdaInUpConvBlock.
        Args:
            dim_in (int): Number of input channels.
            dim_out (int): Number of output channels.
            cond_dim (int): Dimension of the conditioning vector.
            kernel_size (int): Size of the convolutional kernel.
            scale_factor (int): Upsampling scale factor.
            sample_mode (str): Upsampling mode.
        """
        super().__init__()
        self.norm = AdaIN(cond_dim, dim_in)
        self.actv = nn.LeakyReLU(0.2, inplace=True)
        if scale_factor > 0:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=sample_mode)
        else:
            self.upsample = None
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride=1, padding=1)

    def forward(self, x, condition):
        """
        Forward pass of the AdaInUpConvBlock.
        Args:
            x (torch.Tensor): Input tensor.
            condition (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.norm(x, condition)
        x = self.actv(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv(x)
        return x


class GeneratorAdaIn(nn.Module):
    """
    Generator network using AdaIN for conditional image generation.
    """

    def __init__(self, latent_dim, condition_dim, out_channels=1, out_scale=1, sample_mode='bilinear'):
        """
        Initializes the GeneratorAdaIn network.
        Args:
            latent_dim (int): Dimension of the input noise vector.
            condition_dim (int): Dimension of the conditioning vector.
            out_channels (int): Number of output channels in the generated images.
            out_scale (float): Scale factor for the output images.
            sample_mode (str): Upsampling mode.
        """
        super().__init__()
        self.out_scale = out_scale

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_block1 = AdaInUpConvBlock(128, 128, condition_dim, sample_mode=sample_mode)
        self.conv_block2 = AdaInUpConvBlock(128, 64, condition_dim, sample_mode=sample_mode)
        self.conv_block3 = AdaInUpConvBlock(64, 64, condition_dim, sample_mode=sample_mode)
        self.conv_block4 = AdaInUpConvBlock(64, 32, condition_dim, sample_mode=sample_mode)
        self.conv_block5 = AdaInUpConvBlock(32, 16, condition_dim, sample_mode=sample_mode)
        self.conv_block6 = AdaInUpConvBlock(16, out_channels, condition_dim, scale_factor=0)
        self.conv_blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4,
                            self.conv_block5, self.conv_block6]
        self.out_actv = nn.Tanh()

    def forward(self, z, cond):
        """
        Forward pass of the GeneratorAdaIn network.
        Args:
            z (torch.Tensor): Input noise tensor.
            cond (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Generated image tensor.
        """
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        for i, block in enumerate(self.conv_blocks):
            out = block(out, cond)
        img = self.out_actv(out)
        return img * self.out_scale
