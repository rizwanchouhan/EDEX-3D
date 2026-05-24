from collections import OrderedDict

from torch import nn as nn
from torch.hub import load_state_dict_from_url

class VGG19(nn.Module):
    """
    VGG19 model implementation with customizable layer activation indices.

    Args:
        layer_activation_indices (list): List of indices of the layers whose activations are to be retrieved.
        batch_norm (bool): If True, use batch normalization. Default is False.
    """

    def __init__(self, layer_activation_indices, batch_norm=False):
        super().__init__()
        self.layer_activation_indices = layer_activation_indices
        self.blocks = _vgg('vgg19', 'E', batch_norm=batch_norm, pretrained=True, progress=True)
        self.conv_block_indices = []

        self.layers = []
        for bi, block in enumerate(self.blocks):
            for layer in block:
                self.layers += [layer]
                if isinstance(layer, nn.Conv2d):
                    self.conv_block_indices += [bi]

        if len(self.layer_activation_indices) != len(set(layer_activation_indices).intersection(set(self.conv_block_indices))):
            raise ValueError("The specified layer indices are not of a conv block")

        self.net = nn.Sequential(*self.layers)
        self.net.eval()
        self.net.requires_grad_(False)

    def requires_grad_(self, requires_grad: bool = True):
        """
        Override requires_grad_ to ensure that the model is always in eval mode.

        Args:
            requires_grad (bool): Whether gradients are required. Default is True.

        Returns:
            VGG19: Model with requires_grad set to False.
        """
        return super().requires_grad_(False)

    def train(self, mode: bool = True):
        """
        Override train method to ensure that the model is always in eval mode.

        Args:
            mode (bool): Whether to set the model in training mode or not. Default is True.

        Returns:
            VGG19: Model with training mode set to False.
        """
        return super().train(False)

    def forward(self, x):
        """
        Forward pass of the VGG19 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing layer activations and final output.
        """
        layer_outputs = {}
        for bi, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            if bi in self.layer_activation_indices:
                layer_outputs[bi] = x
        layer_outputs['final'] = x
        return layer_outputs


def make_layers(cfg, batch_norm=False):
    """
    Create layers based on configuration.

    Args:
        cfg (list): Configuration list.
        batch_norm (bool): If True, use batch normalization. Default is False.

    Returns:
        nn.ModuleList: List of layers.
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2)])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [nn.ModuleList([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])]
            else:
                layers += [nn.ModuleList([conv2d, nn.ReLU(inplace=True)])]
            in_channels = v
    return nn.ModuleList(layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    """
    Load VGG model layers based on architecture and configuration.

    Args:
        arch (str): Architecture name.
        cfg (str): Configuration name.
        batch_norm (bool): If True, use batch normalization.
        pretrained (bool): If True, load pre-trained weights.
        progress (bool): If True, display download progress.
        **kwargs: Additional keyword arguments.

    Returns:
        nn.ModuleList: List of VGG layers.
    """
    if pretrained:
        kwargs['init_weights'] = False
    layers = make_layers(cfgs[cfg], batch_norm=batch_norm)
    if pretrained:
        archname = arch
        if batch_norm:
            archname += "_bn"
        state_dict = load_state_dict_from_url(model_urls[archname], progress=progress)
        state_dict2 = OrderedDict()
        for key in state_dict.keys():
            if "features" in key:
                state_dict2[key[len("features."):]] = state_dict[key]
        layers_ = []
        for bi, block in enumerate(layers):
            for layer in block:
                layers_ += [layer]
        net = nn.Sequential(*layers_)
        net.load_state_dict(state_dict2)
    return layers


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
