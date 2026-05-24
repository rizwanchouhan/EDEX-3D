import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import models.ResNet as resnet

try:
    from .Swin import create_swin_backbone, swin_cfg_from_name
except ImportError as e:
    print("SWIN not found, will not be able to use SWIN models")


class BaseEncoder(nn.Module):
    """
    Base class for feature encoders.
    """

    def __init__(self, outsize, last_op=None):
        """
        Initializes the BaseEncoder.
        Args:
            outsize (int): Size of the output vector.
            last_op (callable): Last operation applied to the output vector.
        """
        super().__init__()
        self.feature_size = 2048
        self.outsize = outsize
        self._create_encoder()
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.outsize)
        )
        self.last_op = last_op

    def forward_features(self, inputs):
        """
        Forward pass through the encoder to extract features.
        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        return self.encoder(inputs)

    def forward_features_to_output(self, features):
        """
        Forward pass from features to output.
        Args:
            features (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output tensor.
        """
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

    def forward(self, inputs, output_features=False):
        """
        Forward pass through the encoder.
        Args:
            inputs (torch.Tensor): Input tensor.
            output_features (bool): Whether to output features.

        Returns:
            torch.Tensor: Output tensor or tuple of (output tensor, features).
        """
        features = self.forward_features(inputs)
        parameters = self.forward_features_to_output(features)
        if not output_features:
            return parameters
        return parameters, features

    def _create_encoder(self):
        """
        Abstract method to create the encoder.
        """
        raise NotImplementedError()

    def reset_last_layer(self):
        """
        Resets the parameters of the last layer.
        """
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        torch.nn.init.constant_(self.layers[-1].bias, 0)


class ResnetEncoder(BaseEncoder):
    """
    ResNet feature encoder.
    """

    def __init__(self, outsize, last_op=None):
        """
        Initializes the ResnetEncoder.
        Args:
            outsize (int): Size of the output vector.
            last_op (callable): Last operation applied to the output vector.
        """
        super(ResnetEncoder, self).__init__(outsize, last_op)

    def _create_encoder(self):
        """
        Creates the ResNet encoder.
        """
        self.encoder = resnet.load_ResNet50Model()


class SecondHeadResnet(nn.Module):
    """
    Second head network for ResNet encoder.
    """

    def __init__(self, enc: BaseEncoder, outsize, last_op=None):
        """
        Initializes the SecondHeadResnet.
        Args:
            enc (BaseEncoder): Base encoder.
            outsize (int): Size of the output vector.
            last_op (callable): Last operation applied to the output vector.
        """
        super().__init__()
        self.resnet = enc
        self.layers = nn.Sequential(
            nn.Linear(self.resnet.feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        if last_op == 'same':
            self.last_op = self.resnet.last_op
        else:
            self.last_op = last_op

    def forward_features(self, inputs):
        """
        Forward pass through the encoder to extract features.
        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        out1, features = self.resnet(inputs, output_features=True)
        return out1, features

    def forward_features_to_output(self, features):
        """
        Forward pass from features to output.
        Args:
            features (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output tensor.
        """
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters

    def forward(self, inputs):
        """
        Forward pass through the network.
        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out1, features = self.forward_features(inputs)
        out2 = self.forward_features_to_output(features)
        return out1, out2

    def train(self, mode: bool = True):
        """
        Sets the mode for training.
        Args:
            mode (bool): Whether to set the model to training mode.

        Returns:
            SecondHeadResnet: Self.
        """
        self.layers.train(mode)
        return self

    def reset_last_layer(self):
        """
        Resets the parameters of the last layer.
        """
        torch.nn.init.constant_(self.layers[-1].weight, 0)
        torch.nn.init.constant_(self.layers[-1].bias, 0)


class SwinEncoder(BaseEncoder):
    """
    Swin Transformer feature encoder.
    """

    def __init__(self, swin_type, img_size, outsize, last_op=None):
        """
        Initializes the SwinEncoder.
        Args:
            swin_type (str): Type of Swin Transformer.
            img_size (int): Size of the input image.
            outsize (int): Size of the output vector.
            last_op (callable): Last operation applied to the output vector.
        """
        self.swin_type = swin_type
        self.img_size = img_size
        super().__init__(outsize, last_op)

    def _create_encoder(self):
        """
        Creates the Swin Transformer encoder.
        """
        swin_cfg = swin_cfg_from_name(self.swin_type)
        self.encoder = create_swin_backbone(
            swin_cfg, self.feature_size, self.img_size, load_pretrained_swin=True, pretrained_model=self.swin_type)
