import torch  # Importing PyTorch for deep learning functionalities
import torch.nn.functional as F  # Importing PyTorch's functional interface
from losses.EmonetLoader import get_emonet  # Importing a function from a custom module


class EmoNetRegressor(torch.nn.Module):
    """
    EmotionNet regressor module for regression tasks.
    """

    def __init__(self, outsize, last_op=None):
        """
        Initializes the EmoNetRegressor.
        Args:
            outsize (int): Size of the output.
            last_op: Last operation.
        """
        super().__init__()
        self.emonet = get_emonet().eval()
        self.input_image_size = (256, 256) 

        self.feature_to_use = 'emo_feat_2'

        if self.feature_to_use == 'emo_feat_2':
            self.emonet_feature_size = 256
            self.fc_size = 256
        else:
            raise NotImplementedError(f"Not yet implemented for feature '{self.feature_to_use}'")

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.emonet_feature_size, self.fc_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.fc_size, outsize)
        )
        self.last_op = last_op

    def forward(self, images):
        """
        Forward pass through the network.
        Args:
            images: Input images.
        Returns:
            torch.Tensor: Output tensor.
        """
        images = F.interpolate(images, self.input_image_size, mode='bilinear')
        out = self.emonet(images, intermediate_features=True)
        out = self.layers(out[self.feature_to_use])
        return out


class EmonetRegressorStatic(EmoNetRegressor):
    """
    Static version of EmoNetRegressor.
    """

    def __init__(self, outsize, last_op=None):
        """
        Initializes the EmonetRegressorStatic.
        Args:
            outsize (int): Size of the output.
            last_op: Last operation.
        """
        super().__init__(outsize, last_op)
        self.emonet.requires_grad_(False)
        self.emonet.eval()

    def train(self, mode=True):
        """
        Set the module in training mode.
        Args:
            mode (bool): Whether to set the module in training mode.
        Returns:
            EmonetRegressorStatic: The module.
        """
        self.emonet.eval()
        self.layers.train(mode)
        return self
