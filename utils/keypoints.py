import torch  # Import PyTorch library
import numpy as np  # Import NumPy library


class KeypointTransform(torch.nn.Module):
    """
    Base class for keypoint transformations.
    """

    def __init__(self, scale_x=1., scale_y=1.):
        super().__init__()
        self.scale_x = scale_x  # Scale factor for x-axis
        self.scale_y = scale_y  # Scale factor for y-axis

    def set_scale(self, scale_x=1., scale_y=1.):
        """Set scale factors."""
        self.scale_x = scale_x  # Set scale factor for x-axis
        self.scale_y = scale_y  # Set scale factor for y-axis

    def forward(self, points):
        """Forward pass. Subclasses must override this method."""
        raise NotImplementedError()


class KeypointScale(KeypointTransform):
    """
    Class for scaling keypoints.
    """

    def __init__(self, scale_x=1., scale_y=1.):
        super().__init__(scale_x, scale_y)

    def forward(self, points):
        """Scale keypoints."""
        points_ = points.clone()
        points_[..., 0] *= self.scale_x  # Scale x-coordinates
        points_[..., 1] *= self.scale_y  # Scale y-coordinates
        return points_


class KeypointNormalization(KeypointTransform):
    """
    Class for normalizing keypoints.
    """

    def __init__(self, scale_x=1., scale_y=1.):
        super().__init__(scale_x, scale_y)

    def forward(self, points):
        """Normalize keypoints."""
        if isinstance(points, torch.Tensor):
            points_ = points.clone()
        elif isinstance(points, np.ndarray):
            points_ = points.copy()
        else:
            raise ValueError(f"Invalid type of points {str(type(points))}")
        points_[..., 0] -= self.scale_x / 2  # Shift x-coordinates
        points_[..., 0] /= self.scale_x / 2  # Scale x-coordinates
        points_[..., 1] -= self.scale_y / 2  # Shift y-coordinates
        points_[..., 1] /= self.scale_y / 2  # Scale y-coordinates
        return points_

    def inv(self, points):
        """Inverse normalization of keypoints."""
        if isinstance(points, torch.Tensor):
            points_ = points.clone()
        elif isinstance(points, np.ndarray):
            points_ = points.copy()
        else:
            raise ValueError(f"Invalid type of points {str(type(points))}")
        points_[..., 0] *= self.scale_x / 2  # Scale x-coordinates
        points_[..., 0] += self.scale_x / 2  # Shift x-coordinates
        points_[..., 1] *= self.scale_y / 2  # Scale y-coordinates
        points_[..., 1] += self.scale_y / 2  # Shift y-coordinates
        return points_
