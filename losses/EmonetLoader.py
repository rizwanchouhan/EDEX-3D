import inspect
import sys
from pathlib import Path
from utils.other import get_path_to_externals
import torch

def get_emonet(device=None, load_pretrained=True):
    """
    Function to get the EmoNet model instance.

    Args:
        device (torch.device): Device to use for model inference.
        load_pretrained (bool): Whether to load pretrained weights or create an untrained instance.

    Returns:
        EmoNet: EmoNet model instance.
    """
    # Set device
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Add EmoNet path to sys.path if not already present
    path_to_emonet = get_path_to_externals() / "emonet"
    if not (str(path_to_emonet) in sys.path or str(path_to_emonet.absolute()) in sys.path):
        sys.path += [str(path_to_emonet)]

    # Import EmoNet class
    from affectnet.emonet import EmoNet
    n_expression = 8

    # Initialize EmoNet instance
    net = EmoNet(n_expression=n_expression).to(device)

    # Load pretrained weights if specified
    state_dict_path = Path(inspect.getfile(EmoNet)).parent.parent.parent / 'pretrained' / f'emonet_{n_expression}.pth'
    print(f'Loading the EmoNet model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=False)

    # Create an untrained instance if load_pretrained is False
    if not load_pretrained:
        print("Created an untrained EmoNet instance")
        net.reset_emo_parameters()

    net.eval()
    return net
