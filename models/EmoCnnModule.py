import sys  # Importing the sys module to handle system-specific parameters and functions
import torch  # Importing PyTorch for deep learning functionalities
import pytorch_lightning as pl  # Importing PyTorch Lightning for training utilities
import numpy as np  # Importing NumPy for numerical computations
from utils.other import class_from_str  # Importing a function from a custom module
import torch.nn.functional as F  # Importing PyTorch's functional interface
from omegaconf import DictConfig, OmegaConf  # Importing OmegaConf for handling configuration settings
from pytorch_lightning.loggers import WandbLogger  # Importing the WandbLogger for logging with Weights & Biases
from affectnet.AffectNetModule import AffectNetExpressions  # Importing a class from a custom module
from affectnet.AffectNetDataset import Expression7  # Importing a class from a custom module
from pathlib import Path  # Importing Path from pathlib for handling file paths
from utils.lightning_logging import _log_array_image, _log_wandb_image, _torch_image2np  # Importing functions from a custom module
from models.EmoRecogModule import EmotionRecognitionBaseModule  # Importing a class from a custom module
import torchvision.models.vgg as vgg  # Importing VGG models from torchvision
from losses.FRNet import resnet50, load_state_dict  # Importing functions from a custom module
from torch.nn import Linear  # Importing Linear layer from torch.nn
import pytorch_lightning.plugins.environments.lightning_environment as le  # Importing a module for lightning environment


class EmoCnnModule(EmotionRecognitionBaseModule):
    """
    Emotion CNN module for emotion recognition.
    """

    def __init__(self, config):
        """
        Initializes the EmoCnnModule.
        Args:
            config: Configuration object.
        """
        super().__init__(config)
        self.n_expression = config.data.n_expression if 'n_expression' in config.data.keys() else 9

        self.num_outputs = 0
        if self.config.model.predict_expression:
            self.num_outputs += self.n_expression
            self.num_classes = self.n_expression

        if self.config.model.predict_valence:
            self.num_outputs += 1

        if self.config.model.predict_arousal:
            self.num_outputs += 1

        if 'predict_AUs' in self.config.model.keys() and self.config.model.predict_AUs:
            self.num_outputs += self.config.model.predict_AUs

        if config.model.backbone == "resnet50":
            self.backbone = resnet50(num_classes=8631, include_top=False)
            if config.model.load_pretrained:
                load_state_dict(self.backbone, config.model.pretrained_weights)
            self.last_feature_size = 2048
            self.linear = Linear(self.last_feature_size, self.num_outputs)
        elif config.model.backbone[:3] == "vgg":
            vgg_constructor = getattr(vgg, config.model.backbone)
            self.backbone = vgg_constructor(pretrained=bool(config.model.load_pretrained), progress=True)
            self.last_feature_size = 1000
            self.linear = Linear(self.last_feature_size, self.num_outputs) 
        else:
            raise ValueError(f"Invalid backbone: '{self.config.model.backbone}'")

    def get_last_feature_size(self):
        """
        Get the size of the last feature.
        Returns:
            int: Size of the last feature.
        """
        return self.last_feature_size

    def _forward(self, images):
        """
        Forward pass through the network.
        Args:
            images: Input images.
        Returns:
            dict: Dictionary containing various outputs.
        """
        output = self.backbone(images)
        emo_feat_2 = output
        output = self.linear(output.view(output.shape[0], -1))

        out_idx = 0
        if self.predicts_expression():
            expr_classification = output[:, out_idx:(out_idx + self.n_expression)]
            if self.exp_activation is not None:
                expr_classification = self.exp_activation(expr_classification, dim=1)
            out_idx += self.n_expression
        else:
            expr_classification = None

        if self.predicts_valence():
            valence = output[:, out_idx:(out_idx + 1)]
            if self.v_activation is not None:
                valence = self.v_activation(valence)
            out_idx += 1
        else:
            valence = None

        if self.predicts_arousal():
            arousal = output[:, out_idx:(out_idx + 1)]
            if self.a_activation is not None:
                arousal = self.a_activation(arousal)
            out_idx += 1
        else:
            arousal = None

        if self.predicts_AUs():
            num_AUs = self.config.model.predict_AUs
            AUs = output[:, out_idx:(out_idx + num_AUs)]
            if self.AU_activation is not None:
                AUs = self.AU_activation(AUs)
            out_idx += num_AUs
        else:
            AUs = None

        assert out_idx == output.shape[1]

        values = {}
        values["emo_feat_2"] = emo_feat_2
        values["valence"] = valence
        values["arousal"] = arousal
        values["expr_classification"] = expr_classification
        values["AUs"] = AUs
        return values

    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: Input batch.
        Returns:
            dict: Dictionary containing various outputs.
        """
        images = batch['image']

        if len(images.shape) == 5:
            K = images.shape[1]
        elif len(images.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        emotion = self._forward(images)

        valence = emotion['valence']
        arousal = emotion['arousal']

        values = {}
        if self.predicts_valence():
            values['valence'] = valence.view(-1,1)
        if self.predicts_arousal():
            values['arousal'] = arousal.view(-1,1)
        values['expr_classification'] = emotion['expr_classification']
        if self.predicts_AUs():
            values['AUs'] = emotion['AUs']

        if 'n_expression' not in self.config.data:
            if self.n_expression == 8:
                raise NotImplementedError("This here should not be called")
                values['expr_classification'] = torch.cat([
                    values['expr_classification'], torch.zeros_like(values['expr_classification'][:, 0:1])
                                                   + 2*values['expr_classification'].min()],
                    dim=1)

        return values

    def _get_trainable_parameters(self):
        """
        Get the trainable parameters of the model.
        Returns:
            list: List of trainable parameters.
        """
        return list(self.backbone.parameters())

    def _vae_2_str(self, valence=None, arousal=None, affnet_expr=None, expr7=None, prefix=""):
        """
        Convert valence and arousal to string.
        Args:
            valence: Valence value.
            arousal: Arousal value.
            affnet_expr: Affnet expression value.
            expr7: Expression7 value.
            prefix: Prefix for the string.
        Returns:
            str: Converted string.
        """
        caption = ""
        if len(prefix) > 0:
            prefix += "_"
        if valence is not None and not np.isnan(valence).any():
            caption += prefix + "valence= %.03f\n" % valence
        if arousal is not None and not np.isnan(arousal).any():
            caption += prefix + "arousal= %.03f\n" % arousal
        if affnet_expr is not None and not np.isnan(affnet_expr).any():
            caption += prefix + "expression= %s \n" % AffectNetExpressions(affnet_expr).name
        if expr7 is not None and not np.isnan(expr7).any():
            caption += prefix +"expression= %s \n" % Expression7(expr7).name
        return caption

    def _test_visualization(self, output_values, input_batch, batch_idx, dataloader_idx=None):
        """
        Perform visualization during testing.
        Args:
            output_values: Output values.
            input_batch: Input batch.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
        Returns:
            dict: Dictionary containing visualization data.
        """
        return None
        batch_size = input_batch['image'].shape[0]

        visdict = {}
        if 'image' in input_batch.keys():
            visdict['inputs'] = input_batch['image']

        valence_pred = output_values["valence"]
        arousal_pred = output_values["arousal"]
        expr_classification_pred = output_values["expr_classification"]

        valence_gt = input_batch["va"][:, 0:1]
        arousal_gt = input_batch["va"][:, 1:2]
        expr_classification_gt = input_batch["affectnetexp"]

        stage = "test"
        vis_dict = {}

        if self.trainer.is_global_zero:
            if isinstance(self.logger, WandbLogger):
                caption = self._vae_2_str(
                    valence=valence_pred.detach().cpu().numpy()[0],
                    arousal=arousal_pred.detach().cpu().numpy()[0],
                    affnet_expr=torch.argmax(expr_classification_pred, dim=1).detach().cpu().numpy().astype(np.int32)[0],
                    expr7=None, prefix="pred")
                caption += self._vae_2_str(
                    valence=valence_gt.cpu().numpy()[0],
                    arousal=arousal_gt.cpu().numpy()[0],
                    affnet_expr=expr_classification_gt.cpu().numpy().astype(np.int32)[0],
                    expr7=None, prefix="gt")

            i = 0 
            for key in visdict.keys():
                images = _torch_image2np(visdict[key])
                savepath = Path(
                    f'{self.config.inout.full_run_dir}/{stage}/{key}/{self.current_epoch:04d}_{batch_idx:04d}_{i:02d}.png')
                image = images[i]
                if isinstance(self.logger, WandbLogger):
                    im2log = _log_wandb_image(savepath, image, caption)
                elif self.logger is not None:
                    im2log = _log_array_image(savepath, image, caption)
                else:
                    im2log = _log_array_image(None, image, caption)
                name = stage + "_" + key
                if dataloader_idx is not None:
                    name += "/dataloader_idx_" + str(dataloader_idx)
                vis_dict[name] = im2log

            if isinstance(self.logger, WandbLogger):
                self.logger.log_metrics(vis_dict)
        return vis_dict
