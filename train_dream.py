# Import necessary libraries and modules
from affectnet.EmotionsModule import AffWild2DataModule
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
import datetime

# Define the project name
project_name = 'EmotionalDeca'

# Function to prepare data
def prepare_data(cfg):
    """
    Prepares the data module and sequence name for training.

    Args:
        cfg (DictConfig): Configuration settings.

    Returns:
        (DecaDataModule, str): Data module and sequence name.
    """
    dm = DecaDataModule(cfg)
    sequence_name = "ClassicDECA"
    return dm, sequence_name

# Function to create experiment name
def create_experiment_name(cfg_coarse_pre, cfg_coarse, cfg_detail, version=1):
    """
    Creates an experiment name based on configuration settings.

    Args:
        cfg_coarse_pre (DictConfig): Configuration settings for coarse pretraining.
        cfg_coarse (DictConfig): Configuration settings for coarse model.
        cfg_detail (DictConfig): Configuration settings for detail model.
        version (int, optional): Version of the experiment. Defaults to 1.

    Returns:
        str: Experiment name.
    """
    experiment_name = "DECA_"
    if version <= 1:
        experiment_name = experiment_name.replace("/", "_")  # Replacing '/' with '_' in experiment name
        # Append experiment name based on configurations
        if cfg_coarse.model.use_emonet_loss and cfg_detail.model.use_emonet_loss:
            experiment_name += '_EmoLossB'
        elif cfg_coarse.model.use_emonet_loss:
            experiment_name += '_EmoLossC'
        elif cfg_detail.model.use_emonet_loss:
            experiment_name += '_EmoLossD'
        if cfg_coarse.model.use_emonet_loss or cfg_detail.model.use_emonet_loss:
            experiment_name += '_'
            if cfg_coarse.model.use_emonet_feat_1:
                experiment_name += 'F1'
            if cfg_coarse.model.use_emonet_feat_2:
                experiment_name += 'F2'
            if cfg_coarse.model.use_emonet_valence:
                experiment_name += 'V'
            if cfg_coarse.model.use_emonet_arousal:
                experiment_name += 'A'
            if cfg_coarse.model.use_emonet_expression:
                experiment_name += 'E'
            if cfg_coarse.model.use_emonet_combined:
                experiment_name += 'C'

        if cfg_coarse.model.use_emonet_loss or cfg_detail.model.use_emonet_loss:
            experiment_name += 'w-%.05f' % cfg_coarse.model.emonet_weight

        # Appending encoder types to the experiment name
        e_flame_type = 'ResnetEncoder'
        if 'e_flame_type' in cfg_coarse.model.keys():
            e_flame_type = cfg_coarse.model.e_flame_type
        if e_flame_type != 'ResnetEncoder':
            experiment_name += "_EF" + e_flame_type[:6]

        e_detail_type = 'ResnetEncoder'
        if 'e_detail_type' in cfg_detail.model.keys():
            e_detail_type = cfg_detail.model.e_detail_type
        if e_detail_type != 'ResnetEncoder':
            experiment_name += "_ED" + e_detail_type[:6]

        # Appending supervised emotion loss type to the experiment name
        if cfg_coarse.model.use_gt_emotion_loss and cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossB'
        elif cfg_coarse.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossC'
        elif cfg_detail.model.use_gt_emotion_loss:
            experiment_name += '_SupervisedEmoLossD'

        # Appending segmentation and detail type to the experiment name
        if version == 0:
            if cfg_coarse.model.useSeg:
                experiment_name += '_CoSegGT'
            else:
                experiment_name += '_CoSegRend'

            if cfg_detail.model.useSeg:
                experiment_name += '_DeSegGT'
            else:
                experiment_name += '_DeSegRend'

        if cfg_detail.model.useSeg:
            experiment_name += f'_DeSeg{cfg_detail.model.useSeg}'
        else:
            experiment_name += f'_DeSeg{cfg_detail.model.useSeg}'

        # Appending detail L1 and MRF options to the experiment name
        if not cfg_detail.model.use_detail_l1:
            experiment_name += '_NoDetL1'
        if not cfg_detail.model.use_detail_mrf:
            experiment_name += '_NoMRF'

        # Appending background type to the experiment name
        if not cfg_coarse.model.background_from_input and not cfg_detail.model.background_from_input:
            experiment_name += '_BlackB'
        elif not cfg_coarse.model.background_from_input:
            experiment_name += '_BlackC'
        elif not cfg_detail.model.background_from_input:
            experiment_name += '_BlackD'

        # Appending learning rates to the experiment name
        if version == 0:
            if cfg_coarse.learning.learning_rate != 0.0001:
                experiment_name += f'CoLR-{cfg_coarse.learning.learning_rate}'
            if cfg_detail.learning.learning_rate != 0.0001:
                experiment_name += f'DeLR-{cfg_detail.learning.learning_rate}'

        # Appending additional model options to the experiment name
        if version == 0:
            if cfg_coarse.model.use_photometric:
                experiment_name += 'CoPhoto'
            if cfg_coarse.model.use_landmarks:
                experiment_name += 'CoLMK'
            if cfg_coarse.model.idw:
                experiment_name += f'_IDW-{cfg_coarse.model.idw}'

        # Appending shape and detail constrain types to the experiment name
        if cfg_coarse.model.shape_constrain_type != 'exchange':
            experiment_name += f'_Co{cfg_coarse.model.shape_constrain_type}'
        if cfg_detail.model.detail_constrain_type != 'exchange':
            experiment_name += f'_De{cfg_coarse.model.detail_constrain_type}'

        # Appending augmentation flag to the experiment name
        if 'augmentation' in cfg_coarse.data.keys() and len(cfg_coarse.data.augmentation) > 0:
            experiment_name += "_Aug"

        # Appending train coarse flag to the experiment name
        if cfg_detail.model.train_coarse:
            experiment_name += "_DwC"

        # Appending early stopping flag to the experiment name
        if hasattr(cfg_coarse.learning, 'early_stopping') and cfg_coarse.learning.early_stopping \
            and hasattr(cfg_detail.learning, 'early_stopping') and cfg_detail.learning.early_stopping:
            experiment_name += "_early"

    return experiment_name


def train_dream(cfg_coarse_pretraining, cfg_coarse, cfg_detail, start_i=-1, resume_from_previous=True,
               force_new_location=False):
    """
    Trains the DECA model with specified configurations.

    Args:
        cfg_coarse_pretraining (DictConfig): Configuration settings for coarse pretraining.
        cfg_coarse (DictConfig): Configuration settings for coarse model.
        cfg_detail (DictConfig): Configuration settings for detail model.
        start_i (int, optional): Starting stage index. Defaults to -1.
        resume_from_previous (bool, optional): Whether to resume training from the previous checkpoint. Defaults to True.
        force_new_location (bool, optional): Whether to force training to occur in a new location. Defaults to False.
    """
    # Prepare configurations and stage information
    configs = [cfg_coarse_pretraining, cfg_coarse_pretraining, cfg_coarse, cfg_coarse, cfg_detail, cfg_detail]
    stages = ["train", "test", "train", "test", "train", "test"]
    stages_prefixes = ["pretrain", "pretrain", "", "", "", ""]

    # Check if starting from a specific stage or with a new location
    if start_i >= 0 or force_new_location:
        if resume_from_previous:
            resume_i = start_i - 1
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the next stage {start_i})")
        else:
            resume_i = start_i
            print(f"Resuming checkpoint from stage {resume_i} (and will start from the same stage {start_i})")
        checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(configs[resume_i], stages_prefixes[resume_i])
    else:
        checkpoint, checkpoint_kwargs = None, None

    # Determine the output directory and experiment name
    if cfg_coarse.inout.full_run_dir == 'todo' or force_new_location:
        if force_new_location:
            print("The run will be resumed in a new folder (forked)")
            cfg_coarse.inout.previous_run_dir = cfg_coarse.inout.full_run_dir
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        experiment_name = create_experiment_name(cfg_coarse_pretraining, cfg_coarse, cfg_detail)
        full_run_dir = Path(configs[0].inout.output_dir) / (time + "_" + experiment_name)
        exist_ok = True
    else:
        experiment_name = cfg_coarse.inout.name
        len_time_str = len(datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S"))
        if hasattr(cfg_coarse.inout, 'time') and cfg_coarse.inout.time is not None:
            time = cfg_coarse.inout.time
        else:
            time = experiment_name[:len_time_str]
        full_run_dir = Path(cfg_coarse.inout.full_run_dir).parent
        exist_ok = True

    # Create necessary directories and update configurations
    full_run_dir.mkdir(parents=True, exist_ok=exist_ok)
    print(f"The run will be saved  to: '{str(full_run_dir)}'")
    with open("out_folder.txt", "w") as f:
        f.write(str(full_run_dir))

    coarse_pretrain_checkpoint_dir = full_run_dir / "coarse_pretrain" / "checkpoints"
    coarse_pretrain_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse_pretraining.inout.full_run_dir = str(coarse_pretrain_checkpoint_dir.parent)
    cfg_coarse_pretraining.inout.checkpoint_dir = str(coarse_pretrain_checkpoint_dir)
    cfg_coarse_pretraining.inout.name = experiment_name
    cfg_coarse_pretraining.inout.time = time

    coarse_checkpoint_dir = full_run_dir / "coarse" / "checkpoints"
    coarse_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_coarse.inout.full_run_dir = str(coarse_checkpoint_dir.parent)
    cfg_coarse.inout.checkpoint_dir = str(coarse_checkpoint_dir)
    cfg_coarse.inout.name = experiment_name
    cfg_coarse.inout.time = time

    detail_checkpoint_dir = full_run_dir / "detail" / "checkpoints"
    detail_checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    cfg_detail.inout.full_run_dir = str(detail_checkpoint_dir.parent)
    cfg_detail.inout.checkpoint_dir = str(detail_checkpoint_dir)
    cfg_detail.inout.name = experiment_name
    cfg_detail.inout.time = time

    # Save configurations
    conf = DictConfig({})
    conf.coarse_pretraining = cfg_coarse_pretraining
    conf.coarse = cfg_coarse
    conf.detail = cfg_detail
    with open(full_run_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=conf, f=outfile)

    # Create a logger
    wandb_logger = create_logger(
                         cfg_coarse_pretraining.learning.logger_type,
                         name=experiment_name,
                         project_name=project_name,
                         config=OmegaConf.to_container(conf),
                         version=time,
                         save_dir=full_run_dir)

    # Initialize DECA model
    dream = None
    if start_i >= 0 or force_new_location:
        print(f"Loading a checkpoint: {checkpoint} and starting from stage {start_i}")
    if start_i == -1:
        start_i = 0

    # Train DECA model
    for i in range(start_i, len(configs)):
        cfg = configs[i]
        dream = single_stage_dream_pass(dream, cfg, stages[i], stages_prefixes[i], dm=None, logger=wandb_logger,
                                      data_preparation_function=prepare_data,
                                      checkpoint=checkpoint, checkpoint_kwargs=checkpoint_kwargs
                                      )
        checkpoint = None


def configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
              coarse_cfg_default, coarse_overrides,
              detail_cfg_default, detail_overrides):
    """
    Configures the DECA model with specified default settings and overrides.

    Args:
        coarse_pretrain_cfg_default (str): Default configuration file name for coarse pretraining.
        coarse_pretrain_overrides (list of str): List of overrides for coarse pretraining.
        coarse_cfg_default (str): Default configuration file name for coarse model.
        coarse_overrides (list of str): List of overrides for coarse model.
        detail_cfg_default (str): Default configuration file name for detail model.
        detail_overrides (list of str): List of overrides for detail model.

    Returns:
        Tuple[DictConfig, DictConfig, DictConfig]: Configurations for coarse pretraining, coarse model, and detail model.
    """
    # Compose configurations using Hydra
    from hydra.experimental import compose, initialize
    initialize(config_path="../dream_conf", job_name="train_dream")
    cfg_coarse_pretrain = compose(config_name=coarse_pretrain_cfg_default, overrides=coarse_pretrain_overrides)
    cfg_coarse = compose(config_name=coarse_cfg_default, overrides=coarse_overrides)
    cfg_detail = compose(config_name=detail_cfg_default, overrides=detail_overrides)
    return cfg_coarse_pretrain, cfg_coarse, cfg_detail


def configure_and_train(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                        coarse_cfg_default, coarse_overrides,
                        detail_cfg_default, detail_overrides):
    """
    Configures and trains the DECA model with specified default settings and overrides.

    Args:
        coarse_pretrain_cfg_default (str): Default configuration file name for coarse pretraining.
        coarse_pretrain_overrides (list of str): List of overrides for coarse pretraining.
        coarse_cfg_default (str): Default configuration file name for coarse model.
        coarse_overrides (list of str): List of overrides for coarse model.
        detail_cfg_default (str): Default configuration file name for detail model.
        detail_overrides (list of str): List of overrides for detail model.
    """
    # Configure and train DECA model
    cfg_coarse_pretrain, cfg_coarse, cfg_detail = configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)
    train_dream(cfg_coarse_pretrain, cfg_coarse, cfg_detail)


def configure_and_resume(run_path,
                         coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                         coarse_cfg_default, coarse_overrides,
                         detail_cfg_default, detail_overrides,
                         start_at_stage):
    """
    Configures and resumes training of the DECA model from a specified stage.

    Args:
        run_path (str): Path to the previous run.
        coarse_pretrain_cfg_default (str): Default configuration file name for coarse pretraining.
        coarse_pretrain_overrides (list of str): List of overrides for coarse pretraining.
        coarse_cfg_default (str): Default configuration file name for coarse model.
        coarse_overrides (list of str): List of overrides for coarse model.
        detail_cfg_default (str): Default configuration file name for detail model.
        detail_overrides (list of str): List of overrides for detail model.
        start_at_stage (int): Stage to resume training from.
    """
    # Configure and resume training of DECA model
    cfg_coarse_pretrain, cfg_coarse, cfg_detail = configure(coarse_pretrain_cfg_default, coarse_pretrain_overrides,
                                       coarse_cfg_default, coarse_overrides,
                                       detail_cfg_default, detail_overrides)

    cfg_coarse_pretrain_, cfg_coarse_, cfg_detail_ = load_configs(run_path)

    if start_at_stage < 2:
        raise RuntimeError("Resuming before stage 2 makes no sense, that would be training from scratch")
    if start_at_stage == 2:
        cfg_coarse_pretrain = cfg_coarse_pretrain_
    elif start_at_stage == 3:
        raise RuntimeError("Resuming for stage 3 makes no sense, that is a testing stage")
    elif start_at_stage == 4:
        cfg_coarse_pretrain = cfg_coarse_pretrain_
        cfg_coarse = cfg_coarse_
    elif start_at_stage == 5:
        raise RuntimeError("Resuming for stage 5 makes no sense, that is a testing stage")
    else:
        raise RuntimeError(f"Cannot resume at stage {start_at_stage}")

    train_dream(cfg_coarse_pretrain, cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=True, 
               force_new_location=True)


def load_configs(run_path):
    """
    Loads configurations from a previous run.

    Args:
        run_path (str): Path to the previous run.

    Returns:
        Tuple[DictConfig, DictConfig, DictConfig]: Configurations for coarse pretraining, coarse model, and detail model.
    """
    # Load configurations from the previous run
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    cfg_coarse_pretraining = conf.coarse_pretraining
    cfg_coarse = conf.coarse
    cfg_detail = conf.detail
    return cfg_coarse_pretraining, cfg_coarse, cfg_detail


def resume_training(run_path, start_at_stage, resume_from_previous, force_new_location):
    """
    Resumes training of the DECA model from a specified stage.

    Args:
        run_path (str): Path to the previous run.
        start_at_stage (int): Stage to resume training from.
        resume_from_previous (bool): Whether to resume training from the previous checkpoint.
        force_new_location (bool): Whether to force training to occur in a new location.
    """
    # Load configurations and resume training
    cfg_coarse_pretraining, cfg_coarse, cfg_detail = load_configs(run_path)
    train_dream(cfg_coarse_pretraining, cfg_coarse, cfg_detail,
               start_i=start_at_stage,
               resume_from_previous=resume_from_previous,
               force_new_location=force_new_location)


def main():
    """
    Main function to run DECA training or resume training from a previous run.
    """
    configured = False
    if len(sys.argv) >= 4:
        if Path(sys.argv[1]).is_file():
            configured = True
            with open(sys.argv[1], 'r') as f:
                coarse_pretrain_conf = OmegaConf.load(f)
            with open(sys.argv[2], 'r') as f:
                coarse_conf = OmegaConf.load(f)
            with open(sys.argv[3], 'r') as f:
                detail_conf = OmegaConf.load(f)
        else:
            coarse_pretrain_conf = sys.argv[1]
            coarse_conf = sys.argv[2]
            detail_conf = sys.argv[3]
    else:
        coarse_pretrain_conf = "dream_train_coarse_pretrain"
        coarse_conf = "dream_train_coarse"
        detail_conf = "dream_train_detail"
        nr = "none"
        logger = "wandb"

        coarse_pretrain_override = [
                                    'learning/batching=single_gpu_coarse_pretrain_32gb',
                                    f'learning/logging={logger}',
                                    ]
        
        coarse_override = [
                            'learning/batching=single_gpu_coarse',
                           f'model/neural_rendering={nr}',
                            f'learning/logging={logger}',
                           ]

        detail_override = [
                            'learning/batching=single_gpu_detail',
                           f'model/neural_rendering={nr}',
                            f'learning/logging={logger}',
                            f'model.background_from_input=False',
                           ]


    if len(sys.argv) >= 7:
        coarse_pretrain_override = sys.argv[4]
        coarse_override = sys.argv[5]
        detail_override = sys.argv[6]

    if configured:
        train_dream(coarse_pretrain_conf, coarse_conf, detail_conf)
    else:
        configure_and_train(coarse_pretrain_conf, coarse_pretrain_override,
                            coarse_conf, coarse_override, detail_conf, detail_override)


if __name__ == "__main__":
    main()

