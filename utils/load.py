import sys
from pathlib import Path

from omegaconf import OmegaConf

from models.DREAM import DecaModule
from models.IO import locate_checkpoint
from utils.DreamTest import prepare_data
from utils.other import get_path_to_assets


def hack_paths(cfg, replace_root_path=None, relative_to_path=None):
    """
    Modify paths in the configuration file to be relative to a different root directory.

    Args:
        cfg (OmegaConf): The configuration to be modified.
        replace_root_path (str): The new root path to replace.
        relative_to_path (str): The path to which the paths should be made relative.

    Returns:
        OmegaConf: The modified configuration.
    """
    if relative_to_path is not None and replace_root_path is not None:
        # Modify paths in the configuration
        cfg.model.flame_model_path = '/home/abbas/dream/assets/FLAME/geometry/generic_model.pkl'
        cfg.model.flame_lmk_embedding_path = '/home/abbas/dream/assets/FLAME/geometry/landmark_embedding.npy'
        cfg.model.tex_path = '/home/abbas/dream/assets/FLAME/texture/FLAME_albedo_from_BFM.npz'
        cfg.model.topology_path = '/home/abbas/dream/assets/FLAME/geometry/head_template.obj'
        cfg.model.face_mask_path = '/home/abbas/dream/assets/FLAME/mask/uv_face_mask.png'
        cfg.model.face_eye_mask_path = '/home/abbas/dream/assets/FLAME/mask/uv_face_eye_mask.png'
        cfg.model.fixed_displacement_path = '/home/abbas/dream/assets/FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy'
        cfg.model.pretrained_vgg_face_path = '/home/abbas/dream/assets/FaceRecognition/resnet50_ft_weight.pkl'
        cfg.model.pretrained_modelpath = '/home/abbas/dream/assets/DECA/data/deca_model.tar'
        
        # Make data root path relative to a different root directory
        if cfg.data.data_root is not None:
            cfg.data.data_root = str(Path(replace_root_path) / Path(cfg.data.data_root).relative_to(relative_to_path))
        try:
            cfg.inout.full_run_dir = str(Path(replace_root_path) / Path(cfg.inout.full_run_dir).relative_to(relative_to_path))
        except ValueError as e:
            print(f"Skipping hacking full_run_dir {cfg.inout.full_run_dir} because it does not start with '{relative_to_path}'")
    return cfg


def load_deca(conf,
              stage,
              mode,
              relative_to_path=None,
              replace_root_path=None,
              terminate_on_failure=True,
              ):
    """
    Load the DECA model.

    Args:
        conf (OmegaConf): The configuration.
        stage (str): The stage of DECA model.
        mode (str): The mode of DECA model ('best' or 'latest').
        relative_to_path (str): The path to make paths relative to.
        replace_root_path (str): The new root path to replace.
        terminate_on_failure (bool): Whether to terminate if the model loading fails.

    Returns:
        DecaModule: The loaded DECA model.
    """
    print(f"Taking config of stage '{stage}'")
    print(conf.keys())
    if stage is not None:
        cfg = conf[stage]
    else:
        cfg = conf
    if relative_to_path is not None and replace_root_path is not None:
        cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)
    cfg.model.resume_training = False

    checkpoint = locate_checkpoint(cfg, replace_root_path, relative_to_path, mode=mode)
    if checkpoint is None:
        if terminate_on_failure:
            sys.exit(0)
        else:
            return None
    print(f"Loading checkpoint '{checkpoint}'")

    checkpoint_kwargs = {
        "model_params": cfg.model,
        "learning_params": cfg.learning,
        "inout_params": cfg.inout,
        "stage_name": "testing",
    }
    deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return deca


def load_deca_and_data(path_to_models=None,
                       run_name=None,
                       stage=None,
                       relative_to_path=None,
                       replace_root_path=None,
                       mode='best',
                       load_data=True):

    """
    Load the DECA model and associated data.

    Args:
        path_to_models (str): Path to the DECA models.
        run_name (str): Name of the run.
        stage (str): Stage of DECA model.
        relative_to_path (str): Path to make paths relative to.
        replace_root_path (str): The new root path to replace.
        mode (str): The mode of DECA model ('best' or 'latest').
        load_data (bool): Whether to load data associated with the DECA model.

    Returns:
        Union[DecaModule, Tuple[DecaModule, Any]]: Loaded DECA model with or without associated data.
    """
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)

    if stage is None:
        cfg = conf.detail
        cfg.model.resume_training = True
        if relative_to_path is not None and replace_root_path is not None:
            cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)
        deca = DecaModule(cfg.model, cfg.learning, cfg.inout, "testing")
        deca.deca._load_old_checkpoint()
    else:
        deca = load_deca(
            conf,
            stage,
            mode,
            relative_to_path,
            replace_root_path
        )
        cfg = conf[stage]

    train_or_test = 'test'  # Set to 'train' or 'test' based on your requirement
    mode = train_or_test == 'train'
    prefix = stage
    deca.reconfigure(cfg.model, cfg.inout, prefix, downgrade_ok=True, train=mode)
    deca.cuda()
    deca.eval()
    print("DREAM loaded")
    if not load_data:
        return deca
    dm, name = prepare_data(cfg)
    dm.setup()
    return deca, dm


def replace_asset_dirs(cfg, output_dir: Path):
    """
    Replace asset directories in the configuration.

    Args:
        cfg (OmegaConf): The configuration.
        output_dir (Path): The output directory.

    Returns:
        OmegaConf: The modified configuration.
    """
    asset_dir = get_path_to_assets()

    for mode in ["coarse", "detail"]:
        cfg[mode].inout.output_dir = str(output_dir.parent)
        cfg[mode].inout.full_run_dir = str(output_dir / mode)
        cfg[mode].inout.checkpoint_dir = str(output_dir / mode / "checkpoints")

        cfg[mode].model.tex_path = str(asset_dir / "FLAME/texture/FLAME_albedo_from_BFM.npz")
        cfg[mode].model.topology_path = str(asset_dir / "FLAME/geometry/head_template.obj")
        cfg[mode].model.fixed_displacement_path = str(asset_dir / "FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy")
        cfg[mode].model.flame_model_path = str(asset_dir / "FLAME/geometry/generic_model.pkl")
        cfg[mode].model.flame_lmk_embedding_path = str(asset_dir / "FLAME/geometry/landmark_embedding.npy")
        cfg[mode].model.flame_mediapipe_lmk_embedding_path = str(asset_dir / "FLAME/geometry/mediapipe_landmark_embedding.npz")
        cfg[mode].model.face_mask_path = str(asset_dir / "FLAME/mask/uv_face_mask.png")
        cfg[mode].model.face_eye_mask_path = str(asset_dir / "FLAME/mask/uv_face_eye_mask.png")
        cfg[mode].model.pretrained_modelpath = str(asset_dir / "DECA/data/deca_model.tar")
        cfg[mode].model.pretrained_vgg_face_path = str(asset_dir / "FaceRecognition/resnet50_ft_weight.pkl") 
        cfg[mode].model.emonet_model_path = ""  # Set the path as needed
    
    return cfg


def load_model(path_to_models,
              run_name,
              stage,
              relative_to_path=None,
              replace_root_path=None,
              mode='best',
              allow_stage_revert=False,
              ):
    """
    Load the DECA model.

    Args:
        path_to_models (str): Path to the DECA models.
        run_name (str): Name of the run.
        stage (str): Stage of DECA model.
        relative_to_path (str): Path to make paths relative to.
        replace_root_path (str): The new root path to replace.
        mode (str): The mode of DECA model ('best' or 'latest').
        allow_stage_revert (bool): Whether to allow reverting to an earlier stage if loading fails.

    Returns:
        Tuple[DecaModule, OmegaConf]: Loaded DECA model and its configuration.
    """
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)

    conf = replace_asset_dirs(conf, Path(path_to_models) / run_name)
    conf.coarse.checkpoint_dir = str(Path(path_to_models) / run_name / "coarse" / "checkpoints")
    conf.coarse.full_run_dir = str(Path(path_to_models) / run_name / "coarse")
    conf.coarse.output_dir = str(Path(path_to_models))
    conf.detail.checkpoint_dir = str(Path(path_to_models) / run_name / "detail" / "checkpoints")
    conf.detail.full_run_dir = str(Path(path_to_models) / run_name / "detail")
    conf.detail.output_dir = str(Path(path_to_models))
    deca = load_deca(conf,
                     stage,
                     mode,
                     relative_to_path,
                     replace_root_path,
                     terminate_on_failure=not allow_stage_revert
                     )
    if deca is None and allow_stage_revert:
        deca = load_deca(conf,
                         "coarse",
                         mode,
                         relative_to_path,
                         replace_root_path,
                         )

    return deca, conf
