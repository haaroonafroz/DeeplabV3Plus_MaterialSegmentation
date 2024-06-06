from yacs.config import CfgNode as CN

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values for the project.
    Returns:
        CfgNode: Default configuration object.
    """
    _C = CN()

    # Data settings
    _C.DATA = CN()
    _C.DATA.DATASET = 'VOCSegmentation'
    _C.DATA.ROOT = './data/VOCdevkit/VOC2012'

    # Model settings
    _C.MODEL = CN()
    _C.MODEL.NUM_CLASSES = 21
    #_C.MODEL.NUM_BLOCKS = [3, 3, 9, 3]
    #_C.MODEL.HIDDEN_DIMS = [96, 192, 384, 768]

    # Training settings
    _C.TRAIN = CN()
    _C.TRAIN.BASE_LR = 0.001
    _C.TRAIN.MILESTONES = [10, 20]
    _C.TRAIN.GAMMA = 0.1
    _C.TRAIN.BATCH_SIZE = 64
    _C.TRAIN.NUM_EPOCHS = 50
    _C.TRAIN.EARLY_STOPPING = False

    # Augmentation settings
    _C.AUGMENTATION = CN()
    _C.AUGMENTATION.HORIZONTAL_FLIP_PROB = 0.3
    _C.AUGMENTATION.ROTATION_DEGREES = 10

    # Miscellaneous settings
    _C.MISC = CN()
    _C.MISC.RUN_NAME = 'default_run'
    _C.MISC.RESULTS_CSV = './results'
    _C.MISC.SAVE_MODEL_PATH = './saved_models'
    _C.MISC.PRETRAINED_WEIGHTS = ''
    _C.MISC.FROZEN_LAYERS = []
    _C.MISC.NO_CUDA = False

    return _C.clone()

def get_cfg_from_file(cfg_file):
    """
    Load configuration from a file.
    Args:
        cfg_file (str): Path to the configuration file.
    Returns:
        CfgNode: Configuration object.
    """
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.freeze()
    return cfg