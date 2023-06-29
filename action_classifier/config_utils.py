import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import get_cfg
from slowfast.config.defaults import assert_and_infer_cfg

def add_custom_config(cfg):
    # Add your own customized configs.
    cfg.DATA.BRIGHTNESS_PROB = 0.3
    cfg.DATA.BRIGHTNESS_RATIO = 0.2
    cfg.DATA.BLUR_PROB = 0.2
    cfg.DATA.VARIANCE_IMG = False
    cfg.DATA.VAR_DIM = 1
    cfg.TRAIN.EVAL_DATASET = False
    cfg.MODEL.SSV2_PRETRAINED = False
    return cfg

def get_args(cfg_path):
    class Args:
        def __init__(self, cfg_file):
            self.cfg_file = cfg_file
            self.shard_id = 0
            self.num_shards = 1
            self.init_method = 'tcp://localhost:9999'
            self.opts = None

    return Args(cfg_path)
    
def pirate_load_cfg(cfg_path=None,args=None):
    """
    Adapted from the pySlowFast load_cfg function, to allow custom config without tinkering with their code
    """
    
    if args is None:
        args = get_args(cfg_path)
    cfg = get_cfg()
    cfg = add_custom_config(cfg)
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg
    cfg = assert_and_infer_cfg(cfg)

    return cfg