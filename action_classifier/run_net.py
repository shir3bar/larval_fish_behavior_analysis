import os

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from train_utils import train

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)

    cfg = assert_and_infer_cfg(cfg)
    train(cfg, pretrained=True)