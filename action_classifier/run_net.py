import os
import argparse
from slowfast.utils.parser import parse_args
from train_utils import train
from config_utils import pirate_load_cfg

import sys

def parse_args():
    """
    Modified from pySlowfast utils/parser.py
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Start training from a pretrained pytorchvideo checkpoint (on Kinetics unless SSv2 is specified)'
    )
    parser.add_argument(
        '--ssv2',
        action='store_true',
        help='Start training from a pytorchvideo checkpoint pretrained on SomethingSomethingV2 dataset'   
    )
    parser.add_argument(
        '--i3d',
        action='store_true',
        help='Train an I3D model'
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    cfg = pirate_load_cfg(args=args)
    train(cfg, pretrained=args.pretrained, i3d=args.i3d, ssv2=args.ssv2)