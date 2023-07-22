from eval_utils import get_epoch_results,eval_alt_testset
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', help='enter checkpoint path')
    parser.add_argument('cfg', help='enter cfg file path here')
    parser.add_argument('--epoch', type=int, default=10, help='which epoch to evaluate')
    parser.add_argument('--plot', action='store_true', help='save ROC/PRC plots')
    parser.add_argument('--alt_testset_dir', type=str, default=None,
                        help='optionally, evaluate on an alternative test dataset')
    parser.add_argument('--i3d',  action='store_true', help='Evaluate an I3D model')
    return parser

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    if args.alt_testset_dir is None:
        # evaluate the model on the original train/val/test splits
        get_epoch_results(checkpoint_dir=args.checkpoint_dir,
                          epoch=args.epoch, cfg=args.cfg,
                          plot=args.plot, i3d=args.i3d)
    else:
        # evaluate on a different dataset, we assume it has one partition called 'test'
        eval_alt_testset(checkpoint_dir=args.checkpoint_dir,
                          epoch=args.epoch, cfg=args.cfg,
                         testset_path=args.alt_testset_dir,
                         plot=args.plot, i3d=args.i3d)