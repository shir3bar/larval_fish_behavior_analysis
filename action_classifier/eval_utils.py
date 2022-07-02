from slowfast.utils import metrics
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd
from dataset import construct_loader
import os
import argparse
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config

def eval_epoch(model, data_loader, cfg):
    model.eval()
    num_correct = 0
    y_hats = []
    all_labels = []
    stats = {}
    running_err = 0
    all_preds = []
    all_file_names = []
    file_list = data_loader.dataset.dataset._labeled_videos._paths_and_labels
    with torch.no_grad():
        for cur_iter, (inputs, labels, indices, meta) in enumerate(data_loader):
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            preds = model(inputs)
            preds = torch.nn.functional.softmax(preds, dim=1)
            file_names = [file_list[f][0] for f in indices]
            all_file_names += file_names
            y_hat = preds.max(axis=1).indices
            all_preds.append(preds)
            y_hats.append(y_hat)
            all_labels.append(labels)
            num_correct += (labels == y_hat).sum()
            k = min(cfg.MODEL.NUM_CLASSES, 5)  # in case there aren't at least 5 classes in the dataset
            num_topks_correct = metrics.topks_correct(preds, labels, (1, k))
            # Combine the errors across the GPUs.
            top1_err, _ = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            top1_err = top1_err.item()
            running_err += top1_err * preds.size(0)
        all_labels = torch.hstack(all_labels)
        y_hats = torch.hstack(y_hats)
        all_preds = torch.vstack(all_preds)

        tn, fp, fn, tp = confusion_matrix(1 - all_labels.cpu(), 1 - y_hats.cpu()).ravel()
        stats['fns'] = fn
        stats['fps'] = fp
        stats['tns'] = tn
        stats['tps'] = tp
        stats['top1_err'] = (running_err / len(data_loader.dataset))
        stats['accuracy'] = (num_correct / len(data_loader.dataset)).item()
    return all_labels, y_hats, stats, all_preds, all_file_names


def load_checkpoint(path,cfg):
    checkpoint = torch.load(path)
    model = torch.hub.load("facebookresearch/pytorchvideo:main", model="slowfast_r50", pretrained=False)
    model.blocks[6].proj = torch.nn.Linear(in_features=2304, out_features=cfg.MODEL.NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model



def get_split_results(model, cfg, split, epoch,save=False):
    loader = construct_loader(cfg,split)
    labels, y_hats, stats, preds, file_names = eval_epoch(model, loader, cfg)
    results_df = pd.DataFrame({'split':[split]* len(file_names), 'file_name': file_names,
                                     'strike_scores': preds[:, 0].cpu(),
                                     'strike_labels': 1 - (labels).cpu()})
    if save:
        file_path = os.path.join(cfg.OUTPUT_DIR,f'results_{split}_{epoch}.csv')
        results_df.to_csv(file_path)
    return results_df

def pirate_load_cfg(cfg_path):
    class Args:
        def __init__(self, cfg_file):
            self.cfg_file = cfg_file
            self.shard_id = 0
            self.num_shards = 1
            self.init_method = 'tcp://localhost:9999'
            self.opts = None

    args = Args(cfg_path)
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    return cfg

def get_epoch_results(checkpoint_dir, epoch, cfg_path):
    cfg = pirate_load_cfg(cfg_path)
    checkpoint_path = os.path.join(checkpoint_dir, f'pretrained_epoch{epoch}.pt')
    model = load_checkpoint(checkpoint_path, cfg)
    splits = ['train_eval','val_eval','test']
    results = []
    for split in splits:
        df = get_split_results(model, cfg, split, save=False,epoch=epoch)
        results.append(df)
    all_results = pd.concat(results)
    results_path = os.path.join(cfg.OUTPUT_DIR, f'strike_scores_all_splits_epoch{epoch}.csv')
    print(f'Saving results file at {results_path}')
    all_results.to_csv(results_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', help='enter checkpoint path')
    parser.add_argument('cfg_path', help='enter cfg file path here')
    parser.add_argument('-epoch', type=int, default=10, help='which epoch to evaluate')
    return parser

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    get_epoch_results(checkpoint_dir=args.checkpoint_dir,epoch=args.epoch, cfg_path=args.cfg_path)