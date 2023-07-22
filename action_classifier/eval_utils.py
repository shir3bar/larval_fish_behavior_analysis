from slowfast.utils import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import torch
import pandas as pd
from slowfast.datasets.loader import construct_loader
from slowfast.datasets.build import DATASET_REGISTRY
from dataset import Ptvfishbase
import os
import numpy as np
from config_utils import pirate_load_cfg
import matplotlib.pyplot as plt

def plot_roc_precision_recall(stats,split,epoch,save_dir=''):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(stats['fpr'], stats['tpr'])
    plt.title(f'ROC - Feed Class {split}, epoch {epoch} auc={stats["aucroc"]:.2f}')
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.subplot(1, 2, 2)
    plt.plot(stats["recall"], stats["precision"])
    plt.title(f'ROC - Feed Class {split}, epoch {epoch} auc={stats["auprc"]:.2f}')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig(os.path.join(save_dir, f'{split}_epoch{epoch}_ROC_all.png'))

def get_stats(feed_targets, feed_preds):
    tp = ((feed_targets + feed_preds) == 2).sum()
    tn = ((feed_targets + feed_preds) == 0).sum()
    fn = ((feed_targets - feed_preds) == 1).sum()
    fp = ((feed_targets - feed_preds) == -1).sum()
    #fprs = fp / (fp + tn)
    tprs = tp / (tp + fn)
    precision = tp / (tp + fp)
    recall = tprs
    fpr, tpr, _ = roc_curve(feed_targets, feed_preds)
    aucroc= auc(fpr, tpr)
    precisions, recalls, thres = precision_recall_curve(feed_targets, feed_preds)
    f1 = 2 * (precision * recall) / (precision + recall)
    auprc = auc(recalls, precisions)
    return {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp, 'fpr': fpr, 'tpr': tpr,'f1':f1,
            'aucroc': aucroc, 'precision': precisions, 'recall': recalls, 'auprc':auprc}

def eval_epoch(model, loader, cfg, i3d=False):
    model.eval()
    num_correct = 0
    y_hats = []
    all_labels = []
    stats = {}
    running_err = 0
    all_preds = []
    all_file_names = []
    file_list = loader.dataset.dataset._labeled_videos._paths_and_labels
    with torch.no_grad():
        for cur_iter, (inputs, labels, indices, meta) in enumerate(loader):
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            if i3d:
                preds = model(inputs[0])
            else:
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
        stats['top1_err'] = (running_err / len(loader.dataset))
        stats['accuracy'] = (num_correct / len(loader.dataset)).item()
    return all_labels, y_hats, stats, all_preds, all_file_names

def load_checkpoint(path,cfg):
    checkpoint = torch.load(path)
    model = torch.hub.load("facebookresearch/pytorchvideo:main", model="slowfast_r50", pretrained=False)
    model.blocks[6].proj = torch.nn.Linear(in_features=2304, out_features=cfg.MODEL.NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

def get_eval_loader(cfg,split):
    cfg.TRAIN.EVAL_DATASET = True
    eval_loader = construct_loader(cfg,split)
    cfg.TRAIN.EVAL_DATASET = False
    return eval_loader

def get_split_results(model, cfg, split, epoch, plot=False, save=False,i3d=False):
    if split != 'test':
        loader = get_eval_loader(cfg, split)
    else:
        loader = construct_loader(cfg,split)
    labels, y_hats, stats, preds, file_names = eval_epoch(model, loader, cfg)
    results_df = pd.DataFrame({'split':[split]* len(file_names), 'file_name': file_names,
                                     'strike_scores': preds[:, 0].cpu(),
                                     'strike_labels': 1 - (labels).cpu()})
    if plot:
        targets = 1 - labels.cpu() # we want the strike class to be positive
        preds = preds[:, 0].cpu()
        other_stats = get_stats(targets, preds)
        plot_roc_precision_recall(other_stats, split, epoch, save_dir=cfg.OUTPUT_DIR)
    if save:
        file_path = os.path.join(cfg.OUTPUT_DIR,f'results_{split}_{epoch}.csv')
        results_df.to_csv(file_path)
    return results_df



def get_epoch_results(checkpoint_dir, epoch, cfg, plot=False,i3d=False):
    #read config file:
    if type(cfg)==str:
        #cfg argument is a path, load it as cfg:
        cfg = pirate_load_cfg(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    checkpoint_path = os.path.join(checkpoint_dir, f'pretrained_epoch{epoch}.pt')
    model = load_checkpoint(checkpoint_path, cfg)
    splits = ['train','val','test']
    results = []
    for split in splits:
        df = get_split_results(model, cfg, split, save=False, epoch=epoch, plot=plot,i3d=i3d)
        results.append(df)
    all_results = pd.concat(results)
    results_path = os.path.join(cfg.OUTPUT_DIR, f'strike_scores_all_splits_epoch{epoch}.csv')
    print(f'Saving results file at {results_path}')
    all_results.to_csv(results_path, index= False)


def eval_alt_testset(checkpoint_dir,epoch,cfg,testset_path, plot=False, i3d=False):
    #read config file:
    if type(cfg)==str:
        #cfg argument is a path, load it as cfg:
        cfg = pirate_load_cfg(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    checkpoint_path = os.path.join(checkpoint_dir, f'pretrained_epoch{epoch}.pt')
    model = load_checkpoint(checkpoint_path, cfg)
    cfg.DATA.PATH_TO_DATA_DIR = testset_path
    # create a folder with the dataset name to store results, assumes dataset name == folder name:
    exp_name = f'{os.path.basename(testset_path)}_eval'
    save_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), exp_name)
    cfg.OUTPUT_DIR = save_dir
    print(f'saving results at {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    # get results:
    alt_test_results = get_split_results(model, cfg, split='test', epoch=epoch, plot=plot, i3d=i3d)
    # save results:
    alt_test_results.to_csv(os.path.join(save_dir, f'{exp_name}_epoch_{epoch}.csv'), index=False)

