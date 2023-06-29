import numpy as np
import os
import torch
from slowfast.utils.metrics import topk_accuracies,topks_correct
import torch.nn.functional as F
import slowfast.visualization.tensorboard_vis as tb
from sklearn.metrics import confusion_matrix
from slowfast.utils import metrics
from slowfast.datasets import loader
#from dataset import construct_loader
from eval_utils import eval_epoch, load_checkpoint, pirate_load_cfg
from slowfast.datasets.loader import construct_loader
from slowfast.datasets.build import DATASET_REGISTRY
from dataset import Ptvfishbase


def train_one_epoch(model, optim, loader_train, loss_func,cfg, i3d=False,calc_stats=True):
    model.train()
    train_loss = 0
    num_correct = 0
    train_stats = {'fps': 0, 'tns': 0, 'fns': 0, 'tps': 0, 'accuracy': 0}
    y_hats = []
    all_labels = []
    running_err = 0
    for cur_iter, (inputs, labels, _, meta) in enumerate(loader_train):
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
        y_hat = preds.max(axis=1).indices
        y_hats.append(y_hat)
        all_labels.append(labels)
        num_correct += (labels == y_hat).sum()
        loss = loss_func(preds, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        k = min(cfg.MODEL.NUM_CLASSES, 5)  # in case there aren't at least 5 classes in the dataset
        num_topks_correct = metrics.topks_correct(preds, labels, (1, k))
        top1_err, _ = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
        running_err += top1_err * preds.size(0)
        train_loss += loss / preds.shape[0]
    all_labels = torch.hstack(all_labels)
    y_hats = torch.stack(y_hats).ravel()
    if calc_stats:
        tn, fp, fn, tp = confusion_matrix(1 - all_labels.cpu(),
                                          1 - y_hats.cpu()).ravel()  # since feed is label 0 and we want it as label 1,
        train_stats['loss'] = train_loss
        train_stats['fns'] = fn
        train_stats['fps'] = fp
        train_stats['tns'] = tn
        train_stats['tps'] = tp
        train_stats['top1_err'] = (running_err / len(loader_train.dataset))
        train_stats['accuracy'] = num_correct / len(loader_train.dataset)
    else:
        train_stats['loss'] = train_loss
    return model, optim, train_stats, all_labels, y_hats

def load_model(cfg,pretrained=False,i3d=False,ssv2=False):
    if i3d:
        model_name = "i3d_r50"
        lin_features = 2048
    else:
        model_name = "slowfast_r50"
        lin_features = 2304

    if ssv2:
        # load ssv2 pretrained model, note you need to download the checkpoint to the checkpoints/ssv2_pretrained folder
        #os.system('wget https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/ssv2/SLOWFAST_8x8_R50.pyth')
        ckpt_path = './checkpoints/ssv2_pretrained/SLOWFAST_8x8_R50.pyth'
        assert os.path.exists(ckpt_path), print('Oops! SSv2 pretrained model checkpoint not found (see readme)')
        tmp_cfg = cfg.clone()
        tmp_cfg.MODEL.NUM_CLASSES = 174 # change the number of classes to load the checkpoint
        model = load_checkpoint(ckpt_path, tmp_cfg)
    else:
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=pretrained)
    model.blocks[6].proj = torch.nn.Linear(in_features=lin_features, out_features=cfg.MODEL.NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

def train(cfg, model=None,pretrained=True, i3d=False, ssv2=False, val_every=5):
    #read config file:
    if type(cfg)==str:
        #cfg argument is a path, load it as cfg:
        cfg = pirate_load_cfg(cfg_path)
    print('starting train')
    #set random seeds
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    #get data loaders:
    loader_train = construct_loader(cfg, 'train')
    loader_val = construct_loader(cfg, 'val')
    # get model if not given:
    if model is None:
        print('loading model')
        model = load_model(cfg,pretrained,i3d,ssv2)
    if cfg.TENSORBOARD.ENABLE:
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=cfg.SOLVER.MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    exp_dir = cfg.OUTPUT_DIR
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)

    train_f1s = []
    val_f1s = []
    train_recall = []
    val_recall = []
    train_losses = []
    prec_func = lambda st: st['tps'] / (st['tps'] + st['fps'])
    rec_func = lambda st: st['tps'] / (st['tps'] + st['fns'])
    f1_func = lambda st: 2 * (st['precision'] * st['recall']) / (st['precision'] + st['recall'])
    for cur_epoch in range(cfg.SOLVER.MAX_EPOCH):
        loader.shuffle_dataset(loader_train, cur_epoch)
        model, optimizer, train_stats, train_labels, train_y_hats= train_one_epoch(model, optimizer, loader_train, loss_func, cfg, i3d=i3d)
        scheduler.step(train_stats['loss'])
        #train_labels, train_y_hats, train_eval_stats, _, _ = eval_epoch(model, loader_train, cfg, i3d=i3d)

        train_stats['precision'] = prec_func(train_stats)
        train_stats['recall'] = rec_func(train_stats)
        train_stats['f1'] = f1_func(train_stats)
        #train_eval_stats['precision'] = prec_func(train_eval_stats)
        #train_eval_stats['recall'] = rec_func(train_eval_stats)
        #train_eval_stats['f1'] = f1_func(train_eval_stats)

        train_f1s.append(train_stats['f1'])

        train_recall.append(train_stats['recall'])

        train_losses.append(train_stats['loss'])
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/epoch_loss": train_stats['loss'],
                    "Train/epoch_top1_err": train_stats['top1_err'],
                    #"Train_eval/epoch_top1_err": train_eval_stats['top1_err'],
                    "Train/epoch_accuracy": train_stats['accuracy'],
                    "Train/epoch_precision": train_stats['precision'],
                    "Train/epoch_recall": train_stats['recall']
                },
                global_step=cur_epoch,
            )
        print(f'{cur_epoch}/{cfg.SOLVER.MAX_EPOCH}: loss {train_stats["loss"]} '
              f'Train F1 {train_stats["f1"]:.2f}, acc {train_stats["accuracy"]:.2f}, '
              f'recall {train_stats["recall"]:.2f}')
        if cur_epoch % val_every == 0:
            val_labels, val_y_hats, val_stats, _, _ = eval_epoch(model, loader_val, cfg, i3d=i3d)
            val_stats['precision'] = prec_func(val_stats)
            val_stats['recall'] = rec_func(val_stats)
            val_stats['f1'] = f1_func(val_stats)
            val_f1s.append(val_stats['f1'])
            val_recall.append(val_stats['recall'])
            if writer is not None:
                writer.add_scalars(
                    {
                "Val/epoch_top1_err": val_stats['top1_err'],
                "Val/epoch_precision": val_stats['precision'],
                "Val/epoch_recall": val_stats['recall'] },
                global_step=cur_epoch,
            )
            print(f'Val F1 {val_stats["f1"]:.2f}, acc {val_stats["accuracy"]:.2f}, recall {val_stats["recall"]:.2f}')



        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'train_labels': train_labels,
                    'train_y_hats': train_y_hats,
                    'val_labels': val_labels,
                    'val_y_hats': val_y_hats,
                    'scheduler_state': scheduler.state_dict(),
                    'train_stats': train_stats,
                    #'train_eval_stats': train_eval_stats,
                    'val_stats': val_stats},
                   os.path.join(exp_dir, 'checkpoints', f'pretrained_epoch{cur_epoch}.pt'))
    if writer is not None:
        writer.close()
