import random
import cv2
import torch
import slowfast.datasets.utils as utils
from slowfast.datasets.ptv_datasets import PackPathway,DictToTuple,PTVDatasetWrapper
from pytorchvideo.data import (
    LabeledVideoDataset,
    make_clip_sampler,
)
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
    RandomHorizontalFlipVideo,
)
from transform import *
import os
from slowfast.datasets.build import DATASET_REGISTRY

def div255(x):
    """
    Scale clip frames from [0, 255] to [0, 1].
    Args:
        x (Tensor): A tensor of the clip's RGB frames with shape:
            (channel, time, height, width).

    Returns:
        x (Tensor): Scaled tensor by divide 255.
    """
    return x / 255.0



def rgb2gray(x):
    """
    Convert clip frames from RGB mode to GRAYSCALE mode.
    Args:
        x (Tensor): A tensor of the clip's RGB frames with shape:
            (channel, time, height, width).

    Returns:
        x (Tensor): Converted tensor
    """
    return x[[0], ...]

@DATASET_REGISTRY.register()
def Ptvfishbase(cfg, mode):
    """
    Construct the Fishbase video loader with a directory, each directory is split into modes ('train', 'val', 'test')
    and inside each mode are subdirectories for each label class.
    For `train` and `val` mode, a single clip is randomly sampled from every video
    with random cropping, scaling, and flipping. For `test` mode, multiple clips are
    uniformaly sampled from every video with center cropping.
    Args:
        cfg (CfgNode): configs.
        mode (string): Options includes `train`, `val`, or `test` mode.
            For the train and val mode, the data loader will take data
            from the train or val set, and sample one clip per video.
            For the test mode, the data loader will take data from test set,
            and sample multiple clips per video.
    """
    # Only support train, val, and test mode.
    assert mode in [
        "train",
        "val",
        "test",
        #'train_eval',
        #'val_eval',
    ], "Split '{}' not supported".format(mode)

    clip_duration = (
        cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE / cfg.DATA.TARGET_FPS
    )
    path_to_dir = os.path.join(
        cfg.DATA.PATH_TO_DATA_DIR, mode.split('_')[0] #added split to deal with the case of train_eval and val_eval
    )

    labeled_video_paths = LabeledVideoPaths.from_directory(path_to_dir)
    num_videos = len(labeled_video_paths)
    labeled_video_paths.path_prefix = cfg.DATA.PATH_PREFIX
    if not cfg.TRAIN.EVAL_DATASET:
        num_clips = 1
        num_crops = 1

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(div255),
                            RandomColorJitter(brightness_ratio=cfg.DATA.BRIGHTNESS_RATIO, p=cfg.DATA.BRIGHTNESS_PROB), #first trial 0.3
                            RandomGaussianBlur(kernel=13, sigma=(6.0,10.0), p=cfg.DATA.BLUR_PROB), # first trial 0.2
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            ShortSideScale(cfg.DATA.TRAIN_JITTER_SCALES[0]),
                        ]
                        + (
                            [Lambda(rgb2gray)]
                            if cfg.DATA.INPUT_CHANNEL_NUM[0] == 1
                            else []
                        )
                        + (
                            [VarianceImageTransform(var_dim=cfg.DATA.VAR_DIM)]
                            if cfg.DATA.VARIANCE_IMG
                            else []
                        )
                        + (
                            [RandomHorizontalFlipVideo(p=0.5),
                             RandomVerticalFlipVideo(p=0.5),
                             RandomRot90Video(p=0.5)]
                            if cfg.DATA.RANDOM_FLIP
                            else []
                        )
                        + [PackPathway(cfg)]
                    ),
                ),
                DictToTuple(num_clips, num_crops),
            ]
        )

        clip_sampler = make_clip_sampler("random", clip_duration)
        if cfg.NUM_GPUS > 1:
            video_sampler = DistributedSampler
        else:
            video_sampler = (
                RandomSampler if mode == "train" else SequentialSampler
            )
    else:
        num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS
        num_crops = cfg.TEST.NUM_SPATIAL_CROPS

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(div255),
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            ShortSideScale(
                                size=cfg.DATA.TRAIN_JITTER_SCALES[0]
                            ),
                        ]
                        + (
                            [Lambda(rgb2gray)]
                            if cfg.DATA.INPUT_CHANNEL_NUM[0] == 1
                            else []
                        )
                        + (
                            [VarianceImageTransform(var_dim=cfg.DATA.VAR_DIM)]
                            if cfg.DATA.VARIANCE_IMG
                            else []
                        )
                    ),
                ),
                ApplyTransformToKey(key="video", transform=PackPathway(cfg)),
                DictToTuple(num_clips, num_crops),
            ]
        )
        clip_sampler = make_clip_sampler(
            "constant_clips_per_video",
            clip_duration,
            num_clips,
            num_crops,
        )
        video_sampler = (
            DistributedSampler if cfg.NUM_GPUS > 1 else SequentialSampler
        )

    return PTVDatasetWrapper(
        num_videos=num_videos,
        clips_per_video=num_clips,
        crops_per_clip=num_crops,
        dataset=LabeledVideoDataset(
            labeled_video_paths=labeled_video_paths,
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            transform=transform,
            decode_audio=False,
        ),
    )


# def construct_loader(cfg, split):
#     """
#     Constructs the data loader for the given dataset.
#     Args:
#         cfg (CfgNode): configs. Details can be found in
#             slowfast/config/defaults.py
#         split (str): the split of the data loader. Options include `train`,
#             `val`, and `test`.
#     """
#     assert split in ["train", "val", "test","train_eval","val_eval"]
#     if split in ["train"]:
#         batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
#         shuffle = True
#         drop_last = True
#     elif split in ["val", "val_eval","train_eval"]:
#         batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
#         shuffle = False
#         drop_last = False
#     elif split in ["test"]:
#         batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
#         shuffle = False
#         drop_last = False
# 
#     # Construct the dataset
#     dataset = Ptvfishbase(cfg, split)
# 
#     loader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             num_workers=cfg.DATA_LOADER.NUM_WORKERS,
#             pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
#             drop_last=drop_last,
#             collate_fn=None,
#             shuffle=shuffle,
#             worker_init_fn=utils.loader_worker_init_fn(dataset),
#         )
#     return loader


