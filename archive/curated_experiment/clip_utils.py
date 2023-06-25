import sys
sys.path.append('/media/shirbar/DATA/codes/SlowFast/')
import numpy as np
import cv2
import moviepy.editor as mpy
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    ShortSideScale,
    UniformCropVideo,
    UniformTemporalSubsample,
)
from slowfast.datasets.ptv_datasets import Ptvfishbase, PackPathway, div255, rgb2gray
from slowfast.datasets.transform import VarianceImageTransform
def get_clip_bounds(frame_size, centroid, padding):
    x2 = 0
    y2 = 0
    x1 = centroid[0] - padding # Leftmost x point
    if x1 < 0:
        # If negative, shift x2 by the difference of x1 from zero:
        # note that the fish will not be in the center of the frame but we will not have issues with black margins
        # in the videos.
        # Once that is done zero out x1:
        x2 = abs(x1)
        x1 = 0
    x2 += centroid[0] + padding  # Add the padding size the rightmost x point
    if x2 > frame_size[1]:
        # If overshoots original frame size, replace by the frame width and move the x1 back by the difference
        # of x2 from the edge of the frame:
        x1 -= (x2-int(frame_size[1]))
        x2 = int(frame_size[1])
    y1 = centroid[1] - padding  # Leftmost y point
    if y1 < 0:
            # If negative, add the absolute value, i.e the difference from 0, to y2 and replace y1 to 0:
        y2 = abs(y1)
        y1 = 0
    y2 += int(centroid[1] + padding)  # Rightmost y point
    if y2 > frame_size[0]:
        # If overshoots original frame size, replace by the frame height and move y1 back by the difference of y2
        # from the edge of the frame:
        y1 -= (y2 - int(frame_size[0]))
        y2 = int(frame_size[0])
    return x1, x2, y1, y2

def get_frame_volume(vid, clip_duration=100,num_channels=1):
    curr_frame = vid.frame_pointer
    start_frame = curr_frame - round(clip_duration/2)
    end_frame = curr_frame + round(clip_duration/2)
    end_frame = min(end_frame, len(vid))
    vid.frame_pointer = start_frame-1
    counter = 0
    frame_array = []
    while counter<clip_duration:
        ret, frame = vid.read()
        if num_channels==3:
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        else:
            frame = frame[...,np.newaxis]
        if not ret:
            break
        frame_array.append(frame)
        counter += 1
    frame_array = np.array(frame_array)
    return frame_array

def get_clips(vid,frame_num,boxes,clip_duration=100, clip_size=500, num_channels=1, buffer_size=100):
    vid.frame_pointer = frame_num
    frame_array = get_frame_volume(vid, clip_duration,num_channels)
    frame_array = np.transpose(frame_array, (3,0,1,2))
    clip_array = []
    centroids = []
    #clip_array = np.zeros((len(boxes),num_channels, clip_duration, clip_size, clip_size))
    for i,box in enumerate(boxes.values()):
        centroid = ((box[:2]+box[2:])/2).astype(int)
        dims = abs(box[:2]-box[2:])
        max_dim = int(dims.max())
        max_dim += buffer_size
        centroids.append(centroid)
        x,x1,y,y1 = get_clip_bounds(frame_array.shape[2:], centroid, round(clip_size/2))
        clip = torch.from_numpy(frame_array[:,:,y:y1,x:x1]).float()
        clip_array.append(clip)
    return clip_array, centroids

def save_clip(clip, moviepath,transformed=False, mp4=False):
    rgb = clip.shape[0] == 3
    clip = np.transpose(clip, (1,2,3,0))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if transformed:
        frame_list = []
    else:
        video_writer = cv2.VideoWriter(moviepath, fourcc, 30, (clip.shape[2], clip.shape[1]), rgb)
    for frame in clip:
        frame = frame.numpy().astype(np.uint8)
        if transformed:
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            frame_list.append(frame)
        else:
            video_writer.write(np.squeeze(frame))
    if transformed:
        cl = mpy.ImageSequenceClip(frame_list, fps=30)
        if mp4:
            cl.write_videofile(moviepath)
        else:
            cl.write_videofile(moviepath, codec='png')
    else:
        video_writer.release()
    print(f'Saved video to path: {moviepath}')

def transform_clips(clips, cfg, normalize = True):
    transform = Compose([UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),  # Compose([
                         Lambda(div255)]
                         +
                         (
                             [NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),]
                             if normalize
                             else []
                         ) +
                         [
                         ShortSideScale(
                             size=cfg.DATA.TRAIN_JITTER_SCALES[0])
                         ] +
                        (
                            [Lambda(rgb2gray)]
                            if cfg.DATA.INPUT_CHANNEL_NUM[0] == 1
                            else []
                        ) +  # ),
                        (
                            [VarianceImageTransform(var_dim=cfg.DATA.VAR_DIM)]
                            if cfg.DATA.VARIANCE_IMG
                            else []
                        )+
                        [PackPathway(cfg)])

    transformed_clips = [transform(clip) for clip in clips]
    return transformed_clips