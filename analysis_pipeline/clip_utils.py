import sys
sys.path.append('/content/drive/MyDrive/PhD/codes/SlowFas')
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
import os
import shutil
import pandas as pd
import sklearn.metrics


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

def get_frame_volume(vid, curr_frame, clip_duration=100,num_channels=1):
    #curr_frame = vid.frame_pointer

    start_frame = curr_frame - round(clip_duration/2)
    #end_frame = curr_frame + round(clip_duration/2)

    if type(vid) == cv2.VideoCapture:
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        #vid_duration = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        vid.frame_pointer = start_frame-1
        #vid_duration = len(vid)
    #end_frame = min(end_frame, vid_duration)
    counter = 0
    frame_array = []
    while counter<clip_duration:
        ret, frame = vid.read()
        if num_channels==3:
            if not type(vid) == cv2.VideoCapture:
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
    # if type(vid) == cv2.VideoCapture:
    #     vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    # else:
    #     vid.frame_pointer = frame_num
    frame_array = get_frame_volume(vid, frame_num, clip_duration,num_channels)
    frame_array = np.transpose(frame_array, (3,0,1,2))
    clip_array = []
    centroids = []
    #clip_array = np.zeros((len(boxes),num_channels, clip_duration, clip_size, clip_size))
    for i,box in enumerate(boxes.values()):
        try:
            centroid = ((box[:2]+box[2:])/2).astype(int)
        except:
            print(frame_num)
        #dims = abs(box[:2]-box[2:])
        #max_dim = int(dims.max())
        #max_dim += buffer_size
        centroids.append(centroid)
        x,x1,y,y1 = get_clip_bounds(frame_array.shape[2:], centroid, round(clip_size/2))
        clip = torch.from_numpy(frame_array[:,:,y:y1,x:x1]).float()
        clip_array.append(clip)
    return clip_array, centroids

def save_clip(clip, moviepath,transformed=False, mp4=False,verbose=False):
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
    if verbose:
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


def trans_to_list(x):
    x = x.replace('  ', ' ')
    x = x.replace(' ', ',')
    l = x.strip('[').strip(']').split(',')
    l_new = [int(c) for c in l if c.isalnum()]
    return l_new


def find_double_clips(preds_df, min_dist_btwn_centroids=50):
    clips_to_remove_df = pd.DataFrame(columns=preds_df.columns)
    for frame_num, entry in preds_df.groupby('frame'):
        centroids = np.stack(entry.centroid.map(trans_to_list))
        if centroids.shape[0] > 1:
            # more than one entry per frame
            dist_mat = sklearn.metrics.pairwise_distances(X=centroids, metric='l2')
            np.fill_diagonal(dist_mat, np.inf)
            close_clips_idx = np.argwhere(dist_mat < min_dist_btwn_centroids)
            close_clips_idx.sort(axis=1)
            close_clips_idx = np.unique(close_clips_idx, axis=0)
            if close_clips_idx.any():
                for row in close_clips_idx:
                    close_clips = entry.iloc[row]
                    if abs(close_clips.iloc[0].detection_scores - close_clips.iloc[1].detection_scores) < 0.01:
                        # get the more confident entry farthest from the decision border
                        idx_to_remove = close_clips.strike_scores.map(lambda x: abs(0.5 - x)).idxmax()
                    else:
                        idx_to_remove = close_clips.detection_scores.idxmax()
                    remove_subset = close_clips.drop(idx_to_remove, axis=0)
                    # print(close_clips.detection_scores.idxmax())
                    remove_subset['neighbor_clip_row_index'] = idx_to_remove
                    # print(close_clips.iloc[close_clips.detection_scores.idxmax()])

                    if remove_subset.index.item() not in list(clips_to_remove_df.index):
                        clips_to_remove_df = clips_to_remove_df.append(remove_subset)
    return clips_to_remove_df


def confine_doubles(fish_root,vid_name,move=False,preds_filename='preds',dec_thresh=0.5,exp_name='kinetics'):
    seq = vid_name.split('_')[0]
    dph = vid_name.split('_')[1][:-3]
    fish = vid_name.split('_')[2].split('.')[0]
    folder_path = os.path.join(fish_root,dph,seq)
    df_preds = pd.read_csv(os.path.join(folder_path,f'{preds_filename}_{exp_name}.csv'))
    log_path = os.path.join(fish_root,f'experiment_log_{exp_name}_removed.csv')
    if os.path.isfile(log_path):
        experiment_log = pd.read_csv(log_path)
    else:
        experiment_log = pd.read_csv(os.path.join(fish_root,'experiment_log.csv'))
    experiment_log = experiment_log[experiment_log.experiment_name.map(lambda x: x.endswith(f'{exp_name}_exp'))]
    df_detected = df_preds[~df_preds.fish_id.isna()]
    clip_size = experiment_log[experiment_log.video_name==vid_name].frame_size.item()
    double_df = find_double_clips(df_detected,min_dist_btwn_centroids=round(clip_size*0.5))
    double_df.to_csv(os.path.join(folder_path,f'double_clips_to_remove_{round(clip_size*0.5)}px.csv'),index=False)
    IDX_TO_FOLDER = {1: 'strike', 0: 'swim'}
    doubles_folder = os.path.join(folder_path, 'removed_doubles')
    if move:
        os.makedirs(os.path.join(doubles_folder, 'swim'), exist_ok=True)
        os.makedirs(os.path.join(doubles_folder, 'strike'), exist_ok=True)
    df_remove = double_df[double_df[f'{exp_name}_strike_scores']>=dec_thresh]
    for i, entry in df_remove.iterrows():
        #file_name = f'midframe_{entry.frame}_fish_{int(entry.fish_id)}.avi'
        class_path = os.path.join(folder_path, IDX_TO_FOLDER[int(entry.action_preds)])
        if move:
            src = os.path.join(class_path, entry.clip_name)
            dst = os.path.join(doubles_folder, IDX_TO_FOLDER[int(entry.action_preds)], entry.clip_name)
            if os.path.isfile(src):
                shutil.move(src, dst)
            elif os.path.exists(dst):
                print(f'file already copied to {dst}')
            else:
                print(f'couldnt locate file {src}')
    df_preds_removed = df_preds[~df_preds.index.isin(double_df.index)]
    df_preds_removed.to_csv(os.path.join(folder_path, f'{preds_filename}_{exp_name}_without_removed.csv'), index=False)
    print(f'Moved doubles and saved double-free preds to {folder_path}/preds_without_removed.csv')
    experiment_log.loc[experiment_log.video_name==vid_name,'removed_preds'] = len(double_df)
    experiment_log.loc[experiment_log.video_name == vid_name, 'removed_clip'] = len(df_remove)
    experiment_log.to_csv(log_path, index=False)