import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from loaders import *
import random
from cust_parser import get_parser
from utils import get_action_classifications,get_fish_detection,plot_boxes, get_detections_from_preds
from fish_for_fish import make_folders, write_clip, get_fps, load_log
from clip_utils import get_clips, save_clip
import time
import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'running on {DEVICE}')

def get_vid_len(vid,vid_path):
    if vid_path.endswith('.seq'):
        vid_len = len(vid)
    else:
        vid_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    return vid_len

def analyze_vid(root_path, vid_path, vid_name, clip_size,
                  detector, classifier, classifier_cfg, clip_duration=80,
                  thresh=0.5, fish_to_sample=1000, save_clips=True, last_frame=None, start_idx=0):
    # Load video and get video properties:
    vid = load_video(vid_path)
    vid_len = get_vid_len(vid, vid_path)
    fps = get_fps(vid, vid_path)
    all_preds = []
    fish_sampled = 0
    time_sampled = 0
    frame_num = start_idx + round(clip_duration/2) # The first frame we sample should be half a clip length before the start frame index
    # clean vid_name from problematic characters so we can use it for folder name
    exp_name = vid_name.replace(' ', '_')
    exp_name = exp_name.replace('.','-')
    # create results folder:
    folder_path = os.path.join(root_path,exp_name)
    make_folders(folder_path)
    os.makedirs(os.path.join(folder_path, 'sample_frames'), exist_ok=True)
    df_preds = pd.DataFrame(columns=['vid_name','frame','clip_name',
                                     'fish_id','centroid','bboxs','detection_scores','detection_pred_class',
                                     'action_preds','strike_scores','comments'])
    log_path = os.path.join(root_path,'experiment_log.csv')
    log_df = load_log(log_path)
    frames_with_detections = 0
    start_time = time.time()
    frames = []
    fish_bar = tqdm(total=vid_len/(fps*60)) # progress bar for analyzing entire video
    if not last_frame:
        last_frame = vid_len - (round(clip_duration/2) + 1)
    else:
        last_frame = last_frame - (round(clip_duration/2) + 1)
    while ((fish_sampled <= fish_to_sample) and (frame_num<(last_frame))):
        comments = ''
        try:
            if vid_path.endswith('.seq'):
                frame = vid[frame_num]['frame']
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _,frame = vid.read()
        except:
            comments = "can't read frame from video"
        outputs,boxes = get_fish_detection(detector, frame)
        if len(all_preds) == 0 and len(boxes) > 0:
            plot_boxes(frame, outputs, save=True, filepath=os.path.join(folder_path, 'sample_frames',
                                                                    f'{vid_name}_frame_{frame_num}.jpg'))
            plt.close()

        if len(boxes) > 0:
            frames_with_detections += 1

            clips,centroids = get_clips(vid, frame_num, boxes,
                                        clip_duration=clip_duration, clip_size=clip_size, num_channels=3)
            preds, pred_classes, pred_class_names,_ = get_action_classifications(classifier, clips, classifier_cfg,
                                                                                                DEVICE,
                                                                                                 bbox_clip_sizes=False,
                                                                                                 normalize=True,
                                                                                                 verbose=False)
            for i,p in enumerate(pred_classes.flatten()):
                thresh_pred = int(preds[i,0].item()>=thresh)
                if save_clips:
                    clip_name=write_clip(folder_path,clip=clips[i],prediction=thresh_pred,frame_num=frame_num,idx=i,
                               vid_name=vid_name, centroid=centroids[i])
                new_row = {'vid_name':vid_name,'frame':frame_num,'clip_name':clip_name,
                                            'fish_id':i,'centroid':centroids[i],
                                            'bboxs':list(boxes.values())[i],
                                        'detection_scores':list(boxes.keys())[i],
                                            'detection_pred_class':outputs['instances'].pred_classes[i].item(),
                                            'action_preds':thresh_pred,
                                            'strike_scores':preds[i,0].item(),'comments':comments}
                df_preds = pd.concat([df_preds,pd.DataFrame([new_row])],axis=0,ignore_index=True)
            all_preds.append(preds),
            feeding_fish = (preds>=thresh).sum(axis=0)[0].item()

        else:
            # even if we find no fish, insert an empty row to indicate we proccessed the frame:
            new_row = {'vid_name':vid_name,'frame':frame_num,'clip_name':None,
                                        'fish_id':None,'centroid':None, 'bboxs':None,
                                        'detection_scores':None,'detection_pred_class':None,
                                            'action_preds':None, 'strike_scores':None,'comments':comments},
            df_preds = pd.concat([df_preds,pd.DataFrame(new_row,columns=df_preds.columns)],
                                 axis=0,ignore_index=True)
            feeding_fish = 0

        frames.append(frame_num)
        frame_num = frame_num + round(clip_duration*0.75)
        fish_sampled += feeding_fish  # add predicted strikes to fish count
        prev_time = time_sampled
        time_sampled = ((frames[-1]/fps)/60)-((frames[0]/fps)/60)  # time sampled in minutes
        fish_bar.update(time_sampled-prev_time)
    plot_boxes(frame, outputs, save=True, filepath=os.path.join(folder_path, 'sample_frames',
                                                                f'{vid_name}_frame_{frame_num}.jpg'))
    plt.close()
    fish_bar.close()
    df_preds.to_csv(os.path.join(folder_path, 'preds.csv'), index=False)
    execution_time = time.time() - start_time
    tot_strike = fish_sampled
    if len(all_preds)>0:
        tot_swim = (torch.vstack(all_preds)<0.9).sum(axis=0)[0].item()
    else:
        tot_swim = 0


    entry = {'experiment_name': exp_name, 'video_name': vid_name,'fps':fps,
            'num_frames_sampled': len(frames),
             'first_frame': start_idx,
             'last_frame_sampled':frames[-1],
                'duration sampled': time_sampled,
                 'num_frames_with_detections': frames_with_detections,
                 'sample_regime': '5min', 'clip_duration': clip_duration,
                 'frame_size': clip_size,
                 'num_labeled_strike': tot_strike,
                 'num_labeled_swim': tot_swim,
                 'total_clips': tot_swim + tot_strike, 'execution_time': execution_time}
    log_df = pd.concat([log_df, pd.DataFrame([entry])],axis=0, ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(log_path)
    print(execution_time)
    vid.release()



def get_file_list(dir_to_scan, vid_extension, custom_file_list=[]):
    file_list = []
    if len(custom_file_list)>0:
        ls = np.hstack(custom_file_list.values)
    else:
        ls = []
    #ls = [l.lower() for l in ls]
    #print(ls)
    #assert(False)
    for root, dirs, files in os.walk(dir_to_scan):
        for file in files:
            if file.endswith(vid_extension) & ('cali' not in file.lower()):
                if (len(custom_file_list) == 0) or (file.split('.')[0] in ls):
                    file_list.append(os.path.join(root,file))
    print(f'Found {len(file_list)} videos in directory')
    return file_list


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    root_experiment_folder = args.root_dir  # where we save all the video experiments
    os.makedirs(root_experiment_folder, exist_ok=True)
    video_folder = args.video_dir
    vid_name = args.video_name
    if args.detector_name=='fasterRCNN':
        detector_path = f'./models/{args.detector_name}.pth'
    else:
        detector_path = f'./models/{args.detector_name}.pt'
    cfg_path = args.cfg_path
    classifier_path = f'./models/{args.classifier_name}.pt'
    detector, cfg_detect = load_detector(detector_path, confidence=0.5, nms=0.3) #these confidence and nms settings worked for us worth playing around with
    classifier, cfg_classify = load_action_classifier(cfg_path, classifier_path, pytorchvideo=True)

    if vid_name != 'all':
        vid_path = os.path.join(video_folder,vid_name+args.vid_ext)
        if os.path.isfile(vid_path):
            analyze_vid(root_experiment_folder, vid_path,
                          vid_name=vid_name, clip_size=args.clip_size,
                      detector=detector, classifier=classifier, classifier_cfg=cfg_classify,
                        fish_to_sample=args.fish_to_sample, clip_duration=args.clip_duration,
                        save_clips=not args.no_clips,thresh=args.dec_thresh, last_frame=args.last_frame)
        else:
            print(f'Video file not found in path {vid_path}')
    else:

        if len(args.video_list_path) > 0:
            custom_file_list = pd.read_csv(args.video_list_path)
        else:
            custom_file_list = []
        print(len(custom_file_list))
        file_list = get_file_list(video_folder, args.vid_ext, custom_file_list)
        print(f'about to analyze {len(file_list)} videos')
            #file_list = [f for f in file_list if f in custom_file_list]
            #assert(len(file_list)==len(custom_file_list)), f'oops {len(file_list)} doesnt add up'
        assert len(file_list)>0, \
            "Oops! video folder doesn't seem to have any of the experiment videos, check get_all_vids_dict()"
        for curr_path in file_list:
            curr_vid = os.path.basename(curr_path).strip(args.vid_ext)
            print(f'Vid {curr_vid} at work')
            analyze_vid(root_experiment_folder, curr_path,
                          vid_name=curr_vid, clip_size=args.clip_size,
                      detector=detector, classifier=classifier, classifier_cfg=cfg_classify,
                        fish_to_sample=args.fish_to_sample,clip_duration=args.clip_duration,
                        save_clips=not args.no_clips, thresh=args.dec_thresh, last_frame=args.last_frame,
                        start_idx=args.first_frame)
    print('Done!')