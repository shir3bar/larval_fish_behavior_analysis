import os.path

from loaders import *
import torch
import matplotlib.pyplot as plt
from utils import get_action_classifications, get_fish_detection,plot_boxes, get_annts_by_frame
from clip_utils import get_clips, transform_clips, save_clip
import random
#from data_utils import get_coco_style_json
import time
import argparse


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_frames_to_extract(frames_in_raw, init_frames,vid_len, buffer=25, swim_to_strike_ratio=10):
    assert(len(frames_in_raw)==len(init_frames))
    feeding_event_start_frames = [f+i for f,i in zip(frames_in_raw, init_frames)]
    feeding_events_midframes = [f+buffer for f in feeding_event_start_frames]
    frames_to_extract = []
    forbidden_frames = [list(range(f-buffer*2,f+buffer*2)) for f in feeding_events_midframes]
    forbidden_frames = np.hstack(forbidden_frames)
    frame_population = [f for f in range(vid_len) if f not in forbidden_frames]
    frames_to_extract+=random.sample(frame_population,swim_to_strike_ratio*len(feeding_events_midframes))
    frames_to_extract+=feeding_events_midframes
    feeding_events_midframes = np.sort(feeding_events_midframes)
    return np.sort(frames_to_extract), feeding_events_midframes


def get_all_vids_dict():
    """ Get a dictionary with the to sample in each of the 11 videos used in the curated experiment.
    This includes both frames with strike events (and other interactions with prey)
    and about ten times more frames without any events (with a certain buffer around the strike events).
    This function returns a dictionary, the keys are the video sequence name,
    the values are the frame numbers to sample from the video and the middle frames of just the strike events."""
    vid_lens = {'Seq1_23DPH_F02': 899920, 'Seq1_23DPH_F12': 348209,
                'Seq1_24DPH_F10': 769201, 'Seq1_26DPH_F06': 1356653,
                'Seq1_30DPH_F15': 665069, 'Seq2_25DPH_F12': 971277,
                'Seq2_27DPH_F12': 600157, 'Seq2_27DPH_F14': 602101, 'Seq2_28DPH_F15': 601283,
                'Seq2_30DPH_F15': 591703, 'Seq3_13DPH_F02': 1019990}
    frame_in_raw = {'Seq2_30DPH_F15':[150884,175801,180796,180796,180796,20289,208892,299245,299245,
                                      299245,299245,317146,317146,317146,402786,412494,439457,78674,78674],
                    'Seq1_30DPH_F15':[125331, 147970, 332573, 358782, 370196, 515465, 589218],
                    'Seq2_28DPH_F15':[132821,165326,190901,264601,371703,371703,
                                      371703,371703,371703,401423,483840,504807],
                    'Seq1_26DPH_F06':[1175054+10,1175054+10,218416,272303+50,819558,883645,989816,992509],
                    'Seq1_23DPH_F02':[602344,89409-8,304909,48142,48142,48142,555623+10],
                    'Seq2_27DPH_F14':[31266,374568,396679,42832,538532],
                    'Seq3_13DPH_F02':[693298,452348,641003,641003,772988],
                    'Seq1_23DPH_F12':[161156,161156,161156,199086,199086,199086,96149+2],
                    'Seq1_24DPH_F10':[233737-25,343272-30,468752,499465,708759],
                    'Seq2_27DPH_F12':[148928,148928,246290,288911,38432],
                    'Seq2_25DPH_F12':[239033,255151,357167,365361,523155,558732,
                                      639333,730451,769409,817061,848076,930621]
                    }
    init_frames = {'Seq2_30DPH_F15':[20,26,226,1306,1455,28,42,86,175,290,416,117,713,1256,33,22,17,61,743],
                   'Seq1_30DPH_F15':[10,39,41,1,52,1136,23],
                   'Seq2_28DPH_F15':[25,28,1,56,817,1548,184,2246,2347,6,32,2],
                   'Seq1_26DPH_F06':[235,298,16,511,27,97,229,771],
                   'Seq1_23DPH_F02':[210,1556,221,220,291,717,294],
                   'Seq2_27DPH_F14':[14,33,77,5,13],
                   'Seq3_13DPH_F02':[1071,106,1887,2302,211],
                   'Seq1_23DPH_F12':[19,199,336,190,279,787,165],
                   'Seq1_24DPH_F10':[56,145,144,17,85],
                   'Seq2_27DPH_F12':[21,1163,81,87,34],
                   'Seq2_25DPH_F12':[2,56,21,44,73,209,28,68,13,9,29,4]
                   }
    all_vids_dict = {}
    vid_keys = frame_in_raw.keys()
    for vid_key in vid_keys:
        frames_to_extract, feeding_events_midframe = get_frames_to_extract(frame_in_raw[vid_key],
                                                                           init_frames[vid_key],
                                                                           vid_len=vid_lens[vid_key],
                                                                           buffer=25)
        all_vids_dict[vid_key] = {'frames_to_extract': frames_to_extract, 'midframes': feeding_events_midframe}
    return all_vids_dict, vid_lens

def get_clip_sizes(all_vids_dict):
    """ Get the size of clips to crop for each video.
    This is hard-coded according to a visual inspection of videos,
    default is 350x350pixel but this might change due to fish's age or camera zoom."""
    clip_sizes = {}
    default_clip_size = 350
    for VIDEO_NAME, value_dict in all_vids_dict.items():
        clip_sizes[VIDEO_NAME] = default_clip_size
    clip_sizes['Seq1_23DPH_F02'] = 400
    clip_sizes['Seq1_24DPH_F10'] = 650
    clip_sizes['Seq1_26DPH_F06'] = 450
    clip_sizes['Seq2_25DPH_F12'] = 250
    clip_sizes['Seq2_27DPH_F12'] = 550
    clip_sizes['Seq2_27DPH_F14'] = 300
    return clip_sizes

def load_log(path):
    if os.path.isfile(path):
        df_run_params = pd.read_csv(path)
    else:
        df_run_params = pd.DataFrame(columns=['experiment_name','video_name','video_duration',
                                                  'overlap','num_frames_sampled','num_frames_with_detections',
                                                  'sample_regime',
                                                  'clip_duration','frame_size','TP',
                                          'num_labeled_strike','FN','num_labeled_swim',
                                                  'total_clips','execution_time'])
    return df_run_params


def make_folders(path):
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, 'swim', 'original'), exist_ok=True)
        os.makedirs(os.path.join(path, 'strike', 'original'), exist_ok=True)


def write_clip(folder_path,clip,prediction,frame_num,idx):
    if prediction == 0:
        movie_path = os.path.join(folder_path, 'strike', 'original',
                                  f'midframe_{frame_num}_fish_{idx}.avi')
    else:
        movie_path = os.path.join(folder_path, 'swim', 'original',
                                  f'midframe_{frame_num}_fish_{idx}.avi')
    save_clip(clip, movie_path, transformed=False)


def process_video(root_path, vid_path, vid_dict, vid_name, clip_size, vid_duration,
                  detector, classifier, classifier_cfg, start_idx=0, clip_duration=80, save_clips=True):
    vid = load_video(vid_path)
    all_preds = []
    all_classes = []
    nboxes = {}
    frames_to_extract = np.sort(vid_dict['frames_to_extract'])
    midframes = vid_dict['midframes']
    images_to_save = random.sample(list(midframes),3)
    os.makedirs(os.path.join(root_path,'sample_frames'), exist_ok=True)
    for frame_num in images_to_save:
        frame = vid[frame_num]['frame']
        outputs, boxes = get_fish_detection(detector, frame)
        plot_boxes(frame, outputs, save=True,filepath=os.path.join(root_path,'sample_frames',
                                                                   f'{vid_name}_frame_{frame_num}.jpg'))
        plt.close()
    EXP_NAME = f'{vid_name}_curated_exp'
    folder_path = os.path.join(root_path,EXP_NAME)
    make_folders(folder_path)
    df_preds = pd.DataFrame(columns=['frame','fish_id','centroid','bboxs','detection_scores','detection_pred_class',
                                     'action_preds','strike_scores'])
    log_path = os.path.join(root_path,'experiment_log.csv')
    log_df = load_log(log_path)
    frames_with_detections = 0
    start_time = time.time()
    for frame_num in frames_to_extract[start_idx:]:
        print(frame_num)
        frame = vid[frame_num]['frame']
        outputs,boxes = get_fish_detection(detector, frame)
        nboxes[frame_num] = len(boxes)

        if len(boxes)>0:
            frames_with_detections +=1

            clips,centroids = get_clips(vid, frame_num, boxes,
                                        clip_duration=clip_duration, clip_size=clip_size,num_channels=3)
            preds, pred_classes, pred_class_names,_ = get_action_classifications(classifier, clips, classifier_cfg,
                                                                                                DEVICE,
                                                                                                 bbox_clip_sizes=False,
                                                                                                 normalize=True,
                                                                                                 verbose=False)
            for i,p in enumerate(pred_classes.flatten()):
                if save_clips:
                    write_clip(folder_path,clip=clips[i],prediction=p,frame_num=frame_num,idx=i)
                df_preds = df_preds.append({'frame':frame_num,'fish_id':i,'centroid':centroids[i],
                                            'bboxs':list(boxes.values())[i],
                                        'detection_scores':list(boxes.keys())[i],
                                            'detection_pred_class':outputs['instances'].pred_classes[i].item(),
                                            'action_preds':p.item(), 'strike_scores':preds[i,0].item()},
                                   ignore_index=True)
            all_classes.append(pred_classes)
            all_preds.append(preds)
        else:
            df_preds = df_preds.append({'frame':frame_num,'fish_id':None,'centroid':None, 'bboxs':None,
                                        'detection_scores':None,'detection_pred_class':None,
                                            'action_preds':None, 'strike_scores':None},
                                   ignore_index=True)
    df_preds.to_csv(os.path.join(folder_path, 'preds.csv'), index=False)
    execution_time = time.time() - start_time
    tot_strike = len(os.listdir(os.path.join(folder_path, 'strike', 'original')))
    tot_swim = len(os.listdir(os.path.join(folder_path, 'swim', 'original')))

    entry = {'experiment_name': EXP_NAME, 'video_name': vid_name, 'video_duration': vid_duration,
                 'overlap': 0, 'num_frames_sampled': len(frames_to_extract),
                 'num_frames_with_detections': frames_with_detections,
                 'sample_regime': 'directed', 'clip_duration': clip_duration,
                 'frame_size': clip_size, 'TP': np.NaN,
                 'num_labeled_strike': tot_strike,
                 'FN': np.NaN,
                 'num_labeled_swim': tot_swim,
                 'total_clips': tot_swim + tot_strike, 'execution_time': execution_time}
    log_df = log_df.append(entry, ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(execution_time)
    vid.release()


def get_existing_dict(video_folder,all_vids_dict):
    existing_vids_dict = {}
    file_list = [f.split('.')[0] for f in os.listdir(video_folder)]
    for file in file_list:
        if file in all_vids_dict.keys():
            existing_vids_dict[file] = all_vids_dict[file]
    return existing_vids_dict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='enter root path where experiments will be saved')
    parser.add_argument('video_dir', help='enter path to directory containing videos')
    parser.add_argument('-video_name', default='all', help='Specify a single video to analyze, if no name is specified '
                                                           'all videos in the folder will be analyzed')
    parser.add_argument('-clip_duration',type=int,default=80,help='Duration of clips to cut for the action classifier')
    parser.add_argument('-no_clips', action='store_true',
                        help="Don't save the clips used for action classification inference.")
    parser.add_argument('-vid_ext',type=str,default='.avi',help='video type, can be avi or seq')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    root_experiment_folder = args.root_dir  # where we save all the video experiments
    os.makedirs(root_experiment_folder, exist_ok=True)
    video_folder = args.video_dir
    vid_name = args.video_name
    detector_path = './models/detector.pth'
    cfg_path = './models/SLOWFAST_8x8_R50_feed_pretrained.yaml'
    classifier_path = './models/classifier.pt'
    detector, cfg_detect = load_detector(detector_path, confidence=0.5, nms=0.3)
    classifier, cfg_classify = load_action_classifier(cfg_path, classifier_path, pytorchvideo=True)
    all_vids_dict,vid_lens = get_all_vids_dict()
    clip_sizes = get_clip_sizes(all_vids_dict)
    if vid_name != 'all':
        assert vid_name in all_vids_dict.keys(), \
            'Video is not in the video dict meta data so we do not have annotations for it'
        vid_path = os.path.join(video_folder,vid_name+args.vid_ext)
        if os.path.isfile(vid_path):
            process_video(root_experiment_folder, vid_path, vid_dict=all_vids_dict[vid_name],
                          vid_name=vid_name, clip_size=clip_sizes[vid_name], vid_duration=vid_lens[vid_name],
                      detector=detector, classifier=classifier, classifier_cfg=cfg_classify, start_idx=0,
                          clip_duration=args.clip_duration, save_clips=not args.no_clips)
    else:
        existing_vids_dict = get_existing_dict(video_folder,all_vids_dict)
        print(len(existing_vids_dict))
        assert len(existing_vids_dict.keys())>0, \
            "Oops! video folder doesn't seem to have any of the curated experiment videos, check get_all_vids_dict()"
        for curr_vid, frames_dict in existing_vids_dict.items():
            vid_path = os.path.join(video_folder, curr_vid+args.vid_ext)
            process_video(root_experiment_folder, vid_path, vid_dict=all_vids_dict[curr_vid],
                          vid_name=curr_vid, clip_size=clip_sizes[curr_vid], vid_duration=vid_lens[curr_vid],
                      detector=detector, classifier=classifier, classifier_cfg=cfg_classify, start_idx=0,
                          clip_duration=args.clip_duration, save_clips=not args.no_clips)