import os
from tqdm import tqdm
from loaders import *
import random
from parser import get_parser
from utils import get_action_classifications,get_fish_detection,plot_boxes, get_detections_from_preds
from clip_utils import get_clips, save_clip
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(DEVICE)

def load_log(path):
    if os.path.isfile(path):
        df_run_params = pd.read_csv(path)
    else:
        df_run_params = pd.DataFrame(columns=['experiment_name','video_name','fps',
                                              'overlap', 'num_frames_sampled','last_frame_sampled',
                                              'duration sampled','num_frames_with_detections',
                                                  'sample_regime',
                                                  'clip_duration','frame_size',
                                          'num_labeled_strike','num_labeled_swim',
                                                  'total_clips','execution_time'])
    return df_run_params


def make_folders(path):
    #if not os.path.exists(path):
    os.makedirs(os.path.join(path, 'swim'), exist_ok=True)
    os.makedirs(os.path.join(path, 'strike'), exist_ok=True)



def write_clip(folder_path,clip,prediction,frame_num,idx,centroid,vid_name):
    clip_name = ''
    centroid_str = str(centroid[0]) + '-' + str(centroid[1])
    clip_name = f'{vid_name}_midframe_{frame_num}_fish_{idx}_coordinate_[{centroid_str}].avi'
    if prediction == 1:

        movie_path = os.path.join(folder_path, 'strike',clip_name)
    else:
        movie_path = os.path.join(folder_path, 'swim',
                                  clip_name)
    save_clip(clip, movie_path, transformed=False)
    return clip_name

def get_fps(vid, vid_name):
    if vid_name.endswith('.seq'):
        fps = vid.properties['FrameRate']
    else:
        fps = vid.get(cv2.CAP_PROP_FPS)
    return round(fps)
def load_preds(preds_path):
    df = pd.read_csv(preds_path)
    df.centroid = df.centroid.map(
        lambda cent: [int(x) for x in cent.strip('[]').split()] if type(cent) == str else np.NaN)
    df.bboxs = df.bboxs.map(lambda cent: [float(x) for x in cent.strip('[]').split()] if type(cent) == str else np.NaN)
    return df
def only_classify(root_path, vid_path, vid_name, clip_size, classifier, classifier_cfg, clip_duration=80,
                  thresh=0.5, save_clips=True,
                  classifier_name='SSv2'):
    vid = load_video(vid_path)
    fps = get_fps(vid, vid_path)
    all_preds = []
    fish_sampled = 0
    time_sampled = 0
    EXP_NAME = f'{vid_name}_{classifier_name}_exp'
    seq = vid_name.split('_')[0]
    dph = vid_name.split('_')[1][:-3]
    fish = vid_name.split('_')[2].split('.')[0]
    folder_path = os.path.join(root_path,fish,dph,seq)
    make_folders(folder_path)
    preds_path = os.path.join(folder_path, 'preds.csv')

    if not os.path.exists(preds_path):
        print(f'no detections found, cannot classify only, {preds_path}')
        with open(os.path.join(root_path,'log.txt'),'a') as f:
            print(vid_name, file=f)
            print(f'{preds_path} not found', file=f)
        return
    elif os.path.exists(os.path.join(folder_path, f'preds_{classifier_name}.csv')):
        print('already analyzed')
        with open(os.path.join(root_path,'log.txt'),'a') as f:
            print(vid_name, file=f)
            print(f'Already analyzed', file=f)
        return
    df_preds = load_preds(preds_path)
    df_preds[classifier_name+'_strike_scores'] = np.nan
    log_path = os.path.join(root_path,fish,'experiment_log.csv')
    log_df = load_log(log_path)
    frames_with_detections = 0
    start_time = time.time()
    frames = []
    fish_bar = tqdm(total=5)
    for frame_num in sorted(df_preds.frame.unique()):
        comments=''
        try:
            if vid_path.endswith('.seq'):
                frame = vid[frame_num]['frame']
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _, frame = vid.read()
        except:
            comments = "can't read frame from seq"
        entry = df_preds[df_preds.frame == frame_num]
        detect_classes, boxes = get_detections_from_preds(entry)
        if len(boxes) > 0:
            frames_with_detections += 1

            clips,centroids = get_clips(vid, frame_num, boxes,
                                        clip_duration=clip_duration, clip_size=clip_size, num_channels=3)
            preds, pred_classes, pred_class_names,_ = get_action_classifications(classifier, clips, classifier_cfg,
                                                                                                DEVICE,
                                                                                                 bbox_clip_sizes=False,
                                                                                                 normalize=True,
                                                                                                 verbose=False)
            for i, p in enumerate(pred_classes.flatten()):
                thresh_pred = int(preds[i, 0].item() >= thresh)
                if save_clips:
                    clip_name = write_clip(folder_path, clip=clips[i], prediction=thresh_pred, frame_num=frame_num,
                                           idx=i,
                                           vid_name=vid_name, centroid=centroids[i])
                    df_preds.loc[(df_preds.frame == frame_num) & (df_preds.fish_id == float(i)), 'clip_name'] = clip_name
                thresh_pred = int(preds[i, 0].item() >= thresh)
                df_preds.loc[(df_preds.frame==frame_num)&(df_preds.fish_id==float(i)),
                             classifier_name+'_action_preds'] = thresh_pred
                df_preds.loc[(df_preds.frame==frame_num)&(df_preds.fish_id==float(i)),
                             classifier_name+'_strike_scores'] = preds[i, 0].item()
                if len(comments)>0:
                    df_preds.loc[
                        (df_preds.frame == frame_num) & (df_preds.fish_id == float(i)), 'comments'] = comments
            all_preds.append(preds)
            feeding_fish = (preds >= thresh).sum(axis=0)[0].item()
        else:
            feeding_fish = 0

        frames.append(frame_num)
        fish_sampled += feeding_fish  # add predicted strikes to fish count
        prev_time = time_sampled
        time_sampled = ((frames[-1] / fps) / 60) - ((frames[0] / fps) / 60)  # time sampled in minutes
        fish_bar.update(time_sampled - prev_time)
    plt.close()
    fish_bar.close()
    df_preds.to_csv(os.path.join(folder_path, f'preds_{classifier_name}.csv'), index=False)
    execution_time = time.time() - start_time
    tot_strike = fish_sampled
    if len(all_preds)>0:
        tot_swim = (torch.vstack(all_preds)<0.9).sum(axis=0)[0].item()
    else:
        tot_swim = 0


    entry = {'experiment_name': EXP_NAME, 'video_name': vid_name,'DPH':dph,'cohort':fish,'seq':seq,'fps':fps,
                 'overlap': 0.25, 'num_frames_sampled': len(frames),'last_frame_sampled':frames[-1],
                'duration sampled': time_sampled,
                 'num_frames_with_detections': frames_with_detections,
                 'sample_regime': '5min', 'clip_duration': clip_duration,
                 'frame_size': clip_size,
                 'num_labeled_strike': tot_strike,
                 'num_labeled_swim': tot_swim,
                 'total_clips': tot_swim + tot_strike, 'execution_time': execution_time}
    log_df = log_df.append(entry, ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(log_path)
    print(execution_time)
    vid.release()



def fish_in_vid(root_path, vid_path, vid_name, clip_size,
                  detector, classifier, classifier_cfg, clip_duration=80,
                  thresh=0.5, fish_to_sample=1000, save_clips=True, last_frame=None, start_idx=0):
    vid = load_video(vid_path)
    if vid_path.endswith('.seq'):
        vid_len = len(vid)
    else:
        vid_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = get_fps(vid, vid_path)
    all_preds = []
    fish_sampled = 0
    time_sampled = 0
    frame_num = start_idx + round(clip_duration/2)
    #os.makedirs(os.path.join(root_path,'sample_frames'), exist_ok=True)
    EXP_NAME = f'{vid_name}_RDag_exp'
    seq = vid_name.split('_')[0]
    dph = vid_name.split('_')[1][:-3]
    fish = vid_name.split('_')[2].split('.')[0]
    folder_path = os.path.join(root_path,fish,dph,seq)
    make_folders(folder_path)
    os.makedirs(os.path.join(root_path,fish, 'sample_frames'), exist_ok=True)
    df_preds = pd.DataFrame(columns=['vid_name','frame','clip_name',
                                     'fish_id','centroid','bboxs','detection_scores','detection_pred_class',
                                     'action_preds','strike_scores','comments'])
    log_path = os.path.join(root_path,fish,'experiment_log.csv')
    log_df = load_log(log_path)
    frames_with_detections = 0
    start_time = time.time()
    frames = []
    fish_bar = tqdm(total=5)
    if not last_frame:
        last_frame = vid_len-41
    else:
        last_frame = last_frame - 41
    while ((fish_sampled <= fish_to_sample) and (frame_num<(last_frame)) and (time_sampled<=5)):
        comments = ''
        try:
            if vid_path.endswith('.seq'):
                frame = vid[frame_num]['frame']
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _,frame = vid.read()
        except:
            comments = "can't read frame from seq"
        outputs,boxes = get_fish_detection(detector, frame)
        if len(all_preds) == 0 and len(boxes) > 0:
            plot_boxes(frame, outputs, save=True, filepath=os.path.join(root_path,fish, 'sample_frames',
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
                df_preds = df_preds.append({'vid_name':vid_name,'frame':frame_num,'clip_name':clip_name,
                                            'fish_id':i,'centroid':centroids[i],
                                            'bboxs':list(boxes.values())[i],
                                        'detection_scores':list(boxes.keys())[i],
                                            'detection_pred_class':outputs['instances'].pred_classes[i].item(),
                                            'action_preds':thresh_pred,
                                            'strike_scores':preds[i,0].item(),'comments':comments,},
                                   ignore_index=True)
            all_preds.append(preds),
            feeding_fish = (preds>=thresh).sum(axis=0)[0].item()

        else:
            df_preds = df_preds.append({'vid_name':vid_name,'frame':frame_num,'clip_name':None,
                                        'fish_id':None,'centroid':None, 'bboxs':None,
                                        'detection_scores':None,'detection_pred_class':None,
                                            'action_preds':None, 'strike_scores':None,'comments':comments},
                                   ignore_index=True)
            feeding_fish = 0

        frames.append(frame_num)
        frame_num = frame_num + round(clip_duration*0.75)
        fish_sampled += feeding_fish  # add predicted strikes to fish count
        prev_time = time_sampled
        time_sampled = ((frames[-1]/fps)/60)-((frames[0]/fps)/60)  # time sampled in minutes
        fish_bar.update(time_sampled-prev_time)
    plot_boxes(frame, outputs, save=True, filepath=os.path.join(root_path, fish, 'sample_frames',
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


    entry = {'experiment_name': EXP_NAME, 'video_name': vid_name,'DPH':dph,'cohort':fish,'seq':seq,'fps':fps,
                 'overlap': 0.25, 'num_frames_sampled': len(frames),'last_frame_sampled':frames[-1],
                'duration sampled': time_sampled,
                 'num_frames_with_detections': frames_with_detections,
                 'sample_regime': '5min', 'clip_duration': clip_duration,
                 'frame_size': clip_size,
                 'num_labeled_strike': tot_strike,
                 'num_labeled_swim': tot_swim,
                 'total_clips': tot_swim + tot_strike, 'execution_time': execution_time}
    log_df = log_df.append(entry, ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(log_path)
    print(execution_time)
    vid.release()

def get_file_list(dir_to_scan, vid_extension, custom_file_list=[]):
    file_list = []
    ls =  np.hstack(custom_file_list.values)
    #ls = [l.lower() for l in ls]
    #print(ls)
    #assert(False)
    for root, dirs, files in os.walk(dir_to_scan):
        for file in files:
            if file.endswith(vid_extension) & file.startswith('Seq') & ('cali' not in file.lower()):
                if (len(custom_file_list)==0) or (file.split('.')[0] in ls):
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
    detector_path = './models/detector.pth'
    cfg_path = args.cfg_path
    classifier_path = f'./models/{args.classifier_name}.pt'
    detector, cfg_detect = load_detector(detector_path, confidence=0.5, nms=0.3)
    classifier, cfg_classify = load_action_classifier(cfg_path, classifier_path, pytorchvideo=True)

    if vid_name != 'all':
        vid_path = os.path.join(video_folder,vid_name+args.vid_ext)
        if os.path.isfile(vid_path):
            if args.classify_only:
                only_classify(root_experiment_folder, vid_path,
                          vid_name=vid_name, clip_size=args.clip_size, classifier=classifier, classifier_cfg=cfg_classify,
                        clip_duration=args.clip_duration,
                        save_clips=not args.no_clips,thresh=args.dec_thresh, classifier_name=args.classifier_name)
            else:
                fish_in_vid(root_experiment_folder, vid_path,
                          vid_name=vid_name, clip_size=args.clip_size,
                      detector=detector, classifier=classifier, classifier_cfg=cfg_classify,
                        fish_to_sample=args.fish_to_sample, clip_duration=args.clip_duration,
                        save_clips=not args.no_clips,thresh=args.dec_thresh, last_frame=args.last_frame)
        else:
            print(f'Video file not found in path {vid_path}')
    else:

        if len(args.video_list_path)>0:
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
            curr_vid = os.path.basename(curr_path).split('.')[0]
            print(f'Vid {curr_vid} at work')
            if args.classify_only:
                only_classify(root_experiment_folder, curr_path,
                                  vid_name=curr_vid, clip_size=args.clip_size, classifier=classifier,
                                  classifier_cfg=cfg_classify,
                                  clip_duration=args.clip_duration,
                                  save_clips=not args.no_clips, thresh=args.dec_thresh,
                                  classifier_name=args.classifier_name)
            else:
                fish_in_vid(root_experiment_folder, curr_path,
                          vid_name=curr_vid, clip_size=args.clip_size,
                      detector=detector, classifier=classifier, classifier_cfg=cfg_classify,
                        fish_to_sample=args.fish_to_sample,clip_duration=args.clip_duration,
                        save_clips=not args.no_clips, thresh=args.dec_thresh, last_frame=args.last_frame,
                        start_idx=args.first_frame)
#
# def fish_in_video(root_path, vid_path, vid_dict, vid_name, clip_size, vid_duration,
#                   detector, classifier, classifier_cfg, start_idx=0, clip_duration=80, save_clips=True):
#     vid = load_video(vid_path)
#     all_preds = []
#     all_classes = []
#     nboxes = {}
#     frames_to_extract = np.sort(vid_dict['frames_to_extract'])
#     midframes = vid_dict['midframes']
#     images_to_save = random.sample(list(midframes),3)
#     os.makedirs(os.path.join(root_path,'sample_frames'), exist_ok=True)
#     for frame_num in images_to_save:
#         frame = vid[frame_num]['frame']
#         outputs, boxes = get_fish_detection(detector, frame)
#         plot_boxes(frame, outputs, save=True,filepath=os.path.join(root_path,'sample_frames',
#                                                                    f'{vid_name}_frame_{frame_num}.jpg'))
#         plt.close()
#     EXP_NAME = f'{vid_name}_RDag_exp'
#     folder_path = os.path.join(root_path,EXP_NAME)
#     os.makedirs(folder_path,exists_ok=True)
#
#     df_preds = pd.DataFrame(columns=['frame','fish_id','centroid','bboxs','detection_scores','detection_pred_class'])
#     log_path = os.path.join(root_path,'experiment_log.csv')
#     log_df = load_log(log_path)
#     frames_with_detections = 0
#     start_time = time.time()
#     for frame_num in frames_to_extract[start_idx:]:
#         print(frame_num)
#         frame = vid[frame_num]['frame']
#         outputs,boxes = get_fish_detection(detector, frame)
#         nboxes[frame_num] = len(boxes)
#
#         if len(boxes)>0:
#             frames_with_detections +=1
#
#             clips,centroids = get_clips(vid, frame_num, boxes,
#                                         clip_duration=clip_duration, clip_size=clip_size,num_channels=3)
#
#             for i,p in enumerate(pred_classes.flatten()):
#                 if save_clips:
#                     movie_path = os.path.join(folder_path, 'clips',
#                                               f'midframe_{frame_num}_fish_{i}.avi')
#                     save_clip(clips[i], movie_path, transformed=False)
#                 df_preds = df_preds.append({'frame':frame_num,'fish_id':i,'centroid':centroids[i],
#                                             'bboxs':list(boxes.values())[i],
#                                         'detection_scores':list(boxes.keys())[i],
#                                             'detection_pred_class':outputs['instances'].pred_classes[i].item(),
#                                             'action_preds':p.item(), 'strike_scores':preds[i,0].item()},
#                                    ignore_index=True)
#             all_classes.append(pred_classes)
#             all_preds.append(preds)
#         else:
#             df_preds = df_preds.append({'frame':frame_num,'fish_id':None,'centroid':None, 'bboxs':None,
#                                         'detection_scores':None,'detection_pred_class':None,
#                                             'action_preds':None, 'strike_scores':None},
#                                    ignore_index=True)
#     df_preds.to_csv(os.path.join(folder_path, 'preds.csv'), index=False)
#     execution_time = time.time() - start_time
#     tot_strike = len(os.listdir(os.path.join(folder_path, 'strike', 'original')))
#     tot_swim = len(os.listdir(os.path.join(folder_path, 'swim', 'original')))
#
#     entry = {'experiment_name': EXP_NAME, 'video_name': vid_name, 'video_duration': vid_duration,
#                  'overlap': 0, 'num_frames_sampled': len(frames_to_extract),
#                  'num_frames_with_detections': frames_with_detections,
#                  'sample_regime': 'directed', 'clip_duration': clip_duration,
#                  'frame_size': clip_size, 'TP': np.NaN,
#                  'num_labeled_strike': tot_strike,
#                  'FN': np.NaN,
#                  'num_labeled_swim': tot_swim,
#                  'total_clips': tot_swim + tot_strike, 'execution_time': execution_time}
#     log_df = log_df.append(entry, ignore_index=True)
#     log_df.to_csv(log_path, index=False)
#     print(execution_time)
#     vid.release()