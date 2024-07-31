import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='enter root path where experiments will be saved')
    parser.add_argument('video_dir', help='enter path to directory containing videos')
    parser.add_argument('-video_name', default='all', help='Specify a single video to analyze, if no name is specified '
                                                           'all videos in the folder will be analyzed')
    parser.add_argument('-clip_duration',type=int,default=80,help='Duration of clips to cut for the action classifier')
    parser.add_argument('-clip_size',type=int,default=400,help='Size of clips to cut for the action classifier')
    parser.add_argument('-no_clips', action='store_true',
                        help="Don't save the clips used for action classification inference.")
    parser.add_argument('-vid_ext',choices=['.avi','.seq'],
                        type=str,default='.seq',help='video type, can be avi or seq')
    parser.add_argument('-dec_thresh', type=float, default=0.75, help='decision threshold for the classifier')
    parser.add_argument('-fish_to_sample',type=int, default=np.inf, 
                        help='how many clips from the positive class to fish before stopping video analysis')
    parser.add_argument('-last_frame', type=int, default=None,
                        help='Last frame to sample')
    parser.add_argument('-first_frame', type=int, default=0,
                        help='Frame to start sampling from, default at 0')
    parser.add_argument('-classify_only',action='store_true',
                        help='Use stored detections (must be in specific csv format)')
    parser.add_argument('-classifier_name',type=str,default='SlowFastSSv2',choices=['SlowFastSSv2','SlowFastKinetics'], help='Name your classifer .pt file')
    parser.add_argument('-detector_name',type=str,default='fasterRCNN',choices=['fasterRCNN','yolov5'],help='Name your detector .pt file')
    parser.add_argument('-video_list_path',type=str, default='', 
                        help='path to a csv with video file names to analyze (a subset from the video data folder)')
    parser.add_argument('-cfg_path',type=str, 
                        default='./models/SLOWFAST_8x8_R50_feed_pretrained.yaml', 
                        help='Path to action classifier cfg.yaml')
    parser.add_argument('-vid_prefix', type=str, default='Seq', 
                        help='Prefix for video files, for searching videos in directory')
    return parser