import sys
sys.path.append('/media/shirbar/DATA/codes/fish_larvae_feeding')
sys.path.append('/media/shirbar/DATA/codes/SlowFast/')
sys.path.append('/home/shirbar/codes/fish_larvae_feeding')
sys.path.append('/home/shirbar/codes/SlowFast/')
from SEQReader import SEQReader
import torch
import cv2
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.models.ptv_model_builder import PTVSlowFast,PTVResNet
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog


def load_video(path):
    if path.endswith('.seq'):
        vid = SEQReader(path,'<')
        print('Loaded SEQ file')
    elif path.endswith('.avi'):
        vid = cv2.VideoCapture(path)
        print('Loaded avi file as VideoCapture')
    else:
        print('Unsupported file format')
    return vid

def load_detector(path,nms=0.3,confidence=None):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = path
    cfg.DATASETS.TRAIN = ('fish_train',)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    if confidence is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence   # set a custom testing threshold
    MetadataCatalog.get("fish_train").set(thing_classes=["fish", 'blurry_fish'])
    predictor = DefaultPredictor(cfg)
    #model = build_model(cfg)  # returns a torch.nn.Module
    #DetectionCheckpointer(model).load(path)  # load a file, usually from cfg.MODEL.WEIGHTS
    return predictor,cfg


class Args:
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.shard_id = 0
        self.num_shards = 1
        self.init_method = 'tcp://localhost:9999'
        self.opts = None


def load_action_classifier(cfg_path, model_chkpt_path,pytorchvideo=False):
    args = Args(cfg_path)
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    if pytorchvideo:
        model_name = "slowfast_r50"
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=False)
        model.blocks[6].proj = torch.nn.Linear(in_features=2304, out_features=2)
    else:
        model = PTVSlowFast(cfg)  # PTVSlowFast(cfg)#PTVResNet(cfg)##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    model.to(device)
    checkpoint = torch.load(model_chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    return model, cfg



