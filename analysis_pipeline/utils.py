import sys
sys.path.append('/media/shirbar/DATA/codes/SlowFast/')
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import matplotlib.pyplot as plt
from clip_utils import transform_clips
import numpy as np
import json

IDX_TO_CLASSES = {0:'feed', 1:'swim'}

def get_fish_detection(predictor, frame):
    if len(frame.shape) == 2:
        im = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    else:
        im = frame
    outputs = predictor(im)
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    scores = outputs['instances'].scores.cpu().numpy()
    boxes_dict = {}
    for score,box in zip(scores,boxes):
        boxes_dict[score] = box
    return outputs, boxes_dict

def get_detections_from_preds(entry):
    boxes_dict = {}
    for v, k in zip(entry.bboxs, entry.detection_scores):
        if type(v)==list:
            boxes_dict[k] = np.array(v)
    pred_classes = entry.detection_pred_class
    return pred_classes,boxes_dict

def get_batch(clips):
    slow = []
    fast = []
    for clip in clips:
        slow.append(clip[0])
        fast.append(clip[1])
    batch = [torch.stack(slow), torch.stack(fast)]
    return batch

def get_batches(clips,batch_size=4):
    batchs = []
    for i in range(0,len(clips),batch_size):
        batchs.append(get_batch(clips[i:i+batch_size]))
    return batchs

def get_action_classifications(model, clips, cfg, device, verbose=True, normalize=True, bbox_clip_sizes=False):
    transformed_clips = transform_clips(clips,cfg, normalize=normalize)
    model.eval()
    if bbox_clip_sizes:
        preds = []
        with torch.no_grad():
            for clip in transformed_clips:
                batch = get_batch([clip])
                batch = [i.to(device) for i in batch]
                preds.append(model(batch))

        
        preds = torch.vstack(preds)
    else:
        batchs = get_batches(transformed_clips,batch_size=1)
        for batch in batchs:
            batch = [i.to(device) for i in batch]
            with torch.no_grad():
                preds = model(batch)
    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=1).indices
    # Map the predicted classes to the label names
    pred_class_names = [IDX_TO_CLASSES[int(i)] for i in pred_classes]
    if verbose:
        print("Top 1 predicted labels for each sample: \n%s" % ", \n".join(pred_class_names))
    return preds,pred_classes, pred_class_names, transformed_clips


def plot_boxes(frame, predictions, save=False, filepath=''):
    if len(frame.shape)==2:
        im=cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    else:
        im=frame
    fish_metadata = MetadataCatalog.get("fish_train")
    v = Visualizer(im[:, :, ::-1],
                       metadata=fish_metadata,
                       scale=0.5,
        )
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    plt.figure(figsize=(20,10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    if save:
        plt.savefig(filepath)


def get_dicts_from_coco(js):
        imgs_dict = {}
        annts_dict = {}
        for image in js['images']:
            imgs_dict[image['file_name']] = image['id']
        for annts in js['annotations']:
            if annts['image_id'] not in annts_dict.keys():
                annts_dict[annts['image_id']] = [annts]
            else:
                annts_dict[annts['image_id']].append(annts)
        return imgs_dict, annts_dict


def get_annts_by_frame(json_filepath):
    with open(json_filepath, 'r') as f:
        js = json.load(f)
    imgs, annts = get_dicts_from_coco(js)
    new_annts = {}
    for img_id, an in annts.items():
        img_name = list(imgs.keys())[list(imgs.values()).index(img_id)]
        frame = int(img_name.split('_')[-1].split('.')[0])
        boxes = []
        for a in an:
            boxes.append(a['bbox'])
        boxes = np.array(boxes)
        boxes = np.unique(boxes, axis=0)
        boxes[:, 2:4] += boxes[:, 0:2]
        new_annts[frame] = boxes
    return new_annts