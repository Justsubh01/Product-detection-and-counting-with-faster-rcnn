from data_preparation import get_data_loaders
import argparse
import numpy as np
from tqdm import tqdm
import json
import time
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SequentialSampler
from albumentations.pytorch.transforms import ToTensorV2
import torch.backends.cudnn as cudnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import emoji


header = r'''
                   Predictions
            mAP  |  precision  |  recall  | 
'''
raw_line = '             {:6.2f}' + '\u2502{:6.2f}' + '\u2502{:6.2f}'

def make_model(weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    WEIGHTS_FILE = weights
    num_classes = 12
     # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Load the trained weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(WEIGHTS_FILE))
        model = model.to(device)
    else:
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=lambda storage, loc: storage))
        model = model.to(device)
    return model,device

# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union

def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.50]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    precision = 0
    recall = 0
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        try:
            p = tp/(tp+ fp)
        except ZeroDivisionError:
            p=0.0
        try:
            r = tp/(tp + fn)
        except ZeroDivisionError:
            r=0.0
        map_total += m
        precision += p
        recall += r
    
    return map_total / len(thresholds),precision/len(thresholds),recall/len(thresholds)

def sumOfList(list, size):
   if (size == 0):
     return 0
   else:
     return list[size - 1] + sumOfList(list, size - 1)


def test(batch_size=4,weights="saved_weights.pth",iou_thres=0.5):

    model,device = make_model(weights)
    model.eval()

    valid_data_loader = get_data_loaders(valid_batch_size=batch_size)

    start_time = time.time()
    avr = []
    print(emoji.emojize("\nLets wait for evolution.. 🧘‍♂️"))
    out_precision,out_recall = [],[]
    for images,targets in tqdm(valid_data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        prediction = model(images)
        for pred,target in zip(prediction,targets):       
            pred_boxes = pred["boxes"].data.cpu().numpy()
            pred_scores = pred["scores"].data.cpu().numpy()
            boxes_pred = np.array(pred_boxes)
            boxes_true = target["boxes"]
            scores = pred_scores
            out,precision,recall = map_iou(boxes_true,boxes_pred,scores,thresholds=[iou_thres])
            out_precision.append(precision)
            out_recall.append(recall)
            avr.append(out)
    print(header)
    sum_of_ap = sumOfList(avr,len(avr))/len(avr)
    sum_of_pre = sumOfList(out_precision,len(out_precision))/len(out_precision)
    sum_of_recall = sumOfList(out_recall,len(out_recall))/len(out_recall)
    print(raw_line.format(sum_of_ap,sum_of_pre,sum_of_recall))
    print("--- %s seconds ---" % (time.time() - start_time))
    if sum_of_ap > 0.5:
        metrics_dict = {}
        metrics_dict["mAP"] = sum_of_ap
        metrics_dict["precision"] = sum_of_pre
        metrics_dict["recall"] = sum_of_recall
        with open('metrics.json','w',encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=4)            

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",type=int,default=4,help="total batch size for all GPUs")
    parser.add_argument("--weights",default="saved_weights.pth",help="model.pt path(s)")
    parser.add_argument("--iou-thres",type=float,default=0.5, help="confidence threshold")
    opt = parser.parse_args()
    return opt

def main(opt):
    test(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)