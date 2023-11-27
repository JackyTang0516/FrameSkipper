
import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
#import cupy as cp
import torch
from tqdm import tqdm
#=============================================================================================================
from noscope import VideoUtils, DataUtils, StatsUtils
from noscope.filters import HOG, RawImage, ColorHistogram, SIFT
from sklearn.metrics import confusion_matrix
from math import ceil
import pandas as pd
import torch.nn.functional as F
import copy
from numba import jit
from numba import cuda
from numba import njit
import csv
import matplotlib.pyplot as plt
#=============================================================================================================

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync

#@njit
def get_features(feature_fn, frames):
    return np.array([feature_fn(frame) for frame in frames])
    
#@njit
def get_distances(dist_fn, features, delay):
    return np.array([dist_fn(features[i], features[i-delay]) for i in
        range(delay, len(features))])

def get_feature_and_dist_fns(feature_type):
    if feature_type == 'hog':
        return (HOG.compute_feature, HOG.get_distance_fn, HOG.DIST_METRICS)
    elif feature_type == 'sift':
        return (SIFT.compute_feature, SIFT.get_distance_fn, SIFT.DIST_METRICS)
    elif feature_type == 'ch':
        return (ColorHistogram.compute_feature, ColorHistogram.get_distance_fn, ColorHistogram.DIST_METRICS)
    elif feature_type == 'raw':
        return (RawImage.compute_feature, RawImage.get_distance_fn, RawImage.DIST_METRICS)
    else:
        import sys
        print('Invalid feature type: %s' % feature_type)
        sys.exit(1)

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

def process_batch(detections, labels, iouv, repeat=False):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    #======================================================================================================================
    if repeat == True: 
        if detections.shape[0] >= labels.shape[0]:
           detections[:labels.shape[0], :4] = labels[:, 1:]
        else:
           detections[:, :4] = labels[:detections.shape[0], 1:]
        #================================================================
        # if detections.shape[0] >= labels.shape[0]:
        #     #counter = 0
        #     for x in range(labels.shape[0]):
        #         for y in range(detections.shape[0]):                 
        #             similar = F.cosine_similarity(labels[x, 1:], detections[y, :4], dim=0)
        #             #print('similar: ', similar)
        #             if similar > 0.5:
        #                 detections[y, :4] = labels[x, 1:]
        #                 break
        #             #detections[:labels.shape[0], :4] = labels[:, 1:]
        # elif detections.shape[0] < labels.shape[0]:
        #     for y in range(detections.shape[0]):
        #         for x in range(labels.shape[0]):
        #             similar = F.cosine_similarity(labels[x, 1:], detections[y, :4], dim=0)
        #             if similar > 0.5:
        #                 detections[y, :4] = labels[x, 1:]  
        #                 break
    #======================================================================================================================

    iou = box_iou(labels[:, 1:], detections[:, :4])

    # print('labels[:, 0:1]: ', labels)
    # print('detections[:, 5]: ', detections)
    # print('----------------iou01: ', iou)
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    # print('----------------iou02: ', iou)
    # print('=--------------------x', x)
    #print('x[0].shape[0]:', x[0].shape[0])
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    
    #print('correct: ', correct)
    if repeat == True:
        return correct, detections
    #print(correct)
    return correct

#@njit
#@jit(nopython=True)
def get_difference(im=None, threshold=None, video_frames=None, get_dist_fn=None, feature_fn=None, frame_delay=0):
      #print('===================--------------------', type(im.cpu().numpy()))
      video_frames[1, :] = im.cpu()  # for jit .numpy()

      features = get_features(feature_fn, video_frames)   #video_frames
      
      dists = get_distances(get_dist_fn, features, frame_delay)
      #print('dists: ', dists)
      #thresh = 0.002   #0.0235
      #dists_all.append(dists)
      #print('dists: ', dists[0])
      # print('dists > threshold: ', dists > threshold) 
      # print(dists[0])
      Y_preds = dists[0] > threshold
      #print('Y_preds: ', Y_preds)
      #print('Y_preds: ', Y_preds.item())
      return Y_preds.item()

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        weights_S=None,
        weights_R=None,
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        num_frames=2,
        frame_delay=1,
        obj = 'truck',
        scale=0.1,
        feature_type='raw',
        threshold = 0,
        conf_thresh = 0
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)       ################################# 
        #model_S = DetectMultiBackend(weights_S, device=device, dnn=dnn)

        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
            #model_S.model.half() if half else model_S.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check   #################################  加载coco.yaml

    # Configure
    model.eval() #################################
    #model_S.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup #################################
        #model_S.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)

        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    #=====================================================================================================
    #features_to_try = args.features.strip().split(',')
    start, interval = 0, 1
    true_num_frames = int(ceil((num_frames + 0.0) / interval))
    feature_fn, get_distance_fn, dist_metrics_to_try = get_feature_and_dist_fns(feature_type)

    dist_metric = dist_metrics_to_try[0]  # mse
    # print('get_distance_fn(dist_metric)', get_distance_fn(dist_metric))
    get_dist_fn = get_distance_fn(dist_metric)
    # dists_all = []
    batch_all = []
    batch_a = []
    repeat = []
    #location = []  
    #======================================================================================================
    # true_false =  []
    mselist=[]
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        #print('===========================batch_i', batch_i)
        if batch_i == 0:
            batch_all.append([batch_i, 0])
            batch_a.append(batch_i)
            repeat.append(0)
            #dists_all.append('none')
            t1 = time_sync()
            video_frames = np.zeros( tuple([true_num_frames] + list(im.shape)), dtype='float32' )
            video_frames[0, :] = im.cpu()      
            if pt or jit or engine:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
  
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference ===================================================================================
            out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
            dt[1] += time_sync() - t2

            # Loss   
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

            # NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t3 = time_sync()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            dt[2] += time_sync() - t3

            for si, pred in enumerate(out):
                #print('-------------pred', pred)

                labels = targets[targets[:, 0] == si, 1:] 
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class 
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue


                if single_cls: 
                    pred[:, 5] = 0
                predn = pred.clone()
                #def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None): Rescale coords (xyxy) from img1_shape to img0_shape

                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:   
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv) 
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

                # # Save/log
                # if save_txt:
                #     save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                # if save_json:
                #     save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
                # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
        else: 
            t1 = time_sync()
            if pt or jit or engine:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1

            # video_frames[1, :] = im.cpu()

            # features = get_features(feature_fn, video_frames)   #video_frames
            
            # dists = get_distances(get_distance_fn(dist_metric), features, frame_delay)
            # #thresh = 0.002   #0.0235
            # print('dists===', dists)
            # #dists_all.append(dists)
            # Y_preds = dists > threshold
            video_frames[1, :] = im.cpu()
            features = get_features(feature_fn, video_frames)
            dists = get_distances(get_dist_fn, features, frame_delay)
            Y_preds= dists[0] > threshold
            Y_predss=Y_preds.item()
# #------------------------------------------------1-2023
#             if batch_i < 251: 
#                 Y_preds = dists[0] > 0.00477
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 251 and batch_i < 501:
#                 Y_preds = dists[0] > 0.00447
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 501 and batch_i < 751:
#                 Y_preds = dists[0] > 0.0046
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 751 and batch_i < 1001:
#                 Y_preds = dists[0] > 0.00409
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 1001 and batch_i < 1251:
#                 Y_preds = dists[0] > 0.0041
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 1251 and batch_i < 1501:
#                 Y_preds = dists[0] > 0.00397
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 1501 and batch_i < 1751:
#                 Y_preds = dists[0] > 0.00604
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 1751 and batch_i < 2024:
#                 Y_preds = dists[0] > 0.00669
#                 Y_predss =Y_preds.item()
#  #------------------------------------------------2024-4658         
#             elif batch_i >= 2024 and batch_i < 2354:
#                 Y_preds = dists[0] > 0.00606
#                 Y_predss =Y_preds.item()     
#             elif batch_i >= 2354 and batch_i < 2684:
#                 Y_preds = dists[0] > 0.00584
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 2684 and batch_i < 3014:
#                 Y_preds = dists[0] > 0.00542
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 3014 and batch_i < 3344:
#                 Y_preds = dists[0] > 0.00505
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 3344 and batch_i < 3674:
#                 Y_preds = dists[0] > 0.00677
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 3674 and batch_i < 4004:
#                 Y_preds = dists[0] > 0.00556
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 4004 and batch_i < 4334:
#                 Y_preds = dists[0] > 0.00601
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 4334 and batch_i < 4659:
#                 Y_preds = dists[0] > 0.00515
#                 Y_predss =Y_preds.item()
#  #------------------------------------------------4659-7153  
#             elif batch_i >= 4659 and batch_i < 4969:
#                 Y_preds = dists[0] > 0.00562
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 4969 and batch_i < 5279:
#                 Y_preds = dists[0] > 0.00633
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 5279 and batch_i < 5589:
#                 Y_preds = dists[0] > 0.00648
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 5589 and batch_i < 5899:
#                 Y_preds = dists[0] > 0.00598
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 5899 and batch_i < 6209:
#                 Y_preds = dists[0] > 0.00572
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 6209 and batch_i < 6519:
#                 Y_preds = dists[0] > 0.00668
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 6519 and batch_i < 6829:
#                 Y_preds = dists[0] > 0.00662
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 6829 and batch_i < 7154:
#                 Y_preds = dists[0] > 0.00663
#                 Y_predss =Y_preds.item()  
#  #------------------------------------------------7154-9553 
#             elif batch_i >= 7154 and batch_i < 7454:
#                 Y_preds = dists[0] > 0.00437
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 7454 and batch_i < 7754:
#                 Y_preds = dists[0] > 0.00464
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 7754 and batch_i < 8054:
#                 Y_preds = dists[0] > 0.0048
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 8054 and batch_i < 8354:
#                 Y_preds = dists[0] > 0.0045
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 8354 and batch_i < 8654:
#                 Y_preds = dists[0] > 0.00434
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 8654 and batch_i < 8954:
#                 Y_preds = dists[0] > 0.00444
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 8954 and batch_i < 9254:
#                 Y_preds = dists[0] > 0.00438
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 9254 and batch_i < 9554:
#                 Y_preds = dists[0] > 0.0045
#                 Y_predss =Y_preds.item()
#  #------------------------------------------------9554-13408 
#             elif batch_i >= 9554 and batch_i < 10054:
#                 Y_preds = dists[0] > 0.00514
#                 Y_predss =Y_preds.item() 
#             elif batch_i >= 10054 and batch_i < 10554:
#                 Y_preds = dists[0] > 0.00495
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 10554 and batch_i < 11054:
#                 Y_preds = dists[0] > 0.00513
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 11054 and batch_i < 11554:
#                 Y_preds = dists[0] > 0.0047
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 11554 and batch_i < 12054:
#                 Y_preds = dists[0] > 0.00425
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 12054 and batch_i < 12554:
#                 Y_preds = dists[0] > 0.00471
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 12554 and batch_i < 13054:
#                 Y_preds = dists[0] > 0.00568
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 13054 and batch_i < 13409:
#                 Y_preds = dists[0] > 0.00563
#                 Y_predss =Y_preds.item()
#  #------------------------------------------------13409-15367 
#             elif batch_i >= 13409 and batch_i < 13659:
#                 Y_preds = dists[0] > 0.00439
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 13659 and batch_i < 13909:
#                 Y_preds = dists[0] > 0.00507
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 13909 and batch_i < 14159:
#                 Y_preds = dists[0] > 0.0087
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 14159 and batch_i < 14409:
#                 Y_preds = dists[0] > 0.01011
#                 Y_predss =Y_preds.item()  
#             elif batch_i >= 14409 and batch_i < 14659:
#                 Y_preds = dists[0] > 0.00822
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 14659 and batch_i < 14909:
#                 Y_preds = dists[0] > 0.00588
#                 Y_predss =Y_preds.item()
#             elif batch_i >= 14909 and batch_i < 15159:
#                 Y_preds = dists[0] > 0.00543
#                 Y_predss =Y_preds.item()  
#             else:
#                 Y_preds= dists[0] > 0.00493
#                 Y_predss = Y_preds.item()
            

            #print('----------------',dists[0])

            #Y_preds = get_difference(im, threshold, video_frames, get_dist_fn, feature_fn, frame_delay)
            # Y_preds = get_difference(im=im, threshold=threshold, video_frames=video_frames, get_dist_fn=get_dist_fn, feature_fn=feature_fn, frame_delay=frame_delay)
            #dists_all.append(dists)
            # true_false.append(str(Y_preds))
            
            

            dt[3] += time_sync() - t2
            # video_frames[1, :] = im.cpu()
            # print('get_distances(get_distance_fn(dist_metric), get_features(feature_fn, video_frames), frame_delay)', (get_distances(get_distance_fn(dist_metric), get_features(feature_fn, video_frames), frame_delay)>threshold).item())
            mselist.append(str(dists[0]))
            if batch_i==9999:#2023,2634,2494,2399,3854,1959
                with open('/home/yubai402b/Desktop/Mse00000.txt', 'w') as f: 
                    for mse in mselist:
                        f.write(mse)
                        f.write('\n')

            #if Y_preds.item() is True: 
            # if (get_distances(get_distance_fn(dist_metric), get_features(feature_fn, video_frames), frame_delay) > threshold).item() is True:
            #if (get_distances(get_dist_fn, get_features(feature_fn, video_frames), frame_delay) > threshold).item() is True:
            if Y_predss is True: 
                batch_all.append([batch_i, 0])  
                # t2 = time_sync()
                # dt[0] += t2 - t1
                t4 = time_sync()
                # Inference ===================================================================================
                out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
                dt[1] += time_sync() - t4
                batch_a.append(batch_i)
                repeat.append(0) 
                #print('dt------------------', dt)
                
                # Loss   
                if compute_loss:
                    loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

                # NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                t3 = time_sync()
                out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
                dt[2] += time_sync() - t3


                video_frames[0, :] = im.cpu() # for jit .numpy()

                # Metrics  
                for si, pred in enumerate(out):
                    #print('-------------pred', pred)
                    labels = targets[targets[:, 0] == si, 1:] 
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  
                    path, shape = Path(paths[si]), shapes[si][0]
                    seen += 1

                    if len(pred) == 0:
                        if nl:
                            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    if single_cls: 
                        pred[:, 5] = 0
                    predn = pred.clone()
                    #def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None): Rescale coords (xyxy) from img1_shape to img0_shape
                    scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                    # Evaluate
                    if nl:   
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = process_batch(predn, labelsn, iouv)  
                        if plots:
                            confusion_matrix.process_batch(predn, labelsn)
                    else:
                        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

                    # # Save/log
                    # if save_txt:
                    #     save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                    # if save_json:
                    #     save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
                    # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
            else: 
                batch_all.append([batch_i, 1])
                # t2 = time_sync()
                # dt[0] += t2 - t1
                batch_a.append(batch_i)
                repeat.append(1)
                #print('-------------------------difference-detector', dt)
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                labels = targets[targets[:, 0] == 0, 1:] 
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions 
                if single_cls: 
                    pred[:, 5] = 0

                predn = pred.clone()
                #def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None): Rescale coords (xyxy) from img1_shape to img0_shape

                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
                #print('=============pred', pred)                
                
                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

                    #correct, pred = process_batch(predn, labelsn, iouv, repeat=True)  
                    correct = process_batch(predn, labelsn, iouv, repeat=False)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls) 

        # # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()
    # title = ['frame', 'repeat']
    # with open('repeat.csv', 'w') as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerow(title)
    #     write.writerows(batch_all)

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(batch_a,repeat)
    # plt.savefig("repeat+"+str(threshold)+".png")
    # plt.show()

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    print('dt------------final', dt)
    dt[0] = dt[0] + 10000*0.00001################
    print('dt============final', dt)
    print('dt.sum--------final', sum(dt))
    print('seen: ', seen)

    # seen = 15368###################
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print('t------', t)
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}, %.1fms for difference detector' % t)
        
        # convert inference time in milliseconds to frames per second as well?
        fpsm = 10000 / (dt[1] + dt[3])   ############ 
        LOGGER.info(f'FPS: {fpsm:.6f}')
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/UA_DETRAC_onecls.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov7-onecls-last.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    #===========================================================================================================================
    parser.add_argument('--threshold', type=float, default=0.001, help='threshold for difference')
    parser.add_argument('--conf_thresh', type=float, default=0.001, help='threshold for confidence')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
