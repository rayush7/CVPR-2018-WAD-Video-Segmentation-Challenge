from skimage.io import imread

import tqdm
import numpy as np

category2id = {'car':          33,
               'motorbicycle': 34, 
               'bicycle':      35,
               'person':       36, 
               'truck':        38,
               'bus':          39,
               'tricycle':     40,}
id2category = {}
for cat, _id in category2id.items():
    id2category[_id] = cat

    
def compute_iou(path_true, path_pred):
    iou = {}
    for obj, class_id in category2id.items():
        print('Computing iou for', obj, '...')
        iou[obj] = mean_iou(path_true, path_pred, class_id)
    return iou
    
def mean_iou(path_true, path_pred, class_id):
    path_true.sort()
    path_pred.sort()
    iou = 0
    count = 0
    for p_true, p_pred in tqdm.tqdm(zip(path_true, path_pred)):
        label_true = imread(p_true)
        label_true = np.ndarray.astype(label_true/1000, np.int32)
        label_pred = imread(p_pred)
        label_pred = np.ndarray.astype(label_pred/1000, np.int32)
        
        mask_true = (label_true==class_id)
        mask_pred = (label_pred==class_id)

        if np.sum(mask_true)==0.:
            continue
        else:
            intersect = np.sum(mask_true&mask_pred)
            union = np.sum(mask_true|mask_pred)
            iou += intersect/union
            count += 1
    
    if count>0:
        return iou/count
    else:
        return -1
        