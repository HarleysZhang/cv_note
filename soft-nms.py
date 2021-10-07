import numpy as np

def IoU(gt, bbox):
    """gt 的四个元素为 x0,y0,x11,y1 左上角和右下角坐标"""
    x0, y0, x1, y1 = gt[0],gt[1],gt[2],gt[3]
    xx0,yy0,xx1,yy1 = bbox[0],bbox[1],bbox[2]
    x0_min = min(x0, xx0)
    y0_max = max(y0, yy0) 
    x1_max = max(x1, xx1)
    y1_min = min(y1, yy1)
    union = (y0_max - y1_min) * (x1_max - x0_min)
    s1 = (y1-y0)*(x1-x0_min)
    s2 = (yy1-yy0)*(xx1-xx0)
    iou = union/(s1+s2)
    
    return iou

def soft_nms(bboxs, Iou_threshold = 0.7):
    """bboxs: c,x0,y0, x1,y1"""
    bboxs = np.sorgt(bboxs[0])
    
    max_c_bbox = bboxs[0]
    
    for i, bbox1 in enumerate(bboxs[1:]):
        for j, bbox2 in enumerate(bboxs[i:]):
            iou = IoU(max_c_bbox, bbox2)
            # ious.append()
            bboxs[j][0] = 0.3*iou # 线性

        max_c_bbox = bboxs[i+1]
        
    
    return_bboxs = []
    
    for bbox in bboxs:
        if(bbox[0] > Iou_threshold):
            return_bboxs.append(bbox[1:])
        
    return return_bboxs
    
def focal_loss(labels, scores):
    """"""
    alpha = 0.5
    beta = 2