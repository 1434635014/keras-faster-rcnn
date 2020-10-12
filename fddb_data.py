from cv2 import cv2 as cv
import os

import numpy as np
from utils import shapeData as dataSet
from config import Config as config

os_path = 'fddb_data/'
bbox_path = os_path + 'FDDB-folds/'


# 按指定图像大小调整尺寸
def resize_image(image, size = 224):
    # 缩放比例
    scale = 0.0
    w_s = 0.0  # 宽度偏移
    h_s = 0.0  # 高度偏移
    # 获取图片尺寸
    h, w, _ = image.shape
    if h > w:
        scale = h / size
        w_s = (size - (w / scale)) / 2
    else:
        scale = w / size
        h_s = (size - (h / scale)) / 2

    top, bottom, left, right = (0,0,0,0)

    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h,w)

    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。

    # RGB颜色
    BLACK = [0,0,0]
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
    constant = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value = BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv.resize(constant, (size, size)), scale, w_s, h_s

def compute_iou(box, boxes, area, areas):
    y1 = np.maximum(box[0], boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    interSec = np.maximum(y2-y1, 0) * np.maximum(x2-x1, 0)
    union = areas[:] + area - interSec 
    iou = interSec / union
    return iou

def anchor_gen(featureMap_size, ratios, scales, rpn_stride, anchor_stride):
    ratios, scales = np.meshgrid(ratios, scales)
    ratios, scales = ratios.flatten(), scales.flatten()
    
    width = scales / np.sqrt(ratios)
    height = scales * np.sqrt(ratios)
    
    shift_x = np.arange(0, featureMap_size[0], anchor_stride) * rpn_stride
    shift_y = np.arange(0, featureMap_size[1], anchor_stride) * rpn_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    centerX, anchorX = np.meshgrid(shift_x, width)
    centerY, anchorY = np.meshgrid(shift_y, height)
    boxCenter = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    boxSize = np.stack([anchorX, anchorY], axis=2).reshape(-1, 2)
    
    boxes = np.concatenate([boxCenter - 0.5 * boxSize, boxCenter + 0.5 * boxSize], axis=1)
    return boxes

def compute_overlap(boxes1, boxes2):
    areas1 = (boxes1[:,3] - boxes1[:,1]) * (boxes1[:,2] - boxes1[:,0])
    areas2 = (boxes2[:,3] - boxes2[:,1]) * (boxes2[:,2] - boxes2[:,0])
    overlap = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes2.shape[0]):
        box = boxes2[i]
        overlap[:,i] = compute_iou(box, boxes1, areas2[i], areas1)
    return overlap

def build_rpnTarget(boxes, anchors, index):
    rpn_match = np.zeros(anchors.shape[0],dtype=np.int32)
    rpn_bboxes = np.zeros((config.train_rois_num, 4))
    
    iou = compute_overlap(anchors, boxes)
    maxArg_iou = np.argmax(iou, axis=1)
    max_iou = iou[np.arange(iou.shape[0]), maxArg_iou]
    postive_anchor_idxs = np.where(max_iou > 0.4)[0]
    negative_anchor_idxs = np.where(max_iou < 0.1)[0]
    
    rpn_match[postive_anchor_idxs]=1
    rpn_match[negative_anchor_idxs]=-1
    maxIou_anchors = np.argmax(iou, axis=0)
    rpn_match[maxIou_anchors]=1
    
    ids = np.where(rpn_match==1)[0]
    extral = len(ids) - config.train_rois_num // 2
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0
    
    ids = np.where(rpn_match==-1)[0]
    extral = len(ids) - ( config.train_rois_num - np.where(rpn_match==1)[0].shape[0])
    if extral > 0:
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0

    idxs = np.where(rpn_match==1)[0]
    ix = 0
    
    for i, a in zip(idxs, anchors[idxs]):
        gt = boxes[maxArg_iou[i]]
        
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_centy = gt[0] + 0.5 * gt_h
        gt_centx = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_centy = a[0] + 0.5 * a_h
        a_centx = a[1] + 0.5 * a_w
        
        rpn_bboxes[ix] = [(gt_centy - a_centy)/a_h, (gt_centx - a_centx)/a_w, np.log(gt_h / a_h), np.log(gt_w / a_w)]
                         
        rpn_bboxes[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bboxes

_anchors = anchor_gen(config.featureMap_size, config.ratios, config.scales, config.rpn_stride, config.anchor_stride)

def getAllImage():
    imgList = []        # 图片  [img, img]
    bboxList = []       # 标注框  [[bbox1, bbox2], [bbox1]]
    class_idList = []   # 类别  [1, 1]
    rpn_matchList = []
    rpn_bboxesList = []
    to_size = 224   # 统一的尺寸
    scale = 0.0     # 变量缩放比例
    w_s = 0.0       # 变量宽度偏移
    h_s = 0.0       # 变量高度偏移
    fileList = os.listdir(bbox_path)
    for i in range(len(fileList)):
        if (i % 2 == 0):
            with open(bbox_path + fileList[i], 'r') as f:
                line = f.readline().strip()

                read_num = 0    # 0：图片，1：还剩一个坐标点未读取
                while line:
                    if (read_num == 0):
                        img = cv.imread(os_path + line + '.jpg')
                        img, scale, w_s, h_s = resize_image(img, to_size)
                        imgList.append(img)
                        
                        line = f.readline().strip()
                        read_num = int(line)
                        bboxList.append([])
                        class_idList.append([])
                    else:
                        fold_data = line.split()
                        # 长轴，短轴，角度，椭圆中心X，椭圆中心Y，类别1
                        major_axis_radius = int(float(fold_data[0]))
                        minor_axis_radius = int(float(fold_data[1]))
                        # angle = float(fold_data[2])
                        center_x = int(float(fold_data[3]))
                        center_y = int(float(fold_data[4]))
                        class_id = int(float(fold_data[5]))
                        # 将椭圆框转换为矩形框，中心加减长短轴，就是矩形框的两个对角点：
                        i1_pt1 = (center_x - minor_axis_radius, center_y - major_axis_radius)
                        i1_pt2 = (center_x + minor_axis_radius, center_y + major_axis_radius)
                        # W = i1_pt2[0] - i1_pt1[0]
                        # H = i1_pt2[1] - i1_pt1[1]
                        bboxList[len(bboxList) - 1].append([int(i1_pt1[0] / scale + w_s), int(i1_pt1[1] / scale + h_s), int(i1_pt2[0] / scale + w_s),  int(i1_pt2[1] / scale + h_s)])
                        read_num -= 1
                        class_idList[len(bboxList) - 1].append([class_id])
                    line = f.readline().strip()
    for i in range(len(bboxList)):
        bboxList[i] = np.array(bboxList[i])
        class_idList[i] = np.array(class_idList[i])
        rpn_match, rpn_bboxes = build_rpnTarget(bboxList[i], _anchors, i)
        rpn_matchList.append(rpn_match)
        rpn_bboxesList.append(rpn_bboxes)
    return imgList, bboxList, class_idList, rpn_matchList, rpn_bboxesList


# imgList, bboxList, class_idList, rpn_matchList, rpn_bboxesList = getAllImage()
# print(class_idList[1])
# print(rpn_matchList[0])
# print(rpn_bboxesList[0])

# img_num = 1
# img = imgList[img_num]
# class_id = class_idList[img_num]
# for fi in range(len(bboxList[img_num])):
#     fold_data = bboxList[img_num][fi]  # 0：x，1：y，2：w，3：h
#     i1_pt1 = (int(fold_data[0]), int(fold_data[1]))
#     # i1_pt2 = (int(fold_data[0] + fold_data[2]), int(fold_data[1] + fold_data[3]))
#     i1_pt2 = (int(fold_data[2]), int(fold_data[3]))
#     cv.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, color=(255, 0, 255))
# cv.imshow('Image', img)
# cv.waitKey(0)


# dataset = dataSet(config.image_size, config=config)  # change
# image, bbox, class_id, rpn_match, rpn_bbox, _ = data = dataset.load_data()

# print(bbox)
# print(class_id)
# print(rpn_match.shape)
# print(rpn_bbox.shape)