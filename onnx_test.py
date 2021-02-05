# %% Imports and preprocess function
import math
import random
import time

import numpy as np
import onnx
import onnxruntime
from cv2 import cv2
from matplotlib import pyplot as plt

from max_length_list import Base


def rmse(x, y):
    size = x.size
    mse = np.sum((x - y) ** 2) / size
    rmse = math.sqrt(mse)
    return rmse, mse, size


def plot(image, det):
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img, label, (c1[0],
                             c1[1] - 2),
                0, tl / 3, [225, 255, 255],
                thickness=tf, lineType=cv2.LINE_AA)

    for *xyxy, conf, cls in reversed(det):
        plot_one_box(xyxy, image, line_thickness=3)

    _, ax = plt.subplots(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    plt.show()


def plot_time(times):
    _, ax = plt.subplots(1)
    ax.plot(times)
    plt.show


def preprocess(path):
    def letterbox(
            img, new_shape=(640, 640),
            color=(114, 114, 114),
            auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better test mAP)
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # wh padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            # width, height ratios
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    im0 = cv2.imread(path)
    img = letterbox(im0)[0]
    img = np.stack(img, 0)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img / 255.0
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)
    return img.astype(np.float32), im0


# TODO: Change the post-process
def postprocess(output):
    anchor_grid = [
        np.array([[[[[10, 13]]],
                   [[[16, 30]]],
                   [[[33, 23]]]]]),
        np.array([[[[[30, 61]]],
                   [[[62, 45]]],
                   [[[59, 119]]]]]),
        np.array([[[[[116, 90]]],
                   [[[156, 198]]],
                   [[[373, 326]]]]]),
    ]

    def make_grid(nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack(
            (xv, yv),
            2).reshape(
            (1, 1, ny, nx, 2)).astype(
            np.float32)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    z = []
    for i, x in enumerate(output):
        bs, c, ny, nx, no = x.shape
        grid = make_grid(nx, ny)

        y = sigmoid(x)
        y[..., 0:2] = y[..., 0:2] * 2.0 - 0.5 + grid
        y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * anchor_grid[i]

        z.append(y.reshape(bs, -1, no))

    return (np.concatenate(z, 1), output)


def non_max_suppression(prediction, conf_thres=0.0, iou_thres=1.0,
                        classes=None, agnostic=False, labels=()):
    def box_iou(box1, box2):
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (
            np.min(box1[:, None, 2:],
                   box2[:, 2:]) - torch.np(box1[:, None, : 2],
                                           box2[:, : 2])).clamp(0).prod(2)
        # iou = inter / (area1 + area2 - inter)
        return inter / (area1[:, None] + area2 - inter)

    def nms(boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (y2-y1+1) * (x2-x1+1)
        keep = []
        index = scores.argsort()[::-1]
        while index.size > 0:
            # every time the first is the biggst, and add it directly
            i = index[0]
            keep.append(i)

            # calculate the points of overlap
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22-x11+1)    # the weights of overlap
            h = np.maximum(0, y22-y11+1)    # the height of overlap

            overlaps = w*h
            ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)

            idx = np.where(ious <= iou_threshold)[0]
            index = index[idx+1]   # because index start from 1

        return keep

    def xywh2xyxy(x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    time_limit = 10.0  # seconds to quit after
    merge = False  # use merge-NMS
    redundant = True  # require redundant detections

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].astype(np.long) + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = np.stack((x[:, 5:] > conf_thres).nonzero())
            x = np.concatenate(
                (box[i], x[i, j + 5, None], j[:, None].astype(np.float)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = np.concatenate((box, conf, j.astype(np.float)), 1)[
                conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]

        i = nms(boxes, scores, iou_thres)  # NMS
        if len(i) > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.matmul(weights, x[:, :4]).astype(
                np.float) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    def clip_coords(boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0],
            img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, \
            (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


# %%
preprocess_time = []
postprocess_time = []
model_time = []
weight = './weights/inference.yolov5s.v12.onnx'
path = '/cv_data/sifan/images/original/follow/distance/dw800/0/1020.jpg'
model = onnxruntime.InferenceSession(weight)

end = time.time()
for _ in range(256):
    img, im0 = preprocess(path)

    preprocess_time.append(time.time() - end)
    end = time.time()

    output = model.run(None, {'images': img})

    model_time.append(time.time() - end)
    end = time.time()

    if len(output) == 4:    # onnx 10, 11, 12
        pred = output[0]
    else:                   # onnx 10
        pred = postprocess(output)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            human = det[det[:, -1] == 0]

    postprocess_time.append(time.time() - end)
    end = time.time()


# %%
plot(im0, human)
plot_time(preprocess_time)
plot_time(model_time)
plot_time(postprocess_time)

