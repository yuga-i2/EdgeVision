# src/utils.py

import cv2
import numpy as np
from typing import Tuple

# COCO Labels (80 Classes)
COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]


# --------------------
# Letterbox Preprocessing (YOLO Style)
# --------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image to maintain aspect ratio.
    """
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]  # Padding
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    # Resize
    img = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad
    top, bottom = int(dh), int(dh + 0.5)
    left, right = int(dw), int(dw + 0.5)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


# --------------------
# Preprocess for ONNX
# --------------------
def preprocess(img, img_size=640):
    img0 = img.copy()
    img_resized, r, (dw, dh) = letterbox(img0, new_shape=(img_size, img_size))
    img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB; HWC → CHW
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.expand_dims(img_rgb, 0)
    return img_rgb, img0.shape[:2], (img_size, img_size)


# --------------------
# xywh → xyxy
# --------------------
def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    if xywh.size == 0:
        return np.zeros((0, 4))
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)


# --------------------
# Scale boxes back to original image
# --------------------
def scale_coords(boxes: np.ndarray, img_shape: Tuple[int, int], model_shape: Tuple[int, int], ratio: float = None, pad: Tuple[float, float] = None):
    if boxes is None or len(boxes) == 0:
        return boxes

    orig_h, orig_w = img_shape
    model_h, model_w = model_shape

    # If ratio and pad are provided, use them directly (from letterbox)
    if ratio is not None and pad is not None:
        pad_w, pad_h = pad
        # Remove padding first (subtract padding)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - pad_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - pad_h
        # Then scale by ratio (divide by ratio to get original size)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / ratio
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / ratio
    else:
        # Fallback: calculate from model and image shapes
        gain = min(model_w / orig_w, model_h / orig_h)
        pad_w = (model_w - orig_w * gain) / 2
        pad_h = (model_h - orig_h * gain) / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain

    # Clip to image bounds
    boxes[:, 0] = boxes[:, 0].clip(0, orig_w - 1)
    boxes[:, 1] = boxes[:, 1].clip(0, orig_h - 1)
    boxes[:, 2] = boxes[:, 2].clip(0, orig_w - 1)
    boxes[:, 3] = boxes[:, 3].clip(0, orig_h - 1)

    return boxes


# --------------------
# Draw Boxes (SAFE)
# --------------------
def draw_boxes(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, labels=COCO_LABELS):
    if boxes is None or len(boxes) == 0:
        return img

    for box, score, cls in zip(boxes.astype(int), scores, classes.astype(int)):
        x1, y1, x2, y2 = box

        # Safe label lookup
        if 0 <= int(cls) < len(labels):
            label = f"{labels[int(cls)]} {score:.2f}"
        else:
            label = f"class_{int(cls)} {score:.2f}"

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, max(y1 - 22, 0)), (x1 + tw, y1), (0, 255, 0), -1)

        # Label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img
