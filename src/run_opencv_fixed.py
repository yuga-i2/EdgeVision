# src/run_opencv.py
"""
Run real-time webcam object detection with ONNX Runtime.
YOLOv8 object detection using ONNX Runtime for inference.

Usage:
    python -m src.run_opencv --model models/yolov8n.onnx --img-size 640 --conf 0.25 --device cpu --source 0
"""

import argparse
import time
import numpy as np
import onnxruntime as ort
import cv2
from src.utils import letterbox, draw_boxes, xywh_to_xyxy, COCO_LABELS, scale_coords


def preprocess(img, img_size=640):
    """
    Preprocess image using letterbox transformation.
    Returns: (model_input, original_shape, model_input_shape, ratio, pad)
    """
    img0 = img.copy()
    img_resized, r, (dw, dh) = letterbox(img0, new_shape=(img_size, img_size))
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img_resized = np.ascontiguousarray(img_resized).astype(np.float32)
    img_resized /= 255.0
    img_resized = np.expand_dims(img_resized, 0)
    return img_resized, img0.shape[:2], (img_size, img_size), r, (dw, dh)


def postprocess(output, img_shape, model_in_shape, ratio=None, pad=None, conf_thres=0.25, iou_thres=0.45):
    """
    Postprocess ONNX model output.
    
    YOLOv8 ONNX outputs are in pixel coordinates (not normalized).
    Format: (batch, num_detections, 85) where 85 = [x, y, w, h, obj_conf, class_0...class_79]
    
    Returns: (boxes_xyxy, scores, class_ids)
    """
    # If output is a list, take first element
    if isinstance(output, (list, tuple)):
        output = output[0]

    # Handle common ONNX output layouts
    arr = output
    if hasattr(arr, 'ndim') and arr.ndim == 3 and arr.shape[1] in (84, 85) and arr.shape[2] > arr.shape[1]:
        # channels-first layout (batch, channels, num) -> transpose to (batch, num, channels)
        arr = arr.transpose(0, 2, 1)

    # Extract predictions from batch
    try:
        preds = arr[0]
    except Exception:
        print('ERROR: Unexpected model output shape:', getattr(arr, 'shape', None))
        return np.array([]), np.array([]), np.array([])

    # Sanity check: we expect at least 5 values per prediction (xywh + conf)
    if preds.ndim != 2 or preds.shape[1] < 5:
        print('ERROR: Unexpected prediction tensor shape:', preds.shape)
        return np.array([]), np.array([]), np.array([])

    if preds.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Parse output based on number of channels
    C = preds.shape[1]
    
    if C == 85:
        # Standard YOLOv8: [x,y,w,h, obj_conf, class_0...class_79]
        boxes_xywh = preds[:, :4].copy()
        obj_scores = preds[:, 4]
        class_scores = preds[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        final_scores = obj_scores * class_conf
        
    elif C == 84:
        # No objectness score: [x,y,w,h, class_0...class_79]
        boxes_xywh = preds[:, :4].copy()
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        final_scores = class_conf
        
    elif C >= 6:
        # Best-effort: assume [x,y,w,h, objectness, class_0...]
        boxes_xywh = preds[:, :4].copy()
        obj_scores = preds[:, 4]
        class_scores = preds[:, 5:]
        if class_scores.size > 0:
            class_ids = np.argmax(class_scores, axis=1)
            class_conf = class_scores[np.arange(len(class_scores)), class_ids]
            final_scores = obj_scores * class_conf
        else:
            class_ids = np.zeros(len(obj_scores), dtype=int)
            final_scores = obj_scores
            
    elif C >= 5:
        # Minimal: [x,y,w,h, class_0...]
        boxes_xywh = preds[:, :4].copy()
        class_scores = preds[:, 4:]
        if class_scores.size > 0:
            class_ids = np.argmax(class_scores, axis=1)
            class_conf = class_scores[np.arange(len(class_scores)), class_ids]
            final_scores = class_conf
        else:
            class_ids = np.zeros(len(boxes_xywh), dtype=int)
            final_scores = np.ones(len(boxes_xywh))
    else:
        print('ERROR: Unexpected channel count in predictions:', C)
        return np.array([]), np.array([]), np.array([])

    # Convert xywh to xyxy
    # NOTE: YOLOv8 ONNX outputs are already in pixel coordinates (not normalized)
    xyxy = xywh_to_xyxy(boxes_xywh)

    # Scale coordinates back to original image size (remove letterbox padding)
    xyxy = scale_coords(xyxy, img_shape, model_in_shape, ratio=ratio, pad=pad)

    # Prepare boxes for NMS (expects [x,y,w,h] format)
    boxes_for_nms = []
    for x1, y1, x2, y2 in xyxy:
        w = float(x2 - x1)
        h = float(y2 - y1)
        boxes_for_nms.append([float(x1), float(y1), w, h])

    # Apply Non-Maximum Suppression
    try:
        idxs = cv2.dnn.NMSBoxes(boxes_for_nms, final_scores.tolist(), 
                               score_threshold=conf_thres, nms_threshold=iou_thres)
    except Exception as e:
        print('ERROR: NMS failed:', e)
        return np.array([]), np.array([]), np.array([])

    if len(idxs) == 0:
        return np.array([]), np.array([]), np.array([])

    # Extract NMS-filtered results
    idxs = np.array(idxs).reshape(-1)
    boxes = xyxy[idxs]
    scores_out = final_scores[idxs]
    classes = class_ids[idxs]

    return boxes, scores_out, classes


def main(model_path, img_size=640, conf=0.25, device='cpu', source='0', debug=False):
    """
    Main detection loop.
    """
    # Configure ONNX Runtime providers
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # Load ONNX model
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options, providers=providers)
    input_name = sess.get_inputs()[0].name
    
    print(f"✓ Model loaded: {model_path}")
    print(f"✓ Input: {input_name}")
    print(f"✓ Device: {device.upper()}")

    # Determine source type (camera index or file path)
    is_camera = str(source).isdigit()
    
    if is_camera:
        # ==================== CAMERA INPUT ====================
        cam_id = int(source)
        cap = None
        tried = []
        backends = []
        
        # Try different backends for camera
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append(cv2.CAP_MSMF)
        backends.append(None)  # default

        for b in backends:
            try:
                if b is None:
                    cap = cv2.VideoCapture(cam_id)
                    backend_name = 'default'
                else:
                    cap = cv2.VideoCapture(cam_id, b)
                    backend_name = {cv2.CAP_DSHOW: 'CAP_DSHOW', cv2.CAP_MSMF: 'CAP_MSMF'}.get(b, str(b))
                
                if cap is not None and cap.isOpened():
                    print(f"✓ Camera {cam_id} opened with {backend_name}")
                    break
                else:
                    tried.append(backend_name)
                    if cap is not None:
                        cap.release()
                        cap = None
            except Exception:
                tried.append(str(b))
                cap = None

        if cap is None or not cap.isOpened():
            print(f'✗ Could not open camera {cam_id} - tried backends: {tried}')
            print('  Common causes: wrong camera index, camera in use, or OS privacy settings.')
            return

        # Real-time detection loop
        fps_avg = 0
        frame_count = 0
        t0 = time.time()
        
        print("\n🎥 Starting camera stream (Press ESC to stop)...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('✗ Could not grab frame from camera.')
                    break

                # Preprocess
                img_in, orig_shape, model_in_shape, ratio, pad = preprocess(frame, img_size=img_size)

                # Inference
                inputs = {input_name: img_in}
                t1 = time.time()
                outputs = sess.run(None, inputs)
                t2 = time.time()
                inf_time = (t2 - t1) * 1000

                # Postprocess
                boxes, scores, classes = postprocess(outputs, orig_shape, model_in_shape, 
                                                    ratio=ratio, pad=pad, conf_thres=conf)

                # Draw detections
                img_draw = frame.copy()
                img_draw = draw_boxes(img_draw, boxes, scores, classes)

                # Calculate FPS
                frame_count += 1
                fps_avg = frame_count / (time.time() - t0 + 1e-8)

                # Display info
                cv2.putText(img_draw, f"FPS: {fps_avg:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(img_draw, f"Detections: {len(boxes)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Show frame
                cv2.imshow("EdgeVision - YOLOv8 Detection", img_draw)

                # Check for ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    print("\n✓ Detection stopped by user")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    else:
        # ==================== IMAGE INPUT ====================
        img = cv2.imread(source)
        if img is None:
            print(f'✗ Could not read image: {source}')
            return

        print(f"✓ Image loaded: {source}")

        # Preprocess
        img_in, orig_shape, model_in_shape, ratio, pad = preprocess(img, img_size=img_size)

        # Inference
        inputs = {input_name: img_in}
        t1 = time.time()
        outputs = sess.run(None, inputs)
        t2 = time.time()
        inf_time = (t2 - t1) * 1000

        # Postprocess
        boxes, scores, classes = postprocess(outputs, orig_shape, model_in_shape, 
                                            ratio=ratio, pad=pad, conf_thres=conf)

        # Draw detections
        img_draw = img.copy()
        img_draw = draw_boxes(img_draw, boxes, scores, classes)

        # Display results
        print(f"\n✓ Inference time: {inf_time:.1f} ms")
        print(f"✓ Detections: {len(boxes)}")

        cv2.imshow('EdgeVision - YOLOv8 Detection', img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX Real-time Detection")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--img-size", type=int, default=640, help="Model input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", choices=['cpu', 'gpu'], default='cpu', help="Device to use")
    parser.add_argument("--source", default='0', help="Camera index (0-n) or image/video path")
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    main(args.model, args.img_size, args.conf, args.device, args.source, args.debug)
