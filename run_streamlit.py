import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from src.utils import letterbox, draw_boxes, xywh_to_xyxy, COCO_LABELS, scale_coords

st.set_page_config(page_title='EdgeVision - YOLOv8', layout='wide', page_icon='🎯')
st.title('🎯 EdgeVision - YOLOv8 Object Detection')
st.markdown('Real-time detection with ONNX Runtime')

st.sidebar.header('⚙️ Configuration')
model_dir = Path('models')
available_models = list(model_dir.glob('*.onnx'))
if not available_models:
    st.sidebar.error('❌ No ONNX models found in models/')
    st.stop()

model_path = st.sidebar.selectbox('📦 Select Model', [str(m) for m in available_models])
conf_threshold = st.sidebar.slider('🎯 Confidence Threshold', 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider('📐 IoU Threshold', 0.0, 1.0, 0.45, 0.05)
img_size = st.sidebar.selectbox('📐 Input Size', [320, 416, 512, 640], index=3)

@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])

sess = load_model(model_path)
input_name = sess.get_inputs()[0].name

def preprocess(img, size=640):
    img0 = img.copy()
    img_resized, r, (dw, dh) = letterbox(img0, (size, size))
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_resized = np.ascontiguousarray(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_resized, 0), img0.shape[:2], (size, size), r, (dw, dh)

def postprocess(output, img_shape, model_shape, ratio=None, pad=None, conf=0.25):
    if isinstance(output, (list, tuple)):
        output = output[0]
    arr = output
    if hasattr(arr, 'ndim') and arr.ndim == 3 and arr.shape[1] in (84, 85) and arr.shape[2] > arr.shape[1]:
        arr = arr.transpose(0, 2, 1)
    try:
        preds = arr[0]
    except:
        return np.array([]), np.array([]), np.array([])
    if preds.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes_xywh = preds[:, :4]
    scores = preds[:, 4] if preds.shape[1] > 4 else np.ones(len(preds))
    xyxy = xywh_to_xyxy(boxes_xywh)
    xyxy = scale_coords(xyxy, img_shape, model_shape, ratio=ratio, pad=pad)
    return xyxy, scores, np.zeros(len(xyxy), dtype=int)

st.sidebar.success(f'✓ Model loaded')
st.sidebar.info(f'Model: {Path(model_path).name}\nInput: {img_size}x{img_size}')

tab1, tab2 = st.tabs(['📸 Image', '🎥 Camera'])

with tab1:
    st.subheader('Upload Image for Detection')
    uploaded = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded:
        image = Image.open(uploaded)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with st.spinner('Running inference...'):
            img_in, orig_shape, model_shape, r, p = preprocess(img_cv, img_size)
            outputs = sess.run(None, {input_name: img_in})
            boxes, scores, _ = postprocess(outputs, orig_shape, model_shape, r, p, conf_threshold)
        
        img_draw = draw_boxes(img_cv.copy(), boxes, scores, np.zeros(len(boxes)))
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col2:
            st.metric('🎯 Detections', len(boxes))
            if len(boxes) > 0:
                st.write('**Objects detected:**')
                for i, (b, s) in enumerate(zip(boxes, scores)):
                    st.write(f'• Detection {i+1}: {s:.1%}')

with tab2:
    st.subheader('Real-time Camera Detection')

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    start_cam = st.checkbox("🎥 Enable Camera Stream", value=st.session_state.run_cam, key="cam_checkbox")

    if start_cam:
        st.session_state.run_cam = True
        frame_placeholder = st.empty()
        stats = st.empty()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Could not open camera")
        else:
            frame_count = 0

            while st.session_state.run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read camera")
                    break

                frame_small = cv2.resize(frame, (640, 480))

                try:
                    # Run detection
                    img_in, orig_shape, model_shape, r, p = preprocess(frame_small, img_size)
                    outputs = sess.run(None, {input_name: img_in})
                    boxes, scores, _ = postprocess(outputs, orig_shape, model_shape, r, p, conf_threshold)

                    # Draw on BGR
                    draw_bgr = frame_small.copy()
                    det_count = 0
                    for box, score in zip(boxes.astype(int), scores):
                        x1, y1, x2, y2 = [int(max(0, min(c, 639 if i%2==0 else 479))) for i, c in enumerate(box)]
                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(draw_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(draw_bgr, f"{score:.2f}", (x1, max(y1-5, 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            det_count += 1

                    # Convert to RGB
                    frame_rgb_display = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)

                except Exception as e:
                    st.error(f"Detection failed: {str(e)[:100]}")
                    frame_rgb_display = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    det_count = 0

                # Display frame
                frame_placeholder.image(frame_rgb_display, channels="RGB", use_column_width=True)

                frame_count += 1
                with stats.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Frames", frame_count)
                    col2.metric("Detections", det_count)
                    col3.metric("Confidence", f"{conf_threshold:.2f}")

            cap.release()
            cv2.destroyAllWindows()
            st.info(f"✓ Stopped after {frame_count} frames")

    else:
        st.session_state.run_cam = False
