
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = r"model\model.onnx"
IMG_DIR = r"E:\Pj\Source Code Deep Learning\rtdetr_goc\rtdetrv2_pytorch\dataset\test"
SAVE_DIR = r"metrics\test_onnx"

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

session = ort.InferenceSession(MODEL_PATH)
print("Output names:", [o.name for o in session.get_outputs()])

CLASS_NAMES = ["bus", "car", "motor", "truck"]
SCORE_THRESH = 0.3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(img):
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = img_resized.transpose(2, 0, 1)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized, w, h

def cxcywh_to_xyxy(boxes, orig_w, orig_h):
    boxes = boxes.copy()
    boxes[:, 0] *= orig_w
    boxes[:, 1] *= orig_h
    boxes[:, 2] *= orig_w
    boxes[:, 3] *= orig_h

    x_c = boxes[:, 0]
    y_c = boxes[:, 1]
    bw = boxes[:, 2]
    bh = boxes[:, 3]

    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2

    return np.stack([x1, y1, x2, y2], axis=1)

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
    image_paths.extend(Path(IMG_DIR).glob(ext))

print("Found images:", len(image_paths))

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print("Skip unreadable:", img_path)
        continue

    input_tensor, orig_w, orig_h = preprocess(img)
    outputs = session.run(None, {"images": input_tensor})

    pred_logits = outputs[0]
    pred_boxes = outputs[1]

    logits = pred_logits[0]
    boxes = pred_boxes[0]

    probs = sigmoid(logits)
    scores = probs.max(axis=1)
    labels = probs.argmax(axis=1)

    keep = scores > SCORE_THRESH
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    boxes_xyxy = cxcywh_to_xyxy(boxes, orig_w, orig_h)

    vis = img.copy()
    for box, score, label in zip(boxes_xyxy, scores, labels):
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        cls_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{cls_name} {score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    save_path = Path(SAVE_DIR) / img_path.name
    cv2.imwrite(str(save_path), vis)
    print("Saved:", save_path)

print("✅ Done ONNX detect")