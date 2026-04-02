import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR ONNX video detection")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections")
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=["bus", "car", "motor", "truck"],
        help="Class names",
    )
    parser.add_argument("--save", type=str, default="", help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show video window")
    parser.add_argument("--input-name", type=str, default="", help="Optional ONNX input name override")
    parser.add_argument("--use-rgb", action="store_true", help="Convert BGR to RGB before inference")
    return parser.parse_args()


def make_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def letterbox(image: np.ndarray, new_shape=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, r, (dw, dh)


def preprocess(frame: np.ndarray, imgsz: int, use_rgb: bool):
    img, ratio, dwdh = letterbox(frame, imgsz)
    if use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img, ratio, dwdh


def clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def scale_boxes_from_letterbox(boxes: np.ndarray, orig_shape, ratio, dwdh):
    h, w = orig_shape[:2]
    dw, dh = dwdh

    boxes = boxes.copy()
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    boxes = clip_boxes_xyxy(boxes, w, h)
    return boxes


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess_rtdetr(outputs, frame_shape, ratio, dwdh, conf_thres, labels, max_det):
    """
    Hỗ trợ:
    1) labels, boxes, scores
    2) pred_logits, pred_boxes
    """
    num_outputs = len(outputs)
    num_classes = len(labels)

    if num_outputs == 3:
        out0, out1, out2 = outputs[0], outputs[1], outputs[2]

        labels_arr = None
        boxes_arr = None
        scores_arr = None

        for arr in [out0, out1, out2]:
            if arr.ndim == 2 and arr.shape[-1] == 300 and np.issubdtype(arr.dtype, np.integer):
                labels_arr = arr
            elif arr.ndim == 3 and arr.shape[-1] == 4:
                boxes_arr = arr
            elif arr.ndim == 2 and arr.shape[-1] == 300:
                scores_arr = arr

        if labels_arr is None or boxes_arr is None or scores_arr is None:
            labels_arr, boxes_arr, scores_arr = out0, out1, out2

        labels_arr = labels_arr[0]
        boxes_arr = boxes_arr[0]
        scores_arr = scores_arr[0]

        keep = scores_arr >= conf_thres
        labels_arr = labels_arr[keep].astype(int)
        boxes_arr = boxes_arr[keep].astype(np.float32)
        scores_arr = scores_arr[keep].astype(np.float32)

        if len(boxes_arr) > 0 and boxes_arr.max() <= 1.5:
            boxes_arr = cxcywh_to_xyxy(boxes_arr)
            orig_h, orig_w = frame_shape[:2]
            pad_w = int(round(orig_w * ratio + dwdh[0] * 2))
            pad_h = int(round(orig_h * ratio + dwdh[1] * 2))
            infer_w = max(pad_w, pad_h)
            infer_h = infer_w
            boxes_arr[:, [0, 2]] *= infer_w
            boxes_arr[:, [1, 3]] *= infer_h
            boxes_arr = scale_boxes_from_letterbox(boxes_arr, frame_shape, ratio, dwdh)
        else:
            if len(boxes_arr) > 0:
                h, w = frame_shape[:2]
                if boxes_arr[:, 2].max() > w or boxes_arr[:, 3].max() > h:
                    boxes_arr = scale_boxes_from_letterbox(boxes_arr, frame_shape, ratio, dwdh)

        return labels_arr, boxes_arr, scores_arr

    elif num_outputs == 2:
        logits, boxes = outputs[0], outputs[1]
        logits = logits[0]  # [Nq, C]
        boxes = boxes[0]    # [Nq, 4]

        if logits.shape[-1] == num_classes:
            probs = sigmoid(logits)
            cls_ids = np.argmax(probs, axis=1)
            scores = probs[np.arange(len(cls_ids)), cls_ids]
        else:
            probs = softmax(logits, axis=-1)
            cls_ids = np.argmax(probs, axis=1)
            scores = probs[np.arange(len(cls_ids)), cls_ids]

        keep = scores >= conf_thres
        cls_ids = cls_ids[keep]
        scores = scores[keep]
        boxes = boxes[keep]

        order = np.argsort(-scores)[:max_det]
        cls_ids = cls_ids[order]
        scores = scores[order]
        boxes = boxes[order]

        if len(boxes) == 0:
            return (
                cls_ids.astype(int),
                np.zeros((0, 4), dtype=np.float32),
                scores.astype(np.float32),
            )

        boxes = cxcywh_to_xyxy(boxes)
        infer_h = int(round(frame_shape[0] * ratio + dwdh[1] * 2))
        infer_w = int(round(frame_shape[1] * ratio + dwdh[0] * 2))
        boxes[:, [0, 2]] *= infer_w
        boxes[:, [1, 3]] *= infer_h
        boxes = scale_boxes_from_letterbox(boxes, frame_shape, ratio, dwdh)

        return cls_ids.astype(int), boxes.astype(np.float32), scores.astype(np.float32)

    else:
        raise RuntimeError(f"Unsupported number of ONNX outputs: {num_outputs}")


def draw_detections(frame, labels_idx, boxes, scores, class_names):
    for cls_id, box, score in zip(labels_idx, boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cls_name = class_names[int(cls_id)] if 0 <= int(cls_id) < len(class_names) else str(cls_id)
        text = f"{cls_name}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, y_text), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def main():
    args = parse_args()

    model_path = Path(args.model)
    assert model_path.exists(), f"Model not found: {model_path}"

    source = int(args.source) if args.source.isdigit() else args.source

    session = make_session(str(model_path))
    input_meta = session.get_inputs()[0]
    input_name = args.input_name if args.input_name else input_meta.name
    output_names = [o.name for o in session.get_outputs()]

    print("=" * 80)
    print("ONNX Runtime session")
    print(f"Model         : {model_path}")
    print(f"Provider      : CPUExecutionProvider")
    print(f"Input name    : {input_name}")
    print(f"Input shape   : {input_meta.shape}")
    print(f"Output names  : {output_names}")
    print("=" * 80)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter: {args.save}")

    show_enabled = args.show

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp, ratio, dwdh = preprocess(frame, args.imgsz, args.use_rgb)
        outputs = session.run(output_names, {input_name: inp})

        labels_idx, boxes, scores = postprocess_rtdetr(
            outputs=outputs,
            frame_shape=frame.shape,
            ratio=ratio,
            dwdh=dwdh,
            conf_thres=args.conf,
            labels=args.labels,
            max_det=args.max_det,
        )

        vis = frame.copy()
        vis = draw_detections(vis, labels_idx, boxes, scores, args.labels)

        if writer is not None:
            writer.write(vis)

        if show_enabled:
            try:
                cv2.imshow("RT-DETR ONNX Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
            except cv2.error:
                print("[WARNING] OpenCV does not support imshow(). Continue without display.")
                show_enabled = False

    cap.release()
    if writer is not None:
        writer.release()
    if show_enabled:
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()


# python streaming.py --model "E:\Pj\Source Code Deep Learning\rtdetr_goc\model\stronger_reg\model.onnx" --source 32499-392669624_medium.mp4 --save output2.mp4 --imgsz 640 --conf 0.2