"""Neural network wrappers: YOLOv8 + ByteTrack, and Depth Anything V2 (ViT) metric depth."""
import numpy as np
import torch
from PIL import Image

ROAD_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DetectorTracker:
    def __init__(self, weights="yolov8n.pt", conf=0.35, tracker="bytetrack.yaml"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.tracker = conf, tracker

    def __call__(self, frame_bgr):
        r = self.model.track(frame_bgr, persist=True, tracker=self.tracker, conf=self.conf,
                             classes=list(ROAD_CLASSES), verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        b = r.boxes
        return [
            {"track_id": int(i), "cls_name": ROAD_CLASSES.get(int(c), str(int(c))), "conf": float(s),
             "box": xyxy.astype(float)}
            for xyxy, i, c, s in zip(b.xyxy.cpu().numpy(), b.id.cpu().numpy(),
                                     b.cls.cpu().numpy(), b.conf.cpu().numpy())
        ]


class DepthEstimator:
    """Depth Anything V2 Metric-Outdoor: ViT encoder + DPT decoder fine-tuned on
    Virtual KITTI, so its output is depth in metres for driving scenes."""

    def __init__(self, model_id, device=None, max_depth=80.0):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device or pick_device()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.max_depth = max_depth

    @torch.inference_mode()
    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        inputs = self.processor(images=Image.fromarray(frame_bgr[:, :, ::-1]), return_tensors="pt")
        px = inputs["pixel_values"].to(self.device, dtype=next(self.model.parameters()).dtype)
        pred = self.model(pixel_values=px).predicted_depth.float()
        depth = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h, w), mode="bilinear",
                                                align_corners=False)[0, 0].cpu().numpy()
        return np.clip(depth, 0.1, self.max_depth * 2.5).astype(np.float32)
