#!/usr/bin/env python3
"""
run_tracking_detr_iou.py

Option 2:
- Use DETR detector (MMDetection) per frame -> detections
- Assign stable track IDs with a simple IoU tracker
- Run MMPose top-down pose for each tracked bbox
- Save per-frame predictions and optional visualization

Requires:
- mmdet installed (same environment as mmtrack/mmpose often works)
- A DETR config (.py) that matches your trained checkpoint (.pth)

Example:
python src/scripts/run_tracking_detr_iou.py \
  --video-path data/videos/input.mp4 \
  --det-config path/to/your_detr_config.py \
  --det-checkpoint checkpoints/detection/deformable_detr_twostage_refine.pth \
  --det-class-idx 0 \
  --det-thr 0.2 \
  --max-dets 2 \
  --pose-config "$POSE_CONFIG" \
  --pose-checkpoint "$POSE_CHECKPOINT" \
  --device cuda:0 \
  --out-root outputs/detr_ioutrack \
  --save-vis
"""

import os
import sys
import time
import json
import logging
import warnings
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List, Tuple, Optional

import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector

from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmpose.datasets import DatasetInfo

# Optional: repo visualizer
try:
    from bjjtrack.utils import vis_pose_tracking_result  # type: ignore
    HAS_REPO_VIS = True
except Exception:
    HAS_REPO_VIS = False

warnings.filterwarnings("ignore")


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clip_bbox_xyxy(b: np.ndarray, w: int, h: int) -> np.ndarray:
    b = b.copy()
    b[0] = float(np.clip(b[0], 0, w - 1))
    b[1] = float(np.clip(b[1], 0, h - 1))
    b[2] = float(np.clip(b[2], 0, w - 1))
    b[3] = float(np.clip(b[3], 0, h - 1))
    return b


def pad_bbox_xyxy(b: np.ndarray, pad: float, w: int, h: int) -> np.ndarray:
    """
    Pad bbox by fraction of its size (pad=0.1 adds 10% each side).
    b: [x1,y1,x2,y2,score]
    """
    if pad <= 0:
        return clip_bbox_xyxy(b, w, h)
    b = b.copy()
    x1, y1, x2, y2 = b[:4].astype(float)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    dx = bw * pad
    dy = bh * pad
    b[0] = x1 - dx
    b[1] = y1 - dy
    b[2] = x2 + dx
    b[3] = y2 + dy
    return clip_bbox_xyxy(b, w, h)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def fallback_draw(img, pose_results, kpt_thr=0.3):
    out = img.copy()
    for r in pose_results:
        bbox = np.asarray(r.get("bbox", None))
        if bbox is not None and bbox.size >= 4:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            score = float(bbox[4]) if bbox.size >= 5 else 0.0
            tid = int(r.get("track_id", -1))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"id={tid} s={score:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        kpts = r.get("keypoints", None)
        if kpts is None:
            continue
        kpts = np.asarray(kpts)
        if kpts.ndim != 2 or kpts.shape[1] < 2:
            continue
        for i in range(min(kpts.shape[0], 50)):
            x, y = float(kpts[i, 0]), float(kpts[i, 1])
            c = float(kpts[i, 2]) if kpts.shape[1] >= 3 else 1.0
            if c < kpt_thr:
                continue
            cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
    return out


# ----------------------------
# Simple IoU tracker (for ~2 players)
# ----------------------------
@dataclass
class Track:
    track_id: int
    bbox: np.ndarray          # [x1,y1,x2,y2,score]
    last_score: float
    lost: int = 0


class IoUTracker:
    def __init__(self, iou_thr: float = 0.3, max_lost: int = 30, max_tracks: int = 2):
        self.iou_thr = float(iou_thr)
        self.max_lost = int(max_lost)
        self.max_tracks = int(max_tracks)
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        detections: (N,5) [x1,y1,x2,y2,score]
        returns track_bboxes: (M,6) [track_id,x1,y1,x2,y2,score]
        """
        dets = np.asarray(detections, dtype=np.float32)
        if dets.size == 0:
            # no detections: increment lost counters
            for t in self.tracks:
                t.lost += 1
            self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
            return np.zeros((0, 6), dtype=np.float32)

        # Greedy matching based on IoU
        used_det = set()
        for t in self.tracks:
            best_j = -1
            best_iou = 0.0
            for j in range(dets.shape[0]):
                if j in used_det:
                    continue
                cur_iou = iou_xyxy(t.bbox, dets[j])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_thr:
                t.bbox = dets[best_j]
                t.last_score = float(dets[best_j, 4])
                t.lost = 0
                used_det.add(best_j)
            else:
                t.lost += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        # Add new tracks for unmatched detections (up to max_tracks)
        for j in range(dets.shape[0]):
            if j in used_det:
                continue
            if len(self.tracks) >= self.max_tracks:
                break
            new = Track(track_id=self._next_id, bbox=dets[j], last_score=float(dets[j, 4]), lost=0)
            self._next_id += 1
            self.tracks.append(new)

        # Output tracks
        out = []
        for t in self.tracks:
            b = t.bbox
            out.append([t.track_id, b[0], b[1], b[2], b[3], b[4]])
        return np.asarray(out, dtype=np.float32)


# ----------------------------
# Detection extraction
# ----------------------------
def extract_class_dets(det_result, class_idx: int) -> np.ndarray:
    """
    MMDetection inference_detector output varies by version/model:
      - list[np.ndarray] per class: det_result[class_idx] -> (N,5)
      - tuple(list, segm) etc: take det_result[0]
    Returns (N,5): [x1,y1,x2,y2,score]
    """
    res = det_result
    if isinstance(res, tuple):
        res = res[0]
    if isinstance(res, list):
        if len(res) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        idx = int(np.clip(class_idx, 0, len(res) - 1))
        arr = res[idx]
        if arr is None:
            return np.zeros((0, 5), dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 5:
            return arr[:, :5].astype(np.float32)
        return np.zeros((0, 5), dtype=np.float32)

    # ndarray case (rare)
    arr = np.asarray(res, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 5:
        return arr[:, :5].astype(np.float32)
    return np.zeros((0, 5), dtype=np.float32)


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = ArgumentParser()

    p.add_argument("--video-path", type=str, required=True)

    p.add_argument("--det-config", type=str, required=True, help="MMDetection DETR config (.py)")
    p.add_argument("--det-checkpoint", type=str, required=True, help="DETR checkpoint (.pth)")

    p.add_argument("--det-class-idx", type=int, default=0, help="Class index to use (0 is person for COCO)")
    p.add_argument("--det-thr", type=float, default=0.2, help="Detection score threshold")
    p.add_argument("--max-dets", type=int, default=2, help="Keep at most K detections per frame (2 players)")

    p.add_argument("--tracker-iou", type=float, default=0.3, help="IoU threshold for matching")
    p.add_argument("--tracker-max-lost", type=int, default=30, help="Frames to keep track without match")

    p.add_argument("--pose-config", type=str, required=True)
    p.add_argument("--pose-checkpoint", type=str, required=True)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-root", type=str, default="outputs/detr_ioutrack")

    p.add_argument("--bbox-pad", type=float, default=0.05, help="Pad bbox before pose (fraction)")
    p.add_argument("--kpt-thr", type=float, default=0.3)

    p.add_argument("--skip-frames", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=-1)

    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--vis-fps", type=float, default=None)

    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    out_root = args.out_root
    pred_dir = os.path.join(out_root, "predictions")
    frames_dir = os.path.join(pred_dir, "frames")
    vis_dir = os.path.join(out_root, "vis")

    ensure_dir(out_root)
    ensure_dir(pred_dir)
    ensure_dir(frames_dir)
    if args.save_vis:
        ensure_dir(vis_dir)

    logfile = os.path.join(out_root, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logger = logging.getLogger("run_tracking_detr_iou")

    logger.info(f"Video: {args.video_path}")
    logger.info(f"DETR:  cfg={args.det_config} ckpt={args.det_checkpoint}")

    # Load models
    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)

    dataset = pose_model.cfg.data["test"].get("type", None)
    dataset_info_cfg = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = None
    if dataset_info_cfg is not None:
        try:
            dataset_info = DatasetInfo(dataset_info_cfg)
        except Exception as e:
            logger.warning(f"DatasetInfo init failed: {e}")
            dataset_info = None

    tracker = IoUTracker(iou_thr=args.tracker_iou, max_lost=args.tracker_max_lost, max_tracks=2)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Could not open video: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps <= 0:
        in_fps = 30.0
    out_fps = float(args.vis_fps) if args.vis_fps is not None else float(in_fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)

    video_writer = None
    out_vid = None
    if args.save_vis:
        base = os.path.basename(args.video_path)
        out_vid = os.path.join(out_root, f"vis_{base}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_vid, fourcc, out_fps, size)

    frame_idx = 0
    saved = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < args.skip_frames:
            frame_idx += 1
            continue

        if args.max_frames > 0 and saved >= args.max_frames:
            break

        # 1) DETR detection
        det_result = inference_detector(det_model, frame)
        dets = extract_class_dets(det_result, args.det_class_idx)  # (N,5)

        # 2) Score threshold + top-K (2 players)
        if dets.size:
            dets = dets[dets[:, 4] >= float(args.det_thr)]
            if dets.shape[0] > 1:
                order = np.argsort(-dets[:, 4])
                dets = dets[order]
            if args.max_dets > 0 and dets.shape[0] > args.max_dets:
                dets = dets[:args.max_dets]

            # clip/pad later (after tracking)

        # 3) Tracking (assign IDs)
        track_bboxes = tracker.update(dets)  # (M,6) [id,x1,y1,x2,y2,score]

        # 4) Prepare pose bboxes: (M,5) and track_ids
        if track_bboxes.size == 0:
            bboxes_5 = np.zeros((0, 5), dtype=np.float32)
            track_ids = np.zeros((0,), dtype=np.int32)
        else:
            track_ids = track_bboxes[:, 0].astype(np.int32)
            bboxes_5 = track_bboxes[:, 1:6].astype(np.float32)

        # pad/clip before pose
        b2 = []
        ids2 = []
        for i in range(bboxes_5.shape[0]):
            b = pad_bbox_xyxy(bboxes_5[i], args.bbox_pad, w, h)
            b2.append(b)
            ids2.append(track_ids[i])
        if len(b2) > 0:
            bboxes_5 = np.stack(b2, axis=0).astype(np.float32)
            track_ids = np.asarray(ids2, dtype=np.int32)

        # 5) Pose inference
        pose_results = []
        if bboxes_5.shape[0] > 0:
            # IMPORTANT: this ViTPose/MMPose fork expects list[dict] with 'bbox'
            person_results = [{"bbox": bboxes_5[i]} for i in range(bboxes_5.shape[0])]

            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                frame,
                person_results,
                bbox_thr=0.0,  # we already thresholded on det_thr; keep pose stage permissive
                format="xyxy",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=False,
            )

            for i, r in enumerate(pose_results):
                r["track_id"] = int(track_ids[i]) if i < len(track_ids) else -1

        # 6) Save predictions
        npy_path = os.path.join(frames_dir, f"frame_{saved:06d}.npy")
        np.save(npy_path, np.array(pose_results, dtype=object), allow_pickle=True)

        # 7) Visualization
        if args.save_vis:
            if HAS_REPO_VIS:
                vis = vis_pose_tracking_result(
                    pose_model,
                    frame,
                    pose_results,
                    radius=2,
                    thickness=1,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=args.kpt_thr,
                    show=False,
                    sort=True,
                    vis_bg=True,
                )
            else:
                vis = fallback_draw(frame, pose_results, kpt_thr=args.kpt_thr)

            jpg_path = os.path.join(vis_dir, f"{saved:06d}.jpg")
            cv2.imwrite(jpg_path, vis)
            if video_writer is not None:
                video_writer.write(vis)

        if saved % max(1, args.log_every) == 0:
            logger.info(
                f"frame={frame_idx} saved={saved} dets={dets.shape[0] if dets is not None else 0} "
                f"tracks={bboxes_5.shape[0]} poses={len(pose_results)} "
                f"elapsed={(time.time()-t0):.1f}s"
            )

        frame_idx += 1
        saved += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()

    meta = {
        "video": args.video_path,
        "frames_saved": saved,
        "det_config": args.det_config,
        "det_checkpoint": args.det_checkpoint,
        "det_class_idx": args.det_class_idx,
        "det_thr": args.det_thr,
        "max_dets": args.max_dets,
        "tracker_iou": args.tracker_iou,
        "tracker_max_lost": args.tracker_max_lost,
        "pose_config": args.pose_config,
        "pose_checkpoint": args.pose_checkpoint,
        "device": args.device,
        "out_root": args.out_root,
        "vis_video": out_vid if args.save_vis else None,
    }
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Saved {saved} frames to: {frames_dir}")
    if args.save_vis:
        print(f"Visualization saved to: {vis_dir}")
        if out_vid is not None:
            print(f"Video saved to: {out_vid}")


if __name__ == "__main__":
    main()
