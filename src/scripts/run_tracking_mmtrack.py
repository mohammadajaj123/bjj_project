#!/usr/bin/env python3
"""
run_tracking_mmtrack.py

Pipeline:
video -> MMTracking (track_bboxes with IDs) -> MMPose top-down pose per bbox -> save + visualize

Important:
This repo's ViTPose/MMPose fork expects person_results as list[dict] with key 'bbox':
  person_results = [{"bbox": np.array([x1,y1,x2,y2,score])}, ...]
Passing a raw ndarray will crash.

Usage example:
export POSE_CONFIG="ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py"
export POSE_CHECKPOINT="checkpoints/pose/vitpose.pth"

python src/scripts/run_tracking_mmtrack.py \
  --video-path data/videos/input.mp4 \
  --mot-config ViTPose/demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py \
  --pose-config "$POSE_CONFIG" \
  --pose-checkpoint "$POSE_CHECKPOINT" \
  --device cuda:0 \
  --out-root outputs/mmtrack_deepsort \
  --save-vis \
  --bbox-thr 0.01
"""

import os
import sys
import time
import json
import logging
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np

from mmtrack.apis import init_model as init_mot_model
from mmtrack.apis import inference_mot

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
# Helpers
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def clean_checkpoint_arg(x: str):
    """MMTracking init_model should receive None (not empty string) if you want init_cfg URLs to work."""
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("", "none", "null"):
        return None
    return x


def clip_bbox_xyxy(b: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip [x1,y1,x2,y2,score] to image bounds."""
    b = b.copy()
    b[0] = float(np.clip(b[0], 0, w - 1))
    b[1] = float(np.clip(b[1], 0, h - 1))
    b[2] = float(np.clip(b[2], 0, w - 1))
    b[3] = float(np.clip(b[3], 0, h - 1))
    return b


def pad_bbox_xyxy(b: np.ndarray, pad: float, w: int, h: int) -> np.ndarray:
    """
    Pad bbox by a fraction of its size (pad=0.1 adds 10% width/height each side).
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


def fallback_draw(img, pose_results, kpt_thr=0.3):
    """Minimal visualization: bbox + track_id + keypoints."""
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


def _as_array(x):
    if x is None:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(x)


def select_person_track_bboxes(track_bboxes_raw, person_class_idx: int):
    """
    MMTracking can return:
      - ndarray (N,6) or (N,5) etc
      - list of per-class arrays: track_bboxes_raw[class_idx] -> ndarray
    We pick the specified class index when it's a list.
    """
    if isinstance(track_bboxes_raw, np.ndarray):
        return track_bboxes_raw

    if isinstance(track_bboxes_raw, list):
        # list-of-classes case
        if len(track_bboxes_raw) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        idx = int(np.clip(person_class_idx, 0, len(track_bboxes_raw) - 1))
        arr = track_bboxes_raw[idx]
        if arr is None:
            return np.zeros((0, 6), dtype=np.float32)
        return _as_array(arr)

    return _as_array(track_bboxes_raw)


def trackboxes_to_xyxy_score_and_ids(track_boxes: np.ndarray, score_thr: float):
    """
    Convert track boxes into:
      bboxes_5: (N,5) [x1,y1,x2,y2,score]
      track_ids: (N,) int (or -1 if missing)

    Supported formats:
      (N,6): [track_id, x1, y1, x2, y2, score]
      (N,5): [x1, y1, x2, y2, score]
      (N,4): [x1, y1, x2, y2] -> score=1.0
    """
    tb = _as_array(track_boxes)
    if tb.size == 0:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    if tb.ndim != 2:
        raise ValueError(f"Unexpected track_bboxes ndim={tb.ndim}, shape={tb.shape}")

    if tb.shape[1] == 6:
        track_ids = tb[:, 0].astype(np.int32)
        b = tb[:, 1:6].astype(np.float32)
    elif tb.shape[1] == 5:
        track_ids = np.full((tb.shape[0],), -1, dtype=np.int32)
        b = tb.astype(np.float32)
    elif tb.shape[1] == 4:
        track_ids = np.full((tb.shape[0],), -1, dtype=np.int32)
        ones = np.ones((tb.shape[0], 1), dtype=np.float32)
        b = np.concatenate([tb.astype(np.float32), ones], axis=1)
    else:
        raise ValueError(f"Unexpected columns={tb.shape[1]} for track_bboxes: {tb.shape}")

    keep = b[:, 4] >= float(score_thr)
    b = b[keep]
    track_ids = track_ids[keep]
    return b.astype(np.float32), track_ids.astype(np.int32)


def maybe_fallback_to_det_bboxes(mot_res, bbox_thr: float):
    """
    If tracking boxes are empty, try detector boxes.
    MMTracking sometimes exposes det_bboxes per class as list.
    We'll return an ndarray.
    """
    det = mot_res.get("det_bboxes", None)
    if det is None:
        return np.zeros((0, 5), dtype=np.float32)

    det = _as_array(det) if not isinstance(det, list) else det
    # If list-of-classes, pick class 0 by default (person); user can override by config.
    if isinstance(det, list):
        if len(det) == 0 or det[0] is None:
            return np.zeros((0, 5), dtype=np.float32)
        det0 = _as_array(det[0])
        # det0 is usually (N,5): [x1,y1,x2,y2,score]
        if det0.ndim == 2 and det0.shape[1] >= 5:
            det0 = det0[:, :5]
        return det0.astype(np.float32)

    # ndarray case
    if det.ndim == 2 and det.shape[1] >= 5:
        det = det[:, :5]
    # filter here
    if det.size:
        det = det[det[:, 4] >= float(bbox_thr)]
    return det.astype(np.float32)


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = ArgumentParser()
    p.add_argument("--video-path", type=str, default=None,
                   help="Input video path. If omitted, uses outputs/vis_input.mp4 if it exists, else errors.")

    p.add_argument("--mot-config", type=str, required=True,
                   help="MMTracking config (e.g. deepsort_faster-rcnn...).")
    p.add_argument("--mot-checkpoint", type=str, default="",
                   help="Checkpoint for the whole MOT model. Leave empty to use config init_cfg URLs.")

    p.add_argument("--pose-config", type=str, required=True, help="MMPose config file")
    p.add_argument("--pose-checkpoint", type=str, required=True, help="MMPose checkpoint file")

    p.add_argument("--device", type=str, default="cuda:0", help="Device for MOT + Pose (e.g., cuda:0 or cpu)")
    p.add_argument("--out-root", type=str, default="outputs/mmtrack_deepsort", help="Output directory root")

    # Default lowered to help get both players
    p.add_argument("--bbox-thr", type=float, default=0.01, help="BBox score threshold for pose input")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint threshold for visualization")

    p.add_argument("--person-class-idx", type=int, default=0,
                   help="When MMTracking returns per-class list, which class index to use (usually 0=person).")

    p.add_argument("--max-persons", type=int, default=10,
                   help="Keep at most K boxes per frame after sorting by score (-1 keeps all).")

    p.add_argument("--min-box-area", type=float, default=0.0,
                   help="Drop boxes with area < this value (pixels^2). Useful to remove tiny false positives.")

    p.add_argument("--bbox-pad", type=float, default=0.0,
                   help="Pad bbox by this fraction (0.1 = +10%% each side). Helps capture limbs.")

    p.add_argument("--skip-frames", type=int, default=0, help="Skip first N frames")
    p.add_argument("--max-frames", type=int, default=-1, help="Process at most N frames (-1 = all)")

    p.add_argument("--save-vis", action="store_true", help="Save per-frame JPGs + an mp4 video")
    p.add_argument("--vis-fps", type=float, default=None, help="FPS for output video (default: same as input)")

    p.add_argument("--log-every", type=int, default=25, help="Log every N saved frames")
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # Resolve video path
    if args.video_path is None:
        fallback = "outputs/vis_input.mp4"
        if os.path.exists(fallback):
            args.video_path = fallback
        else:
            print("ERROR: --video-path not provided and outputs/vis_input.mp4 not found.", file=sys.stderr)
            sys.exit(2)

    print(f"[INFO] Using video: {args.video_path}")

    # Output dirs
    out_root = args.out_root
    pred_dir = os.path.join(out_root, "predictions")
    frames_dir = os.path.join(pred_dir, "frames")
    vis_dir = os.path.join(out_root, "vis")

    ensure_dir(out_root)
    ensure_dir(pred_dir)
    ensure_dir(frames_dir)
    if args.save_vis:
        ensure_dir(vis_dir)

    # Logging
    logfile = os.path.join(out_root, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logger = logging.getLogger("run_tracking_mmtrack")

    # Load models
    mot_ckpt = clean_checkpoint_arg(args.mot_checkpoint)
    mot = init_mot_model(args.mot_config, mot_ckpt, device=args.device)

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)

    # DatasetInfo (may be None depending on config)
    dataset = pose_model.cfg.data["test"].get("type", None)
    dataset_info_cfg = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = None
    if dataset_info_cfg is not None:
        try:
            dataset_info = DatasetInfo(dataset_info_cfg)
        except Exception as e:
            logger.warning(f"Could not construct DatasetInfo: {e}")
            dataset_info = None

    # Video
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

    # Loop
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

        # 1) MOT inference
        mot_res = inference_mot(mot, frame, frame_idx)

        # 2) Get tracking boxes (prefer track_bboxes)
        track_bboxes_raw = mot_res.get("track_bboxes", None)
        track_bboxes = select_person_track_bboxes(track_bboxes_raw, args.person_class_idx)

        # 3) Convert to (N,5) + ids
        bboxes_5, track_ids = trackboxes_to_xyxy_score_and_ids(track_bboxes, score_thr=args.bbox_thr)

        # 4) Fallback to detector boxes if tracking gave nothing
        #    (in that case track_ids will be -1)
        if bboxes_5.shape[0] == 0:
            det_b = maybe_fallback_to_det_bboxes(mot_res, bbox_thr=args.bbox_thr)
            if det_b.shape[0] > 0:
                bboxes_5 = det_b.astype(np.float32)
                track_ids = np.full((bboxes_5.shape[0],), -1, dtype=np.int32)

        # 5) Optional filters: min area, max persons
        if bboxes_5.shape[0] > 0:
            # clip + pad
            b2 = []
            tids2 = []
            for i in range(bboxes_5.shape[0]):
                b = bboxes_5[i]
                b = pad_bbox_xyxy(b, args.bbox_pad, w, h)
                area = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
                if args.min_box_area > 0 and area < args.min_box_area:
                    continue
                b2.append(b)
                tids2.append(track_ids[i])
            if len(b2) == 0:
                bboxes_5 = np.zeros((0, 5), dtype=np.float32)
                track_ids = np.zeros((0,), dtype=np.int32)
            else:
                bboxes_5 = np.stack(b2, axis=0).astype(np.float32)
                track_ids = np.asarray(tids2, dtype=np.int32)

            # sort by score desc
            if bboxes_5.shape[0] > 1:
                order = np.argsort(-bboxes_5[:, 4])
                bboxes_5 = bboxes_5[order]
                track_ids = track_ids[order]

            if args.max_persons > 0 and bboxes_5.shape[0] > args.max_persons:
                bboxes_5 = bboxes_5[:args.max_persons]
                track_ids = track_ids[:args.max_persons]

        # 6) Pose inference
        pose_results = []
        if bboxes_5.shape[0] > 0:
            # CRITICAL FIX: ViTPose/mmpose fork expects list[dict] with 'bbox'
            person_results = [{"bbox": bboxes_5[i]} for i in range(bboxes_5.shape[0])]

            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                frame,
                person_results,
                bbox_thr=args.bbox_thr,
                format="xyxy",
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=False,
            )

            # Attach track_id
            for i, r in enumerate(pose_results):
                r["track_id"] = int(track_ids[i]) if i < len(track_ids) else -1

        # 7) Save per-frame predictions
        npy_path = os.path.join(frames_dir, f"frame_{saved:06d}.npy")
        np.save(npy_path, np.array(pose_results, dtype=object), allow_pickle=True)

        # 8) Visualization
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
                f"frame={frame_idx} saved={saved} boxes={bboxes_5.shape[0]} poses={len(pose_results)} "
                f"elapsed={(time.time()-t0):.1f}s"
            )

        frame_idx += 1
        saved += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()

    # Save meta
    meta = {
        "video": args.video_path,
        "frames_saved": saved,
        "skip_frames": args.skip_frames,
        "bbox_thr": args.bbox_thr,
        "kpt_thr": args.kpt_thr,
        "person_class_idx": args.person_class_idx,
        "max_persons": args.max_persons,
        "min_box_area": args.min_box_area,
        "bbox_pad": args.bbox_pad,
        "mot_config": args.mot_config,
        "mot_checkpoint": mot_ckpt,
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
