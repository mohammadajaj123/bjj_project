import os
import glob
import cv2
import numpy as np

IN_VIDEO = "data/videos/input.mp4"
IN_DIR   = "outputs/mmtrack_deepsort"
OUT_DIR  = "outputs/mmtrack_deepsort_vis"

os.makedirs(OUT_DIR, exist_ok=True)

# COCO skeleton edges (0-indexed)
SKELETON = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (0, 1), (1, 3),      # face
    (0, 2), (2, 4),
    (0, 5), (0, 6),
]

def load_person(npy_path):
    """
    Handles these possible saved formats:
      1) dict
      2) object ndarray of size 1 containing dict
      3) object ndarray containing list-of-dicts / dicts
    Returns either:
      - a dict (single person)
      - a list[dict] (multiple people)
    """
    x = np.load(npy_path, allow_pickle=True)

    # common: array([dict], dtype=object)
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.size == 1:
            x = x.item()
        else:
            # could be list of dicts stored as object array
            x = list(x)

    if isinstance(x, dict):
        return x

    if isinstance(x, list):
        # filter to dicts only
        xs = [p for p in x if isinstance(p, dict)]
        if not xs:
            raise ValueError(f"No dict entries in {npy_path}. Got list types: {[type(p) for p in x[:5]]}")
        return xs

    raise ValueError(f"Unexpected npy format in {npy_path}: {type(x)}")

def parse_bbox(bbox):
    """
    Try to interpret bbox robustly.

    Expected possibilities:
      - [x1, y1, x2, y2]
      - [x1, y1, x2, y2, score]  (score usually <= 1 or <= 100)
      - [x1, y1, w, h, score]    (less likely here)
    Your case: last value was ~469, so it's almost certainly NOT a score.
    We treat it as coord, and set score=None unless it looks like a real probability.
    """
    b = np.array(bbox, dtype=float).flatten()
    if b.size < 4:
        raise ValueError(f"bbox too short: {b}")

    # base coords
    x1, y1, a, c = b[0], b[1], b[2], b[3]
    score = None

    if b.size >= 5:
        last = float(b[4])

        # If last looks like a probability/confidence, keep it as score
        # (some libs use 0-1, some 0-100)
        if 0.0 <= last <= 1.0 or (0.0 <= last <= 100.0 and last not in (a, c)):
            score = last
            x2, y2 = a, c
        else:
            # last is not a valid score -> assume bbox is xyxy and ignore last
            x2, y2 = a, c
    else:
        x2, y2 = a, c

    # fix if bbox is actually xywh (w,h) not x2,y2
    # Heuristic: if x2 <= x1 or y2 <= y1, interpret as w/h
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)

    return x1, y1, x2, y2, score

def draw_one_person(img, person, kpt_thr=0.2):
    bbox = person.get("bbox", None)
    tid  = person.get("track_id", -1)
    kpts = person.get("keypoints", None)

    # Draw bbox if present
    if bbox is not None:
        x1, y1, x2, y2, score = parse_bbox(bbox)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"id={tid}"
        if score is not None:
            label += f" score={score:.2f}"
        cv2.putText(
            img, label, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # Draw keypoints/skeleton if present
    if kpts is not None:
        kpts = np.array(kpts, dtype=float)
        if kpts.ndim != 2 or kpts.shape[1] != 3:
            return img

        # keypoints
        for i in range(kpts.shape[0]):
            x, y, c = kpts[i]
            if c < kpt_thr:
                continue
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

        # skeleton
        for a, b in SKELETON:
            xa, ya, ca = kpts[a]
            xb, yb, cb = kpts[b]
            if ca < kpt_thr or cb < kpt_thr:
                continue
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

    return img

def main():
    npy_files = sorted(glob.glob(os.path.join(IN_DIR, "frame_*.npy")))
    print("Found npy frames:", len(npy_files))
    if not npy_files:
        raise SystemExit("No .npy files found.")

    cap = cv2.VideoCapture(IN_VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {IN_VIDEO}")

    for idx, npy_path in enumerate(npy_files):
        ok, frame = cap.read()
        if not ok:
            print("Video ended early at frame", idx)
            break

        data = load_person(npy_path)

        # data can be dict (single) or list[dict] (multiple)
        if isinstance(data, dict):
            frame = draw_one_person(frame, data)
        else:
            # draw all people
            for person in data:
                frame = draw_one_person(frame, person)

        out_path = os.path.join(OUT_DIR, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()
    print("Done. Wrote frames to:", OUT_DIR)

if __name__ == "__main__":
    main()
