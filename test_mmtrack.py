from mmtrack.apis import init_model, inference_mot
import cv2

cfg = "ViTPose/demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py"
checkpoint = "checkpoints/det/faster_rcnn_r50_fpn_1x_coco.pth"

print("Initializing tracker...")
model = init_model(cfg, checkpoint, device="cuda:0")

cap = cv2.VideoCapture("data/videos/input.mp4")
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = inference_mot(model, frame, frame_id=frame_id)
    print("keys:", result.keys())
    for k, v in result.items():
        print(k, type(v))
    break

    print(f"Frame {frame_id}: result type = {type(result)}")

    frame_id += 1
    if frame_id > 3:
        break

cap.release()
print("TEST FINISHED SUCCESSFULLY")
