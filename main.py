import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Model load
model = YOLO('yolov8n.pt')
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
memory = {}

cap = cv2.VideoCapture('carVideo.mp4')
# video save korar jonno sothik size
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    car_dets = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in [2,3,5,7]: # car, motorbike, bus, truck
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                car_dets.append([x1, y1, x2, y2, conf])

    if len(car_dets) > 0:
        tracks = tracker.update(np.array(car_dets))
    else:
        tracks = tracker.update(np.empty((0,5)))

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cx = (x1 + x2) // 2
        if track_id not in memory:
            memory[track_id] = cx
            continue
        diff = cx - memory[track_id]
        if abs(diff) > 8:
            if diff > 0:
                direction = "Right ->"
                color = (0, 255, 0)
            else:
                direction = "<- Left WRONG"
                color = (0, 0, 255)
        else:
            direction = "Stop"
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {direction}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        memory[track_id] = cx

    out.write(frame)
    cv2.imshow("Car Direction - Afjal", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()