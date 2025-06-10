import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

def run_vehicle_detection(input_video_path):
    model = YOLO('yolov8n.pt')
    class_list = model.names
    cap = cv2.VideoCapture(input_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('static/output_video.mp4', fourcc, fps, (width, height))

    line_y_top = 300
    line_y_bottom = 430
    class_counts_up = defaultdict(int)
    class_counts_down = defaultdict(int)
    crossed_ids_top = set()
    crossed_ids_bottom = set()
    previous_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.line(frame, (690, line_y_top), (1130, line_y_top), (0, 0, 255), 2)
        cv2.line(frame, (690, line_y_bottom), (1130, line_y_bottom), (0, 0, 255), 2)

        results = model.track(frame, persist=True, classes=None)

        if results and results[0].boxes is not None and results[0].boxes.data.shape[0] > 0:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id
            classes = results[0].boxes.cls
            confs = results[0].boxes.conf

            if ids is not None:
                track_ids = ids.int().cpu().tolist()
                class_indices = classes.int().cpu().tolist()
                confidences = confs.cpu().tolist()

                for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    class_name = class_list[class_idx]

                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID:{track_id} {class_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    prev_cy = previous_positions.get(track_id, None)

                    if prev_cy is not None and prev_cy < line_y_bottom <= cy and track_id not in crossed_ids_bottom:
                        crossed_ids_bottom.add(track_id)
                        class_counts_down[class_name] += 1

                    if prev_cy is not None and prev_cy > line_y_top >= cy and track_id not in crossed_ids_top:
                        crossed_ids_top.add(track_id)
                        class_counts_up[class_name] += 1

                    previous_positions[track_id] = cy

                y_offset = 30
                for cls in set(list(class_counts_up) + list(class_counts_down)):
                    up = class_counts_up[cls]
                    down = class_counts_down[cls]
                    cv2.putText(frame, f"{cls}: Up={up} | Down={down}", (50, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    data = []
    for cls in set(list(class_counts_up.keys()) + list(class_counts_down.keys())):
        data.append({
            'Class': cls,
            'Count_Up': class_counts_up.get(cls, 0),
            'Count_Down': class_counts_down.get(cls, 0)
        })

    df = pd.DataFrame(data)
    df.to_csv('static/counts.csv', index=False)
