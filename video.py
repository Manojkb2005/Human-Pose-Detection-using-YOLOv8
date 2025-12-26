import cv2
import os
from ultralytics import YOLO

# ===============================
# PATHS
# ===============================
input_path = "pose_project/data1"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# ===============================
# LOAD YOLOv8 POSE MODEL
# ===============================
model = YOLO("yolov8n-pose.pt")

# ===============================
# RESIZE SETTINGS
# ===============================
MAX_WIDTH = 800   # maximum width of output frame
MAX_HEIGHT = 600  # maximum height of output frame

def resize_frame(frame, max_w=MAX_WIDTH, max_h=MAX_HEIGHT):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1:  # only shrink if bigger than max
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    return frame

# ===============================
# PROCESS VIDEO
# ===============================
for filename in os.listdir(input_path):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_path = os.path.join(input_path, filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Failed to open video:", filename)
            continue

        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(output_folder, f"output_{filename}")
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()
            annotated = resize_frame(annotated)  # resize frame

            # Initialize video writer once
            if out is None:
                h, w, _ = annotated.shape
                out = cv2.VideoWriter(out_path, fourcc, 30, (w, h))

            out.write(annotated)
            cv2.imshow("YOLOv8 Human Pose", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if out:
            out.release()

cv2.destroyAllWindows()
print("YOLOv8 Pose detection completed.")
