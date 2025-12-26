import cv2
import os
from ultralytics import YOLO


# Dataset PATHS

image_folder = "pose_project/dataset"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# LOAD YOLOv8 POSE MODEL

model = YOLO("yolov8n-pose.pt")  

# PROCESS IMAGES

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):

        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print("Failed to load:", filename)
            continue

        # Pose inference
        results = model(image)

        # Draw keypoints + skeleton
        annotated = results[0].plot()

        # Show output
        cv2.imshow("YOLOv8 Human Pose", annotated)
        cv2.waitKey(500)

        # Save output
        cv2.imwrite(os.path.join(output_folder, filename), annotated)

cv2.destroyAllWindows()
print("YOLOv8 Pose detection completed.")
