from ultralyticsplus import YOLO
import torch  # For checking GPU availability
import cv2
import time

# Check if CUDA (GPU) is available, else use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLOv8 model (you can change this to yolov8s.pt or other models)
model = YOLO('yolov8l.pt').to(device)  # Load the model to the appropriate device

model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# Open the video file (or use 0 for webcam)
video_path = "video.mov"  # Change to your video file path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define codec and create VideoWriter object to save the output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Confidence threshold (in percentage)
confidence_threshold = 0.1  # 10% confidence

# Start time for processing
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Access all the detected boxes, confidence scores, and class labels
    boxes = results[0].boxes  # Boxes object containing detections

    for box in boxes:
        conf = box.conf.item()  # Get the confidence score (as a float)

        # Only plot the box if the confidence is above the threshold
        if conf > confidence_threshold:
            # Get the box coordinates and class information
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = results[0].names[int(box.cls[0])]  # Class name

            # Define the label text
            label_text = f'{label} {conf:.2f}'

            # Get the width and height of the label text
            (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw a blue bounding box (BGR color: (255, 0, 0) for blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bounding box

            # Draw a filled rectangle (background for the label)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (255, 0, 0), -1)  # Filled blue background

            # Put the label text in white color (BGR color: (255, 255, 255) for white)
            cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Label text


    # Show the frame with filtered detections (optional, if you want to see it live)
    # cv2.imshow('YOLOv8 Detection', frame)

    # Write the frame with detections to the output video
    out.write(frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# End time for processing
end_time = time.time()

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
