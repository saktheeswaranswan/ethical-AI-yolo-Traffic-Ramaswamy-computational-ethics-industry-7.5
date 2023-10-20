import cv2
import numpy as np
import os
import csv
import datetime

# Load YOLOv3 Tiny model and labels
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load CSV file with class labels and binary crop values
with open('class_crops.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    class_crops = {row[0]: int(row[1]) for row in csvreader}
    class_thresholds = {row[0]: int(row[2]) for row in csvreader}

# Initialize webcam
cap = cv2.VideoCapture(0)  # You can change the index for other cameras

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = None

# Get the current date and time in Indian Standard Time (IST)
current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, 30))

# Define the output folder for cropped objects
output_folder = "cropped_objects"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a subfolder for the current date and time
timestamp = current_time.strftime("%Y-%m-%d_%I-%M-%S%p")
output_folder = os.path.join(output_folder, timestamp)
os.makedirs(output_folder)

# Initialize class-specific time counters
class_time_counters = {label: 0 for label in classes}

# Initialize dictionary to store threshold reached events
threshold_events = {label: [] for label in classes}

while True:
    ret, frame = cap.read()

    # Record the frame
    if out is None:
        video_filename = f"recorded_video_{timestamp}.avi"
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))  # Adjust resolution and frame rate as needed
    out.write(frame)

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust this threshold as needed
                label = str(classes[class_id])

                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # Check if the class should be cropped and has a time threshold
                if label in class_crops and class_crops[label] == 0 and label in class_thresholds:
                    class_time_counters[label] += 1

                    if class_time_counters[label] >= class_thresholds[label]:
                        # Save the video from the start to finish
                        video_filename = f"{label}_reached_threshold_{timestamp}.avi"
                        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))  # Adjust resolution and frame rate as needed
                        class_time_counters[label] = 0  # Reset the time counter

                        # Log the threshold event
                        event_time = current_time.strftime("%I:%M%p")
                        video_length = len(frames) / 20  # in seconds
                        threshold_events[label].append([event_time, video_length])
                        frames = []  # Reset frames

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Process detected objects and draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            class_id = class_ids[i]
            label = str(classes[class_id])
            confidence = confidences[i]

            if label in class_crops and class_crops[label] == 0:
                x, y, w, h = boxes[i]
                cropped_object = frame[y:y + h, x:x + w]
                class_folder = os.path.join(output_folder, label)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                object_filename = f'{label}_{current_time.strftime("%I-%M-%S%p")}.jpg'
                object_filepath = os.path.join(class_folder, object_filename)
                cv2.imwrite(object_filepath, cropped_object)

            color = (0, 255, 0)  # You can change the color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save the threshold event data to separate CSV files
for label, events in threshold_events.items():
    if events:
        csv_filename = f"{label}_threshold_events.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Time", "Video Length (s)"])
            csvwriter.writerows(events)

