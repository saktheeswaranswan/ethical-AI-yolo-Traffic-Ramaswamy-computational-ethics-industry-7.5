import cv2
import numpy as np
import os
import csv

# Load YOLOv3 Tiny model and labels
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load CSV file with class labels and binary crop values
with open('class_crops.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    class_crops = {row[0]: int(row[1]) for row in csvreader}

# Initialize webcam
cap = cv2.VideoCapture(0)  # You can change the index for other cameras

while True:
    ret, frame = cap.read()

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
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Process detected objects
    for i in range(len(boxes)):
        if i in indexes:
            class_id = class_ids[i]
            label = str(classes[class_id])
            confidence = confidences[i]
            if label in class_crops and class_crops[label] == 1:
                # Crop and save the detected object
                x, y, w, h = boxes[i]
                cropped_object = frame[y:y + h, x:x + w]
                cv2.imwrite(f'{label}.jpg', cropped_object)

            # Draw bounding boxes on the frame
            color = (0, 255, 0)  # You can change the color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()

