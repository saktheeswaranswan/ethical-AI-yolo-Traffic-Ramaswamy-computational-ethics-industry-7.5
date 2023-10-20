import csv
import random

# Read class labels from coco.names
with open('coco.names', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Generate random thresholds between 1 and 5000 milliseconds (1 to 5 seconds)
class_data = [{'label': label, 'crop': random.randint(0, 1), 'threshold': random.randint(1, 5)} for label in class_labels]

# Output CSV file name
output_file = 'class_crops.csv'

# Write the class data to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['label', 'crop', 'threshold']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for item in class_data:
        writer.writerow(item)

print(f'CSV file "{output_file}" created successfully with randomly assigned thresholds for all classes from coco.names.')

