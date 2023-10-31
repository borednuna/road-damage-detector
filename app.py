from flask import Flask, render_template, Response
import cv2
import datetime
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

app = Flask(__name__)

# Define your classes
CLASSES = ['background', 'crack', 'damage', 'pothole', 'pothole_water', 'pothole_water_m']
# Define the desired width and height for the resized frames
desired_width = 320
# Define a variable to keep track of the frame count
frame_count = 0
# Define the frequency at which to run object detection (e.g., every 5 frames)
detection_frequency = 50

# Function to create the model
def create_model(num_classes):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Define a new head for the detector with the required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Video capture from your camera (adjust the index as needed)
camera = cv2.VideoCapture(0)

# Create the model and load weights here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=6).to(device)
model.load_state_dict(torch.load('./static/model/model20.pth', map_location=device))
model.eval()

# Directory where you want to save processed images
OUTPUT_DIR = 'processed_images'
detection_threshold = 0.5

record_dir = "./static/record"
recorded_buffer = []

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#filename with date and time
filename = f'{record_dir}/{timestamp}.mp4'

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            height, width, _ = frame.shape
            new_height = int(desired_width * height / width)
            # Resize the frame to the desired resolution
            frame = cv2.resize(frame, (desired_width, new_height))
            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Make the pixel range between 0 and 1
            image /= 255.0
            # Bring color channels to the front
            image = np.transpose(image, (2, 0, 1)).astype(float)
            # Convert to tensor
            image = torch.tensor(image, dtype=torch.float)
            # Add batch dimension
            image = torch.unsqueeze(image, 0)
            image = image.to(device)
            with torch.no_grad():
                outputs = model(image)

            # Load all detections to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            # Carry further only if there are detected boxes
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                # Filter out boxes according to `detection_threshold`
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # Get all the predicted class names
                pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

                # Draw the bounding boxes and write the class name on top of it
                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255), 2)  # Color (0, 0, 255) represents red
                    cv2.putText(frame, pred_classes[j],
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2, lineType=cv2.LINE_AA)
                    
                # Convert the frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                recorded_buffer.append(buffer)
                frame = buffer.tobytes()
                
                yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_post_process_frame():
    while True:
        for buffer in recorded_buffer:
            frame = buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/record_feed')
def record_feed():
    return Response(generate_post_process_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/post_process')
def post_process():
    return render_template('post_process.html')

if __name__ == '__main__':
    app.run(debug=True)
