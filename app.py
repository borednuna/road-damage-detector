import datetime
import os
from flask import Flask, render_template, request, redirect, Response, url_for
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
socketio = SocketIO(app)

frame_urls = []

processing_done = threading.Event()  # Event to signal when video processing is done

@socketio.on('progress_update')
def handle_progress(progress):
    socketio.emit('progress_update', {'progress': progress})

CLASSES = ['background', 'lateral cracking', 'alligator cracking', 'longitudinal cracking', 'pothole', 'alligator cracking']
desired_width = 768

def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def generate_frames(video_path):
    global frame_urls  # Mark frame_urls as a global variable
    frame_urls = []  # Clear the frame_urls list for a new video
    vid = cv2.VideoCapture(video_path)
    detection_threshold = 0.4
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # clear folder /static/frames/ content
    for filename in os.listdir('./static/frames/'):
        os.remove('./static/frames/' + filename)

    for frame_count in range(total_frames):
        if frame_count % 25 == 0:
            progress = (frame_count / total_frames) * 100
            socketio.emit('progress_update', progress)
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Set the video capture to the specified frame
            success, frame = vid.read()
            print(f'Frame {frame_count} of {total_frames}')

            if not success:
                break

            height, width, _ = frame.shape
            new_height = int(desired_width * height / width)
            frame = cv2.resize(frame, (desired_width, new_height))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1)).astype(float)
            image = torch.tensor(image, dtype=torch.float)
            image = torch.unsqueeze(image, 0)
            image = image.to(device)

            detected_frames = None

            with torch.no_grad():
                outputs = model(image)

            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]


                for j, box in enumerate(draw_boxes):
                    # skip if top left corner of box is higher than 1/2 of frame height
                    if box[1] < frame.shape[0] / 2:
                        continue
                    confidence = scores[j]  # Confidence score for the detected object
                    label_text = f'{pred_classes[j]}: {confidence:.2f}'  # Combine label and confidence
                    cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255), 2)
                    cv2.putText(frame, label_text,
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2, lineType=cv2.LINE_AA)
                    detected_frames = frame.copy()

            if frame is not None and detected_frames is not None and frame_count % 20 == 0:
                frame_filename = f'frame_{frame_count}.jpg'
                frame_path = f'./static/frames/{frame_filename}'
                cv2.imwrite(frame_path, detected_frames)
                frame_url = f'/static/frames/{frame_filename}'
                frame_urls.append(frame_url)

    vid.release()
    yield 'data:100\n\n'  # Indicate 100% progress at the end

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=6).to(device)
model.load_state_dict(torch.load('./static/model/model18.pth', map_location=device))
model.eval()

def process_video(video_path):
    global frame_urls
    for frame_url in generate_frames(video_path):
        pass  # Yield the frames, but we don't need them here
    processing_done.set()  # Signal that processing is done

@app.route('/')
def post_process():
    return render_template('post_process.html')

@app.route('/processed', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)

    video_file = request.files['video']

    if video_file.filename == '':
        return redirect(request.url)

    if not video_file.filename.endswith('.mp4'):
        return 'File is not a video'

    # if int(request.content_length) > 100000000:
    #     return 'File is too large'

    video_path = './uploads/' + str(datetime.datetime.now().timestamp()) + '.mp4'
    video_file.save(video_path)

    # If the folder uploads contains more than 2 files, delete the oldest one
    files = sorted(os.listdir('./uploads'))
    if len(files) > 2:
        os.remove('./uploads/' + files[0])

    # Start video processing in a background thread
    processing_thread = threading.Thread(target=process_video, args=(video_path,))
    processing_thread.start()

    # Wait for video processing to complete
    processing_done.wait()

    return redirect(url_for('display_frames'))

@app.route('/display_frames')
def display_frames():
    global frame_urls

    return render_template('display_frames.html', frame_urls=frame_urls)

if __name__ == '__main__':
    socketio.run(app, debug=True)
