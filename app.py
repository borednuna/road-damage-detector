from flask import Flask, render_template, Response
import cv2
import datetime
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

app = Flask(__name__)

CLASSES = ['pothole', 'longitudinal cracking', 'lateral cracking', 'alligator cracking']
desired_width = 320
frame_count = 0
detection_frequency = 50

def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load('./static/model/model20.pth', map_location=device))
model.eval()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

@app.route('/post_process')
def post_process():
    return render_template('post_process.html')

if __name__ == '__main__':
    app.run(debug=True)
