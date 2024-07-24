import os
import shutil
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import base64
import datetime
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
import subprocess
import threading
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG)

source_folder = ""
app = Flask(__name__, static_folder=f'{source_folder}static', static_url_path='')
app.config['UPLOAD_FOLDER'] = f'{source_folder}uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Excel file setup
excel_file_path = f'{source_folder}detections.xlsx'
if not os.path.exists(excel_file_path):
    df = pd.DataFrame(columns=["plate_text", "timestamp"])
    df.to_excel(excel_file_path, index=False)

socketio = SocketIO(app, cors_allowed_origins="*")

# Load models


# Class name mapping for character detection
class_name_mapping = {
    '00': '0', '01': '1', '02': '2', '03': '3', '04': '4', '05': '5',
    '06': '6', '07': '7', '08': '8', '09': '9',
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
    'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
    'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
    'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
}

stop_detection_flag = False
rtsp_link = ""

def detect_characters(image):
    char_results = char_model(image, conf=0.3)
    plate_text = []
    upper_row = []
    lower_row = []

    for char_result in char_results:
        char_boxes = char_result.boxes.xyxy.cpu().numpy()
        char_scores = char_result.boxes.conf.cpu().numpy()
        char_classes = char_result.boxes.cls.cpu().numpy()

        keep = nms(torch.tensor(char_boxes), torch.tensor(char_scores), iou_threshold=0.5)
        char_boxes = char_boxes[keep]
        char_scores = char_scores[keep]
        char_classes = char_classes[keep]

        for char_box, char_score, char_cls in zip(char_boxes, char_scores, char_classes):
            if char_score > 0.3:
                class_name = char_model.names[int(char_cls)]
                if class_name in class_name_mapping:
                    character = class_name_mapping[class_name]
                    cx1, cy1, cx2, cy2 = char_box[:4]
                    if cy1 < image.shape[0] // 2:
                        upper_row.append((cx1, character))
                    else:
                        lower_row.append((cx1, character))

    upper_row.sort(key=lambda x: x[0])
    lower_row.sort(key=lambda x: x[0])

    upper_row_str = ''.join([char for _, char in upper_row])
    lower_row_str = ''.join([char for _, char in lower_row])
    plate_text_str = upper_row_str + lower_row_str

    if plate_text_str and len(plate_text_str) > 1 and plate_text_str[1] == 'L' and plate_text_str[0] in ['0', 'O', 'D', 'Q']:
        plate_text_str = 'D' + plate_text_str[1:]

    return plate_text_str[:10]

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/start_stream', methods=['POST'])
def start_stream():
    global stop_detection_flag
    stop_detection_flag = False
    logging.debug("RTSP stream started")

    def generate():
        cap = cv2.VideoCapture(rtsp_link)
        if not cap.isOpened():
            logging.error("Failed to open RTSP stream")
            return

        frame_count = 0
        frames_processed = 0
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if frame_rate <= 0:
            frame_rate = 30  # Default frame rate
        interval = int(frame_rate / 3)  # Process every frame for 10 FPS
        detected_strings = set()

        detection_entries = []

        try:
            df = pd.read_excel(excel_file_path, engine='openpyxl')
        except Exception as e:
            logging.error(f"Error reading Excel file: {e}")
            return jsonify({"error": "Error reading Excel file"}), 500

        while cap.isOpened():
            if stop_detection_flag:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                logging.warning("Received an empty frame.")
                continue
            if frames_processed % interval == 0:
                plate_results = plate_model(frame, conf=0.2)
                for plate_result in plate_results:
                    boxes = plate_result.boxes.xyxy.cpu().numpy()
                    scores = plate_result.boxes.conf.cpu().numpy()
                    classes = plate_result.boxes.cls.cpu().numpy()
                    for box, score, cls in zip(boxes, scores, classes):
                        if score > 0.2 and cls == 0:
                            x1, y1, x2, y2 = box[:4]
                            plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
                            plate_text = detect_characters(plate_img)

                            if plate_text and plate_text not in detected_strings:
                                detected_strings.add(plate_text)
                                detection_time = datetime.datetime.now()
                                image_data = base64.b64encode(cv2.imencode('.jpg', plate_img)[1]).decode()
                                detection_entry = {
                                    "image_data": image_data,
                                    "plate_text": plate_text,
                                    "timestamp": detection_time.strftime('%Y-%m-%d %H:%M:%S')
                                }
                                detection_entries.append(detection_entry)
                                socketio.emit('detection', detection_entry)

            frames_processed += 1

        cap.release()

        if detection_entries:
            new_df = pd.DataFrame(detection_entries)
            try:
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_excel(excel_file_path, index=False, engine='openpyxl')
            except Exception as e:
                logging.error(f"Error saving Excel file: {e}")
                return jsonify({"error": "Error saving Excel file"}), 500

        return jsonify({"message": "Detection complete", "detected_strings": list(detected_strings)}), 200

    thread = threading.Thread(target=generate)
    thread.start()
    return jsonify({"status": "Processing started"})


@app.route('/get_detections')
def get_detections():
    df = pd.read_excel(excel_file_path)
    detections = df.to_dict(orient='records')
    return jsonify(detections)

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global stop_detection_flag
    stop_detection_flag = True
    return jsonify({"message": "Detection stopped successfully"}), 200

@app.route('/delete_detections', methods=['DELETE'])
def delete_detections():
    df = pd.DataFrame(columns=["plate_text", "timestamp"])
    df.to_excel(excel_file_path, index=False)
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f'Failed to delete {file_path}. Reason: {e}')
    return jsonify({"message": "All detections deleted successfully"}), 200

@socketio.on('connect')
def connect():
    emit('message', {'data': 'Connected'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
