from flask import Flask, render_template, Response, redirect, url_for
import cv2
import torch
from pyzbar import pyzbar
import argparse
import threading
import time
from collections import deque

app = Flask(__name__)
camera = cv2.VideoCapture(0) 
camera_active = None
model = None

class FrameProcessor:
    def __init__(self, queue_size=5):
        self.frame_queue = deque(maxlen=queue_size)
        self.result_queue = deque(maxlen=queue_size)
        self.processing = False

    def add_frame(self, frame):
        if len(self.frame_queue) < self.frame_queue.maxlen:
            self.frame_queue.append(frame)

    def process_frames(self):
        while self.processing:
            if len(self.frame_queue) > 0:
                frame = self.frame_queue.popleft()
                results = model(frame)
                self.result_queue.append((frame, results))
            else:
                time.sleep(0.01)

    def start(self):
        self.processing = True
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.start()

    def stop(self):
        self.processing = False
        self.thread.join()

    def get_result(self):
        if len(self.result_queue) > 0:
            return self.result_queue.popleft()
        return None, None

def decode_barcodes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rotated_frame = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    barcodes = pyzbar.decode(rotated_frame)
    return barcodes

@app.route('/dashboard')
def dashboard():
    page = "dashboard"
    return render_template('dashboard.html', page=page)

@app.route('/homepage')
def userpage():
    page = "userpage"
    return render_template('userpage.html', page=page)

@app.route("/")
def login():
    return render_template('login.html')

@app.route("/signup")
def signup():
    return render_template('Register.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    if not camera_active:
        camera.open(0) 
        camera_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
   
@app.route('/get_video_url')
def get_video_url():
    return url_for('video_feed', _external=True)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, model
    camera_active = False
    model = None
    camera.release()  
    return redirect(url_for('userpage'))

def generate_frames():
    global model
    padding = 25
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./static/barcode_input/model.pt').to(device)

    frame_processor = FrameProcessor()
    frame_processor.start()

    last_barcode_time = 0
    barcode_cooldown = 1.0  # seconds
    target_fps = 30
    frame_time = 1.0 / target_fps

    while camera_active == True:
     
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 360))

            frame_processor.add_frame(frame)
            processed_frame, results = frame_processor.get_result()


            if processed_frame is not None:
                detections = results.pandas().xyxy[0]

                for i, detection in detections.iterrows():
                    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
                    x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]

                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(processed_frame.shape[1], x2 + padding)
                    y2 = min(processed_frame.shape[0], y2 + padding)

                    confidence = detection['confidence']

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f'{"Barcode"} {confidence:.2f}'
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(processed_frame, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - baseline), (0, 255, 0), cv2.FILLED)
                    cv2.putText(processed_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    cropped_img = processed_frame[y1:y2, x1:x2]
                    barcodes = decode_barcodes(cropped_img)

                    current_time = time.time()
                    if current_time - last_barcode_time > barcode_cooldown:
                        for code in barcodes:
                            data = code.data.decode('utf-8')
                            print(data)
                            last_barcode_time = current_time

                _, buffer = cv2.imencode('.jpg', processed_frame)
                processed_frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')
                    
            time.sleep(max(0, frame_time - (time.time() - last_barcode_time)))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cameraId', '-i', default=0, help='Index of the webcam to use (overwritten by --cameraPath)')
    parser.add_argument('--cameraDevice', '-d', help='Absolute path to the webcam device (overwrites --cameraId)')
    parser.add_argument('--printDelay', '-t', help='Delay until code is typed again')
    parser.add_argument('--appendEnter', action=argparse.BooleanOptionalAction, default=False, help='Press enter key when code is scanned')
    parser.add_argument('--type', action=argparse.BooleanOptionalAction, default=True, help='Type scanned code with a virtual keyboard')
    args = parser.parse_args()

    delay = 0
    if (args.printDelay):
        delay = int(args.printDelay)

    app.run(port=8080, debug=True, threaded=True)
