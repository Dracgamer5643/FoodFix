import torch
import cv2
from pyzbar import pyzbar
import argparse
import threading
import time
from collections import deque

padding = 25
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt').to(device)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Window', 640, 360)

def type_data(raw_data):
    print(raw_data)

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

def main_loop():
    frame_processor = FrameProcessor()
    frame_processor.start()

    last_barcode_time = 0
    barcode_cooldown = 1.0  # seconds
    target_fps = 30
    frame_time = 1.0 / target_fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
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

                # Decode barcodes only in the detected region
                cropped_img = processed_frame[y1:y2, x1:x2]
                barcodes = decode_barcodes(cropped_img)

                current_time = time.time()
                if current_time - last_barcode_time > barcode_cooldown:
                    for code in barcodes:
                        data = code.data.decode('utf-8')
                        type_data(data)
                        last_barcode_time = current_time

            cv2.imshow('Resized Window', processed_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Implement frame rate limiting
        time.sleep(max(0, frame_time - (time.time() - last_barcode_time)))

    frame_processor.stop()
    cv2.destroyAllWindows()

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

    main_loop()