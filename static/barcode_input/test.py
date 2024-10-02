import torch
import cv2
from pyzbar import pyzbar 
import argparse

padding = 25

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom',path='model.pt').to(device)

cap = cv2.VideoCapture('http://192.168.0.100:8080/video')
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Resized Window', 640, 360)

def type_data(raw_data):
  print(raw_data)

def main_loop():
	while cap.isOpened():
		_, frame = cap.read()
		results = model(frame)
		detections = results.pandas().xyxy[0]

		for i, detection in detections.iterrows():
			x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
			x1, y1, x2, y2 = [round(num) for num in [x1, y1, x2, y2]]


			x1 = max(0, x1 - padding)
			y1 = max(0, y1 - padding)
			x2 = min(frame.shape[1], x2 + padding)
			y2 = min(frame.shape[0], y2 + padding)

			confidence = detection['confidence']

			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

			label = f'{"Barcode"} {confidence:.2f}'
			label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			cv2.rectangle(frame, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - baseline), (0, 255, 0), cv2.FILLED)
			cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
			
			cropped_img = frame[y1:y2, x1:x2]
			gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
			rotated_frame = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
			barcodes = pyzbar.decode(rotated_frame)

			for code in barcodes:
				x, y , width, height = code.rect
				cv2.rectangle(frame, (x, y),(x + width, y + height), (0, 255, 0), 2)
				
				data = code.data.decode('utf-8')
				type_data_debounced(data)

		cv2.imshow('Resized Window', frame)
		if cv2.waitKey(1) & 0xFF == 27:
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--cameraId', '-i', default=0, help='Index of the webcam to use (overwritten by --cameraPath)')
  parser.add_argument('--cameraDevice', '-d', help='Absolute path to the webcam device (overwrites --cameraId)')
  parser.add_argument('--printDelay', '-t', help='Delay until code is typed again')
  parser.add_argument('--appendEnter', action=argparse.BooleanOptionalAction, default=False, help='Press enter key when code is scanned')
  parser.add_argument('--type', action=argparse.BooleanOptionalAction, default=True, help='Type scanned code with a virtual keyboard')
  args=parser.parse_args()

  delay = 0
  if (args.printDelay):
    delay = int(args.printDelay)
  type_data_debounced = type_data

  
  main_loop()