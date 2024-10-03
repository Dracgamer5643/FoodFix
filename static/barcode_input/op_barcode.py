import argparse
import time

import cv2
from pyautogui import press, typewrite
from pyzbar import pyzbar

# Installation Guide of pyzbar: https://pypi.org/project/pyzbar/


# HACK I am too lazy to allow different timestamps to allow multiple debounced functions
last_execution_s = 0
def debounce(function, delay=3):
  def debounced_function(data):
    global last_execution_s

    time_s = int(time.time())
    if ((time_s - last_execution_s) < delay): return
    last_execution_s = time_s

    function(data)

  return debounced_function


def type_data(raw_data):
  print(raw_data)
  if (args.type):
    typewrite(raw_data)
    if (args.appendEnter):
      press('enter')


def find_codes(frame):
  codes = pyzbar.decode(frame)
    
  for code in codes:
    x, y , width, height = code.rect
    cv2.rectangle(frame, (x, y),(x + width, y + height), (0, 255, 0), 2)
    
    data = code.data.decode('utf-8')
    type_data_debounced(data)
    
  return frame


def main_loop():
  cameraPath = None
  if (args.cameraDevice):
    cameraPath = args.cameraDevice
  else:
    cameraPath = int(args.cameraId)
  
  camera = cv2.VideoCapture(0)

  ret, frame = camera.read()

  try:
    while ret:
      ret, frame = camera.read()
      frame = find_codes(frame)
      cv2.imshow('Scanner', frame)
      cv2.waitKey(1)
  except Exception as e:
    print(e)
    camera.release()
    cv2.destroyAllWindows()

  camera.release()
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--cameraId', '-i', default=0, help='Index of the webcam to use (overwritten by --cameraPath)')
  parser.add_argument('--cameraDevice', '-d', help='Absolute path to the webcam device (overwrites --cameraId)')
  parser.add_argument('--printDelay', '-t', help='Delay until code is typed again')
  parser.add_argument('--appendEnter', action=argparse.BooleanOptionalAction, default=False, help='Press enter key when code is scanned')
  parser.add_argument('--type', action=argparse.BooleanOptionalAction, default=True, help='Type scanned code with a virtual keyboard')
  args=parser.parse_args()

  # HACK
  delay = 3
  if (args.printDelay):
    delay = int(args.printDelay)
  type_data_debounced = debounce(type_data, delay=delay)

  main_loop()