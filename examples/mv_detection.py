import jetson.inference
import jetson.utils
import os, sys
import cv2

def app(filename):
  net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
  display = jetson.utils.glDisplay()
  t = "i"
  ext = os.path.splitext(filename)[1]
  ext = ext.lower()
  if ext == "":           # Image is from Camera
    print("Image from camera")
    camera = jetson.utils.gstCamera()
    t = "c"
  elif ext == ".mp4" or ext == ".mov" or ext == ".avi":     # Image is from video file
    print("image from video file")
    t = "v"
  else:                   # Image is from a image file
    print("video from image file")

  while display.IsOpen():
    if t == "c":
      img, width, height = camera.CaptureRGBA()
    if t == "i":
      img, width, height = jetson.utils.loadImageRGBA(filename)
    else:
      img, width, height = getFrame(filename)

    detections = net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
    display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

nframe = 0
vid = None

def getFrame(filename):
  global nframe
  global vid
  if nframe == 0:
    vid = cv2.VideoCapture(filename)
  nframe = nframe + 1
  ret_val, frame = vid.read()
  frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
  w = frame.shape[1]
  h = frame.shape[0]
  img = jetson.utils.cudaFromNumpy(frame_rgba)
  return img, w,h

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("missing filename to open")
    exit(1)
  print("filename: " + sys.argv[1])
  app(sys.argv[1])
  
