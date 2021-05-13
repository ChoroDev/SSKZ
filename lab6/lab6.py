import cv2
import imutils
import numpy as np

def detectPedestrians(imageName):
  # Initializing the HOG person
  # detector
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
  # Reading the Image
  image = cv2.imread(imageName)
  image = imutils.resize(image, width=min(400, image.shape[1]))

  # Detecting all the regions in the
  # Image that has a pedestrians inside it
  (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)

  # Drawing the regions in the Image
  for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

  cv2.imwrite('detected_' + imageName, image)

detectPedestrians('1.jpg')
detectPedestrians('2.jpg')
detectPedestrians('3.jpg')
