import cv2

def detectCars(srcName):
  img = cv2.imread(srcName)
  # OpenCV opens images as BRG
  # but we want it as RGB We'll
  # also need a grayscale version
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Use minSize because for not
  # bothering with extra-small
  # dots that would look like STOP signs
  objData = cv2.CascadeClassifier('cars.xml')
  found = objData.detectMultiScale(img_gray, 1.1, 1)

  # Don't do anything if there's
  # no sign
  amount_found = len(found)
  if amount_found != 0:
    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:
      # We draw a green rectangle around
      # every recognized sign
      cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

  cv2.imwrite('detected_' + srcName, img)

def detectBus(srcName):
  img = cv2.imread(srcName)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  objData = cv2.CascadeClassifier('bus_front.xml')
  found = objData.detectMultiScale(img_gray, 1.11, 1)

  amount_found = len(found)
  if amount_found != 0:
    for (x, y, width, height) in found:
      cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

  cv2.imwrite('detected_' + srcName, img)

detectCars('1.jpg')
detectBus('2.jpeg')
