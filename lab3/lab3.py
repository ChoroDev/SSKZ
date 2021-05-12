import cv2

outputFileName = "lab3_out.mp4"
srcFileName = 'lab3.mp4'

# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
faceFrontCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read videofile
cap = cv2.VideoCapture(srcFileName)
# Height, width, FPS
width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
colored = True
out = cv2.VideoWriter()
out.open(outputFileName, fourcc, fps, (width, height), colored)

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    faces = faceFrontCascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Write changed frame
    out.write(frame)

    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
  else:
    break

cap.release()
out.release()
