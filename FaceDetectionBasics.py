import cv2
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils


cam = cv2.VideoCapture('Videos/2.mp4')
#cam = cv2.VideoCapture(0)
prevTime = 0
currTime = 0

while True:
    success, img = cam.read()
    #img = cv2.resize(img,(480,640))
    img = cv2.resize(img,(640,480))
    imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            ##This is the default drawing function
            #mpDraw.draw_detection(img, detection)

            ## But now we are attempting to draw ourselves.
            #print(id,detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            score = detection.score[0]
            bboxC = detection.location_data.relative_bounding_box
            height, width, channel = img.shape
            bbox = int(bboxC.xmin*width), int(bboxC.ymin*height), int(bboxC.width*width), int(bboxC.height*height)
            cv2.putText(img, f'{int(score*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255),3)
            cv2.rectangle(img,bbox,(255,0,255),2)



    currTime =time.time()
    fps = 1/ (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
