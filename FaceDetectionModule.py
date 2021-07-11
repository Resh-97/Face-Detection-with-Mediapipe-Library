import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture('Videos/2.mp4')
#cam = cv2.VideoCapture(0)

class faceDetection:

    def __init__(self, detectionConfidence = 0.5, model_selection=0):
        self.detectionConfidence = detectionConfidence
        self.model_selection = model_selection

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.detectionConfidence, self.model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw = True):
        imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(results.detections)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                score = detection.score[0]
                bboxC = detection.location_data.relative_bounding_box
                height, width, channel = img.shape
                bbox = int(bboxC.xmin*width), int(bboxC.ymin*height), int(bboxC.width*width), int(bboxC.height*height)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(img, f'{int(score*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255),3)

        return img, bboxs

    def fancyDraw(self, img, bbox, length=20, thickness =5, rectThickness = 1):
        x,y,w,h = bbox
        x1,y1 = x+w, y+h
        cv2.rectangle(img,bbox,(255,0,255),rectThickness)
        #Topleft x,y
        cv2.line(img,(x,y),(x+length, y),(255,0,255),thickness)
        cv2.line(img,(x,y),(x, y+length),(255,0,255),thickness)

        #Top Right x1,y
        cv2.line(img,(x1,y),(x1-length, y),(255,0,255),thickness)
        cv2.line(img,(x1,y),(x1, y+length),(255,0,255),thickness)

        #Bottomleft x,y1
        cv2.line(img,(x,y1),(x+length, y1),(255,0,255),thickness)
        cv2.line(img,(x,y1),(x, y1-length),(255,0,255),thickness)

        #Bottom Right x1,y1
        cv2.line(img,(x1,y1),(x1-length, y1),(255,0,255),thickness)
        cv2.line(img,(x1,y1),(x1, y1-length),(255,0,255),thickness)
        return img

def main():
    prevTime = 0
    currTime = 0
    detector = faceDetection(detectionConfidence =0.75)
    while True:
        success, img = cam.read()
        #img = cv2.resize(img, (640,480))
        img = cv2.resize(img,(640,480))
        img , bboxs = detector.findFaces(img, draw =True)
        print(bboxs)

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()
