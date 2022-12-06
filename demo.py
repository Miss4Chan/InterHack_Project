import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image,ImageFilter
import dlib
import time
import math
from shapely.geometry import Point, Polygon
import plate_recognition as pr


carCascade = cv2.CascadeClassifier('cars.xml')
#print(cv2.CascadeClassifier.empty(carCascade))
video = cv2.VideoCapture("gd-demo.mp4") 

x1,x2,x3,x4 = 520, 600, 1100,1300
y1,y2 = 530, 700

coord=[[x1,y2],[x2,y1],[x4,y2],[x3,y1]]
dist = 3


def GetLicenseSegment(img):
    #Should be from video, will change later 
    #img = cv2.imread(path,cv2.IMREAD_COLOR) #read image and convert to match resolution we need
    #img = cv2.resize(img, (620,480))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(img, 11, 17, 17) 
    edged = cv2.Canny(gray, 30, 200)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(cnts) > 0):
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        
        mask = getMask(cnts,img,gray)

        if(mask.any()!=None):
            (x, y) = np.where(mask == 255)
        if len(x) == 0 or len(y) == 0:
            return
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]
        return Cropped
    else:
        return

def getMask(cnts,img,gray):
    screenCnt = None
    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen

        if len(approx) == 4:
            screenCnt = approx
            break
        if screenCnt is None:
            detected = 0
            #print ("No contour detected")
        else:
            detected = 1
        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape,np.uint8)
    #new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)       
    #new_image = cv2.bitwise_and(img,img,mask=mask)
    return mask 
        
def GetTextFromPlate(croppedImg):
    #Read the number plate 
    if croppedImg is None:
        return ""
    text = pytesseract.image_to_string(croppedImg, config='--psm 11')
    return text

#estimate speed function
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

def GetSingleCarImg(coordinates,x,y,w,h,img):
    poly = Polygon(coordinates)
    central_point = Point(x+w//2,y+h//2)
    if poly.contains(central_point):
        single_car_img = img[y:y+h,x:x+w]
        cropped = GetLicenseSegment(single_car_img)
        text = GetTextFromPlate(cropped)
        #print(text)

def trackMultipleObjects():

    while True:
        ret, img = video.read()
        img = cv2.resize(img,(1920,1080))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cars=carCascade.detectMultiScale(gray,1.25,3)

        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2) 

        cv2.line(img, (coord[0][0],coord[0][1]),(coord[1][0],coord[1][1]),(0,0,255),2)
        cv2.line(img, (coord[0][0],coord[0][1]), (coord[2][0],coord[2][1]), (0, 0, 255), 2)
        cv2.line(img, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)
        cv2.line(img, (coord[1][0],coord[1][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)

        for (x, y, w, h) in cars:
            tim1=0
            if(x>=coord[0][0] and y==coord[0][1]):
                cv2.line(img, (coord[0][0], coord[0][1]), (coord[1][0], coord[1][1]), (0, 255,0), 2)
                tim1= time.time() 
                print("Car Entered.")

            if (x>=coord[2][0] and y==coord[2][1]):
                cv2.line(img, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) 
                tim2 = time.time() 
                print("Car Left.")
                print("Speed in (m/s) is:", dist/((tim2-tim1)))

        cv2.imshow('img',img) #Shows the frame
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def GetLicenseFromSegment():
    croppedImg = GetLicenseSegment("") 
    text = GetTextFromPlate(croppedImg)
    print("The plate is: ",text)
    
def main(path):
    img=cv2.imread(path)
    cropped = GetLicenseSegment(img)
    text = GetTextFromPlate(cropped)
    print(text)

if __name__ == '__main__':
    #trackMultipleObjects()
    pr.img_to_string("test222.jpeg")