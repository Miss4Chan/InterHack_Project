import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image,ImageFilter
import dlib
import time
import math
from shapely.geometry import Point, Polygon

#Global vars
carCascade = cv2.CascadeClassifier('/Users/despinamisheva/InterHack_Project-1/cars.xml') #Haar classifier for cars
print(cv2.CascadeClassifier.empty(carCascade))
video = cv2.VideoCapture("//Users/despinamisheva/InterHack_Project-1/VID_20221203_160141.mp4") #testing video can change or be placed elsewhere
#Used for showing output should change 
#Coordinates of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

#4K Vid
#x1,x2,x3,x4 = 680,890,2900,3400
#y1,y2= 1700,2150

#FHD Vid
x1,x2,x3,x4 = 520, 600, 1100,1300
y1,y2 = 530, 700

coord=[[x1,y2],[x2,y1],[x4,y2],[x3,y1]]
#Distance between two horizontal lines in (meter)
dist = 3



def GetLicenseSegment(img):
    #Should be from video, will change later 
    #img = cv2.imread(path,cv2.IMREAD_COLOR) #read image and convert to match resolution we need
    #img = cv2.resize(img, (620,480))
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
    #convert to grey scale
    gray = cv2.bilateralFilter(img, 11, 17, 17) #Blur to reduce noise

    # Sobel Edge Detection
    # sobelx = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
    # sobely = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
    # sobelxy = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3) # Combined X and Y Sobel Edge Detection




    # cv2.imshow("x",sobelx)
    # cv2.imshow("y",sobely)
    # cv2.imshow("xy",sobelxy)
    # cv2.waitKey(0)

    # cv2.imwrite("gray.png",gray)

    # image = Image.open("gray.png",'r')
    # image = image.convert("L")

    # edged  = image.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0)) #Perform Edge detection
    
    # edged.save("edged.png")
    # edged = cv2.imread("edged.png")
    edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
    #cv2.imshow("x",edged)
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour=
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(cnts) > 0):
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        
        mask = getMask(cnts,img,gray)
        #new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        #new_image = cv2.bitwise_and(img,img,mask=mask)
        #Now crop
        if(mask.any()!=None):
            (x, y) = np.where(mask == 255)
        if len(x) == 0 or len(y) ==0:
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
            print ("No contour detected")
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

    # p1 = Point(x, y+h)
    # p2 = Point(x+w, y+h)
    # if poly.contains(p1) and poly.contains(p2):

    central_point = Point(x+w//2,y+h//2)
    if poly.contains(central_point):
        single_car_img = img[y:y+h,x:x+w]
        cropped = GetLicenseSegment(single_car_img)
        text = GetTextFromPlate(cropped)
        print(text)


#tracking multiple objects
def trackMultipleObjects():

    while True:
        ret, img = video.read()
        img = cv2.resize(img,(1920,1080))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cars=carCascade.detectMultiScale(gray,1.8,2)

        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2) #BGR
            #GetSingleCarImg(coord,x,y,w,h,gray)


        cv2.line(img, (coord[0][0],coord[0][1]),(coord[1][0],coord[1][1]),(0,0,255),2)  #First horizontal line
        cv2.line(img, (coord[0][0],coord[0][1]), (coord[2][0],coord[2][1]), (0, 0, 255), 2) #Vertical left line
        cv2.line(img, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Second horizontal line
        cv2.line(img, (coord[1][0],coord[1][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Vertical right line

        for (x, y, w, h) in cars:
            tim1=0
            if(x>=coord[0][0] and y==coord[0][1]):
                cv2.line(img, (coord[0][0], coord[0][1]), (coord[1][0], coord[1][1]), (0, 255,0), 2) #Changes line color to green
                tim1= time.time() #Initial time
                print("Car Entered.")

            if (x>=coord[2][0] and y==coord[2][1]):
                cv2.line(img, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Changes line color to green
                tim2 = time.time() #Final time
                print("Car Left.")
                #We know that distance is 3m
                print("Speed in (m/s) is:", dist/((tim2-tim1)))

        cv2.imshow('img',img) #Shows the frame
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def GetLicenseFromSegment():
    croppedImg = GetLicenseSegment("") #each car individually
    text = GetTextFromPlate(croppedImg)
    print("The plate is: ",text)
    
def main(path):
    img=cv2.imread(path)
    cropped = GetLicenseSegment(img)
    text = GetTextFromPlate(cropped)
    print(text)


if __name__ == '__main__':
    #trackMultipleObjects()
    main("/Users/despinamisheva/InterHack_Project-1/test2.jpeg")