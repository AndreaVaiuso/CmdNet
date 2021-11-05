import cv2
import sys
from time import sleep

def readStream(file:str,camera,sleeptime=1):
    if(camera.isnumeric()):
        cap = cv2.VideoCapture(0)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(file, frame)
            sleep(sleeptime)
        cap.release()
        cv2.destroyAllWindows()
    else:
        #CONNECT TO IP CAMERA AND GET FRAMES
        print("Connect to IP camera:", camera)

if __name__ == "__main__":
    print("Starting camera capture service")
    file = sys.argv[1]
    camera = sys.argv[2]
    readStream(file,camera)

