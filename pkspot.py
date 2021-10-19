import cv2
import os
import sys
import xml.etree.ElementTree as ET

def main():
    xmlfile = sys.argv[1]
    imgfile = sys.argv[2]
    img_array = cv2.imread(imgfile)
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    for child in root:
        print("PATCH")
        for patch in child:
            print("\t",patch.tag,":",patch.text)

    
if __name__ == "__main__":
    main()