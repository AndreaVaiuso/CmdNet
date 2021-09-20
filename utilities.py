import matplotlib.pyplot as plt
from skimage import transform
import os
import cv2
import math

def toBool(val):
  if val == "1" or val.lower() == "true" or val.lower() == "y" or val.lower() == "yes" or val.lower() == "t": return 1
  if val == "0" or val.lower() == "false" or val.lower() == "n" or val.lower() == "no" or val.lower() == "f": return 0
  return -1

def secToTime(n):
    day = n // (24 * 3600)
    n = n % (24 * 3600)
    hour = n // 3600
    n %= 3600
    minutes = n // 60
    n %= 60
    seconds = math.ceil(n)
    return str(math.ceil(day))+"d:"+str(math.ceil(hour))+"h:"+str(math.ceil(minutes))+"m:"+str(seconds)+"s"

def imgpad(im,h,w):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(h)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = w - new_size[1]
  delta_h = h - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)
  color = [0, 0, 0]
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
  return new_im

def drawCurves(history):
    #Summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","test"],loc="upper left")
    plt.show()
    #Summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def summarizeAccuracy(history,mname):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","validation"],loc="upper left")
    plt.show()

def summarizeLoss(history,mname):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def reshape_image(image, row, col, channel):
    return transform.resize(image,(row, col, channel))

from PIL import Image
import numpy as np
from skimage import transform
def load_image(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (150, 150, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def printProgressBar(iteration, total, prefix = '>', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print(f'\r{prefix} {percent}% [{bar}] {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def listFileInDir(dir,extension="",prnt=0):
  dirlist = []
  dirstr,i = "\n",1
  for dir in os.listdir(dir):
    if(dir.endswith(extension)): 
      dirlist.append(dir)
      dirstr += "<" + str(i) + "> " + dir + "\n"
      i += 1
  if prnt: printBorder(dirstr)
  return dirlist

def qry(question):
  resp = ""

  while(resp.lower()!="y" and resp.lower()!="n"):
    resp = input(question+" (y/n): ")
  if resp=="n": return 0
  else: return 1

def printBorder(msg):
    msg_lines=msg.split('\n')
    max_length=max([len(line) for line in msg_lines])
    count = max_length +2 

    dash = "*"*count 
    print("*%s*" %dash)

    for line in msg_lines:
        half_dif=(max_length-len(line))
        if half_dif==0:
            print("* %s *"%line)
        else:
            print("* %s "%line+' '*half_dif+'*')    
    bordermsg = "*%s*"%dash
    print(bordermsg)
    return bordermsg

def isFloat(str_val):
  try:
    float(str_val)
    return True
  except ValueError:
    return False

def isInt(str_val):
  try:
    int(str_val)
    return True
  except ValueError:
    return False

def isBoolean(str_val):
  try:
    bool(str_val)
    return True
  except ValueError:
    return False