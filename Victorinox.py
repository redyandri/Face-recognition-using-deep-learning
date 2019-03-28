import csv
import numpy as np
import pandas as pd
from shutil import copyfile
import os
import cv2

class Victorinox(object):

    def __init__(self):
        return

    def gatherPhotoByNIP(self,nip_csv,img_folder,dest_flder):
        df=pd.read_csv(nip_csv)
        nip=np.array(df)
        nip=nip.tolist()
        nip2=[]
        for n in nip:
            for m in n:
                nip2.append(str(m))
        found=[]
        for dir,folders, files in os.walk(img_folder):
            if len(files)>0:
                c=0
                for f in files:
                    f2="".join(f.split(".")[0:-1])
                    if nip2.__contains__(f2):
                        found.append(f2)
                        src=os.path.join(dir,f)
                        dst=os.path.join(dest_flder,f2+".jpg")
                        copyfile(src,dst)
                        c+=1
        print "copied %d files"%c
        print "not found: %s"%str(self.diffList(nip2,found))
        return

    def diffList(self, first, second):
        second = set(second)
        return [item for item in first if item not in second]\

###########################################
    def sampleFacesByCam(self,
                    haarcascade_xml_path="haarcascade_frontalface_default.xml",
                               wait_milisecons=100,
                            sample_num=10):
        cam = cv2.VideoCapture(0)

        face_detector = cv2.CascadeClassifier(haarcascade_xml_path)

        count = 0
        while (True):
            ret, img = cam.read()
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite("images/User_" + str(count) + ".jpg", img[y1:y2, x1:x2])
                cv2.imshow('image', img)
            k = cv2.waitKey(wait_milisecons) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= sample_num:  # Take 30 face sample and stop video
                break
        cam.release()
        cv2.destroyAllWindows()

###########################################
    def sampleFacesByPhoto(self,
                           photo_path="/a.jpg",
                           dest_folder="/a",
                    haarcascade_xml_path="haarcascade_frontalface_default.xml"):
        img=cv2.imread(photo_path,0)
        filename=os.path.basename(photo_path)
        nip=os.path.splitext(filename)
        face_detector = cv2.CascadeClassifier(haarcascade_xml_path)
        faces = face_detector.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # Save the captured image into the datasets folder
            dst=os.path.join(dest_folder,filename)
            cv2.imwrite(dst, img[y1:y2, x1:x2])
            cv2.imshow('image', img)