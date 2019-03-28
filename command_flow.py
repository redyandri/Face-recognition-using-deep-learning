from Victorinox import Victorinox
import csv
import numpy as np
import pandas as pd
from shutil import copyfile
import os

tool=Victorinox()

# tool.gatherPhotoByNIP(nip_csv="/home/andri/miniconda3/envs/python2.7_env/app/mofface/data/pusintek_employee",
#                       img_folder="/home/andri/Documents/data/HRISPhoto",
#                       dest_flder="/home/andri/Documents/data/pusintek_employee")

photo_dir="/home/andri/Documents/data/psi_pusintek"
dest_folder="/home/andri/miniconda3/envs/python2.7_env/app/mofface/images"
haar="/home/andri/miniconda3/envs/python2.7_env/app/mofface/haarcascade_frontalface_default.xml"
for dirpath, dirs, files in os.walk(photo_dir):
    for f in files:
        fp=os.path.join(dirpath,f)
        tool.sampleFacesByPhoto(photo_path=fp,
                                dest_folder=dest_folder,
                                haarcascade_xml_path=haar)
        print "crop %s"%str(f)